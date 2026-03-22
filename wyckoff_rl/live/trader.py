"""
Wyckoff Trader — adapter-driven trading system.

Pipeline: DataAdapter → RangeBarBuilder → LiveFeatureEngine → InferenceEngine → OrderAdapter

Three modes:
  replay   SCID file → SimExecutor         (offline testing, any time)
  live     IB ticks  → IB orders           (market hours, paper or real)
  tail     SC .scid  → IB orders           (SC provides data, IB executes)

Usage:
    # Replay (offline — no broker needed)
    python -m wyckoff_rl.live.trader replay \\
        --checkpoint path/to/actor.pt \\
        --scid /opt/SierraChart/Data/NQH26-CME.scid \\
        --start-date 2026-03-18

    # Live via IB
    python -m wyckoff_rl.live.trader live \\
        --checkpoint path/to/actor.pt \\
        --expiry 20260618

    # Tail SC file + IB execution
    python -m wyckoff_rl.live.trader tail \\
        --checkpoint path/to/actor.pt \\
        --scid /opt/SierraChart/Data/NQM26-CME.scid \\
        --expiry 20260618
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .range_bar_builder import RangeBarBuilder
from .live_features import LiveFeatureEngine, TRAINING_FEATURE_INDICES
from .inference import InferenceEngine
from .adapters import (
    DataAdapter, OrderAdapter,
    SCIDReplayAdapter, IBLiveAdapter, SCIDTailAdapter,
    SimExecutor, IBExecutor,
)
from ..feature_config import ALL_FEATURES

logger = logging.getLogger("wyckoff_trader")

# Feature name → index in the training-feature vector (36 features)
FEATURE_NAME_TO_IDX = {
    ALL_FEATURES[all_idx]: train_idx
    for train_idx, all_idx in enumerate(TRAINING_FEATURE_INDICES)
}

# Pre-validated veto presets discovered via analyze_filter_candidates.py
# Each preset: list of (feature_name, operator, threshold) — OR logic
VETO_PRESETS = {
    "split3": [
        ("delta_ratio", "<", -0.044),
        ("cvd_slope_fast", ">", 0.037),
    ],
    "split4": [
        ("wave_shortening_up", ">", 0.264),
        ("cvd_divergence", ">", 0.284),
    ],
    "wavenet_s0": [
        ("wave_vol_trend_up", "<", -0.186),
        ("delta_ratio", "<", -0.064),
    ],
}


class VetoFilter:
    """
    Entry veto filter — blocks new position entries when any rule fires.

    Rules are OR'd: if ANY rule matches, the entry is vetoed.
    Only vetoes entries (from flat); never prevents exits.
    """

    def __init__(self, rules: list[tuple[str, str, float]]):
        self._rules = []
        for feat_name, op, thresh in rules:
            idx = FEATURE_NAME_TO_IDX.get(feat_name)
            if idx is None:
                raise ValueError(f"Unknown feature: {feat_name}")
            if op not in (">", "<"):
                raise ValueError(f"Operator must be '>' or '<', got '{op}'")
            self._rules.append((feat_name, idx, op, thresh))

    def should_veto(self, features) -> tuple[bool, str]:
        """Check if entry should be vetoed based on latest bar features."""
        for feat_name, idx, op, thresh in self._rules:
            val = float(features[idx])
            if (op == ">" and val > thresh) or (op == "<" and val < thresh):
                return True, f"{feat_name}={val:.4f} {op} {thresh}"
        return False, ""

    def describe(self) -> str:
        return " OR ".join(f"{n} {o} {t}" for n, _, o, t in self._rules)


class WyckoffTrader:
    """
    Main trading orchestrator — adapter-driven.

    Wires: DataAdapter → RangeBarBuilder → FeatureEngine → Inference → OrderAdapter

    Parameters
    ----------
    checkpoint_path : str
        Path to the actor .pt checkpoint.
    data_adapter : DataAdapter
        Tick data source (SCID replay, IB live, or SCID tail).
    order_adapter : OrderAdapter
        Order execution backend (SimExecutor or IBExecutor).
    range_size : float
        Range bar size in points (40.0 for NQ).
    window_size : int
        Sliding window bars (30).
    feature_indices : list[int], optional
        Training feature column indices.
    continuous_sizing : bool
        Position sizing mode.
    initial_amount : float
        For PnL normalization in agent state.
    max_contracts : int
        Max position size.
    log_dir : str
        Directory for trade logs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        data_adapter: DataAdapter,
        order_adapter: OrderAdapter,
        range_size: float = 40.0,
        window_size: int = 30,
        feature_indices: list[int] = None,
        continuous_sizing: bool = False,
        initial_amount: float = 1000.0,
        max_contracts: int = 1,
        log_dir: str = "live_logs",
        veto_rules: list[tuple[str, str, float]] | None = None,
    ):
        self.data_adapter = data_adapter
        self.order_adapter = order_adapter
        self.range_size = range_size
        self.window_size = window_size
        self.max_contracts = max_contracts
        self.log_dir = log_dir
        self.initial_amount = initial_amount
        self._running = False

        feat_idx = feature_indices or TRAINING_FEATURE_INDICES

        # 1. Range Bar Builder
        self.bar_builder = RangeBarBuilder(
            range_size=range_size,
            on_bar=self._on_bar_complete,
        )

        # 2. Feature Engine
        self.feature_engine = LiveFeatureEngine(
            buffer_size=200,
            feature_indices=feat_idx,
            reversal_points=range_size,
        )

        # 3. Inference Engine
        self.inference = InferenceEngine(
            checkpoint_path=checkpoint_path,
            window_size=window_size,
            n_features=len(feat_idx),
            continuous_sizing=continuous_sizing,
            initial_amount=initial_amount,
        )

        # Position / PnL tracking
        self._position: float = 0.0
        self._entry_price: float = 0.0
        self._last_bar_price: float = 0.0
        self._unrealized_pnl: float = 0.0
        self._cash: float = 0.0
        self._tick_count: int = 0
        self._bar_count: int = 0
        self._n_trades: int = 0
        self._n_vetoed: int = 0
        self._total_pnl: float = 0.0
        self._trade_log_path: str = ""

        # Veto filter
        self.veto_filter = VetoFilter(veto_rules) if veto_rules else None

        # Wire data adapter callback
        self.data_adapter.on_tick = self._on_tick

    def _on_tick(self, price: float, size: float, is_uptick: bool, timestamp: float):
        """Tick callback → bar builder."""
        self._tick_count += 1
        self.bar_builder.on_tick(price, size, is_uptick, timestamp)

        # Update unrealized PnL
        if self._position != 0:
            pos_sign = 1 if self._position > 0 else -1
            self._unrealized_pnl = (price - self._entry_price) * pos_sign

    def _on_bar_complete(self, bar):
        """Bar callback → features → inference → order."""
        self._bar_count += 1
        self._last_bar_price = bar.close

        # 1. Compute features
        features = self.feature_engine.add_bar(bar)
        if features is None:
            logger.debug(f"Bar #{self._bar_count}: insufficient data for features "
                         f"({self.feature_engine.n_bars} bars buffered)")
            return

        # 2. Push features into sliding window
        self.inference.push_features(features)

        if not self.inference.ready:
            logger.debug(f"Bar #{self._bar_count}: warming up "
                         f"({self.inference.bars_seen}/{self.window_size} bars)")
            return

        # 3. Run inference
        target_pos, raw_action = self.inference.get_action(
            position=self._position,
            unrealized_pnl=self._unrealized_pnl,
            cash=self._cash,
        )

        # Clamp target
        target_pos = max(-self.max_contracts, min(self.max_contracts, target_pos))

        logger.info(
            f"Bar #{self._bar_count}: close={bar.close:.2f} | "
            f"action={raw_action:+.4f} → target={target_pos:+.1f} | "
            f"pos={self._position:+.0f} | "
            f"ticks={self._tick_count}"
        )

        # 4. Veto filter: block new entries if any rule fires
        if (self.veto_filter is not None
                and self._position == 0 and target_pos != 0):
            veto, reason = self.veto_filter.should_veto(
                self.inference._window[-1]
            )
            if veto:
                logger.info(f"  ✘ VETO entry → {target_pos:+.1f} | {reason}")
                self._n_vetoed += 1
                return

        # 5. Execute position change
        delta = int(round(target_pos - self._position))
        if delta != 0:
            self._execute_trade(delta, bar.close, raw_action, bar)

    def _execute_trade(self, delta: int, price: float, raw_action: float, bar):
        """Execute a trade through the order adapter."""
        # Compute realized PnL on close/reduce
        pnl = 0.0
        new_pos = self._position + delta
        if self._position != 0:
            if (self._position > 0 and delta < 0) or (self._position < 0 and delta > 0):
                closed = min(abs(self._position), abs(delta))
                pnl_pts = (price - self._entry_price) * (1 if self._position > 0 else -1)
                pnl = closed * pnl_pts * 20.0  # NQ $20/point

        ok = self.order_adapter.place_order(delta, price)
        if not ok:
            return

        self._position = new_pos
        if new_pos != 0:
            self._entry_price = price
        self._n_trades += 1
        self._total_pnl += pnl
        self._cash += pnl / 20.0  # back to points for agent state

        action = "BUY" if delta > 0 else "SELL"
        if pnl != 0:
            logger.info(f"  → {action} {abs(delta)} @ {price:.2f} | "
                        f"PnL: ${pnl:+,.2f} | Total: ${self._total_pnl:+,.2f}")

        self._log_trade(action, abs(delta), price, new_pos, pnl, raw_action, bar)

    def _log_trade(self, action, qty, price, pos_after, pnl, raw_action, bar):
        """Append trade to CSV log."""
        if not self._trade_log_path:
            return
        row = {
            "timestamp": datetime.fromtimestamp(bar.timestamp, tz=timezone.utc).isoformat(),
            "bar_num": self._bar_count,
            "bar_close": bar.close,
            "raw_action": f"{raw_action:.6f}",
            "action": action,
            "quantity": qty,
            "position_after": pos_after,
            "pnl_realized_usd": f"{pnl:.2f}",
            "total_pnl_usd": f"{self._total_pnl:.2f}",
            "equity": f"{self.order_adapter.get_account_value():.2f}",
        }
        file_exists = os.path.exists(self._trade_log_path)
        with open(self._trade_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def start(self):
        """Start the trader. Blocks until data exhausted or stopped."""
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._trade_log_path = os.path.join(self.log_dir, f"trades_{ts}.csv")

        mode = type(self.data_adapter).__name__
        executor = type(self.order_adapter).__name__

        logger.info("=" * 60)
        logger.info("WYCKOFF TRADER")
        logger.info("=" * 60)
        logger.info(f"Mode:       {mode} → {executor}")
        logger.info(f"Range bars: {self.range_size}pt NQ")
        logger.info(f"Window:     {self.window_size} bars × "
                     f"{self.inference.n_features} features")
        logger.info(f"State dim:  {self.inference.state_dim}")
        logger.info(f"Sizing:     {'continuous' if self.inference.continuous_sizing else 'binary'}")
        logger.info(f"Max pos:    {self.max_contracts} contracts")
        if self.veto_filter:
            logger.info(f"Veto:       {self.veto_filter.describe()}")
        logger.info(f"Trade log:  {self._trade_log_path}")
        logger.info("=" * 60)

        # Sync position if order adapter supports it
        self._position = self.order_adapter.get_position()

        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            self.data_adapter.start()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        """Graceful shutdown."""
        self._running = False
        self.data_adapter.stop()

        # Flatten position
        if self._position != 0:
            logger.info(f"Flattening position: {self._position}")
            delta = -int(self._position)
            if self._last_bar_price > 0:
                self.order_adapter.place_order(delta, self._last_bar_price)

        # Summary
        logger.info("=" * 60)
        logger.info("SESSION SUMMARY")
        logger.info(f"  Ticks processed:  {self._tick_count:,}")
        logger.info(f"  Bars completed:   {self._bar_count}")
        logger.info(f"  Trades executed:  {self._n_trades}")
        if self._n_vetoed:
            logger.info(f"  Entries vetoed:   {self._n_vetoed}")
        logger.info(f"  Total P&L:        ${self._total_pnl:+,.2f}")
        logger.info(f"  Final equity:     ${self.order_adapter.get_account_value():,.2f}")
        logger.info(f"  Trade log:        {self._trade_log_path}")
        logger.info("=" * 60)

    def _signal_handler(self, signum, frame):
        logger.info(f"Signal {signum} received")
        self._running = False
        self.data_adapter.stop()


# ═════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════

def _load_env():
    """Load .env file if present."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip())


def main():
    _load_env()

    parser = argparse.ArgumentParser(
        description="Wyckoff NQ Trader — replay / live / tail modes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Replay one day of SCID data (no broker needed)
  python -m wyckoff_rl.live.trader replay \\
      --checkpoint checkpoints/actor.pt \\
      --scid /opt/SierraChart/Data/NQH26-CME.scid \\
      --start-date 2026-03-18

  # Live paper trading via IB TWS
  python -m wyckoff_rl.live.trader live \\
      --checkpoint checkpoints/actor.pt \\
      --expiry 20260618

  # Tail SC data file + IB execution
  python -m wyckoff_rl.live.trader tail \\
      --checkpoint checkpoints/actor.pt \\
      --scid /opt/SierraChart/Data/NQM26-CME.scid \\
      --expiry 20260618
""",
    )

    # — Subcommands ————————————————————————————————————————————————
    sub = parser.add_subparsers(dest="mode", required=True)

    # Common args
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--checkpoint", type=str, required=True,
                        help="Path to actor .pt checkpoint")
    common.add_argument("--range-size", type=float, default=40.0)
    common.add_argument("--continuous", action="store_true",
                        default=os.environ.get("CONTINUOUS_SIZING", "").lower() == "true")
    common.add_argument("--max-contracts", type=int, default=1)
    common.add_argument("--log-dir", type=str, default="live_logs")
    common.add_argument("--veto-preset", type=str, default=None,
                        choices=list(VETO_PRESETS.keys()),
                        help="Named veto filter preset (e.g. split3, split4)")

    # — replay ————————————————————————————————————————————————————
    p_replay = sub.add_parser("replay", parents=[common],
                              help="Replay SCID file with simulated fills")
    p_replay.add_argument("--scid", type=str, required=True,
                          help="Path to .scid file")
    p_replay.add_argument("--start-date", type=str, default=None,
                          help="Start date YYYY-MM-DD")
    p_replay.add_argument("--end-date", type=str, default=None,
                          help="End date YYYY-MM-DD")
    p_replay.add_argument("--speed", type=float, default=0.0,
                          help="Replay speed (0=max, 1=realtime, 10=10x)")
    p_replay.add_argument("--max-records", type=int, default=None,
                          help="Max records to replay")
    p_replay.add_argument("--initial-capital", type=float, default=250_000.0)

    # — live ——————————————————————————————————————————————————————
    p_live = sub.add_parser("live", parents=[common],
                            help="Live trading via IB TWS/Gateway")
    p_live.add_argument("--host", type=str,
                        default=os.environ.get("IB_HOST", "127.0.0.1"))
    p_live.add_argument("--port", type=int,
                        default=int(os.environ.get("IB_PORT", "7497")))
    p_live.add_argument("--client-id", type=int,
                        default=int(os.environ.get("IB_CLIENT_ID", "1")))
    p_live.add_argument("--expiry", type=str, required=True,
                        help="NQ contract expiry (e.g., 20260618)")

    # — tail ——————————————————————————————————————————————————————
    p_tail = sub.add_parser("tail", parents=[common],
                            help="Tail SC .scid file + IB execution")
    p_tail.add_argument("--scid", type=str, required=True,
                        help="Path to active .scid file")
    p_tail.add_argument("--poll-interval", type=float, default=0.1,
                        help="SCID poll interval in seconds")
    p_tail.add_argument("--host", type=str,
                        default=os.environ.get("IB_HOST", "127.0.0.1"))
    p_tail.add_argument("--port", type=int,
                        default=int(os.environ.get("IB_PORT", "7497")))
    p_tail.add_argument("--client-id", type=int,
                        default=int(os.environ.get("IB_CLIENT_ID", "1")))
    p_tail.add_argument("--expiry", type=str, required=True,
                        help="NQ contract expiry (e.g., 20260618)")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # — Build adapters based on mode ——————————————————————————————
    if args.mode == "replay":
        data_adapter = SCIDReplayAdapter(
            scid_path=args.scid,
            start_date=args.start_date,
            end_date=args.end_date,
            speed=args.speed,
            max_records=args.max_records,
        )
        order_adapter = SimExecutor(initial_capital=args.initial_capital)

    elif args.mode == "live":
        data_adapter = IBLiveAdapter(
            host=args.host, port=args.port,
            client_id=args.client_id, nq_expiry=args.expiry,
        )
        order_adapter = IBExecutor(data_adapter.ib)

    elif args.mode == "tail":
        data_adapter = SCIDTailAdapter(
            scid_path=args.scid,
            poll_interval=args.poll_interval,
        )
        # IB for execution only (SC handles data)
        from .ib_connector import IBConnector
        ib = IBConnector(
            host=args.host, port=args.port,
            client_id=args.client_id, nq_expiry=args.expiry,
        )
        ib.connect()
        order_adapter = IBExecutor(ib)

    veto_rules = VETO_PRESETS.get(args.veto_preset) if args.veto_preset else None

    trader = WyckoffTrader(
        checkpoint_path=args.checkpoint,
        data_adapter=data_adapter,
        order_adapter=order_adapter,
        range_size=args.range_size,
        continuous_sizing=args.continuous,
        max_contracts=args.max_contracts,
        log_dir=args.log_dir,
        veto_rules=veto_rules,
    )
    trader.start()


if __name__ == "__main__":
    main()
