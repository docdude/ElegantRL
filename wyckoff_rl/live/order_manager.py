"""
Order Manager — translates target positions into IB orders.

Tracks current position, computes deltas, submits MKT orders,
and provides risk controls (max position, daily loss limit).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single executed trade."""
    timestamp: float
    action: str        # "BUY" or "SELL"
    quantity: int
    price: float       # fill price (or last known price)
    position_after: float
    pnl_realized: float = 0.0


class OrderManager:
    """
    Manages position transitions and risk controls.

    Parameters
    ----------
    ib_connector : IBConnector
        Connection to IB for order submission.
    max_contracts : int
        Maximum absolute position (default: 1 for paper trading).
    daily_loss_limit : float
        Maximum daily loss in USD before halting (default: 5000).
    cost_per_side : float
        Commission + slippage estimate per side per contract (USD).
    """

    def __init__(
        self,
        ib_connector,
        max_contracts: int = 1,
        daily_loss_limit: float = 5000.0,
        cost_per_side: float = 5.0,
    ):
        self.ib = ib_connector
        self.max_contracts = max_contracts
        self.daily_loss_limit = daily_loss_limit
        self.cost_per_side = cost_per_side

        # State
        self.current_position: float = 0.0  # in contracts (signed)
        self.entry_price: float = 0.0
        self.daily_pnl: float = 0.0
        self.total_pnl: float = 0.0
        self.trades: list[TradeRecord] = []
        self._halted = False

    def sync_position(self):
        """Sync position from IB."""
        self.current_position = self.ib.get_position()
        logger.info(f"Synced position from IB: {self.current_position}")

    def set_target_position(self, target: float, last_price: float) -> Optional[TradeRecord]:
        """
        Move to target position.

        Parameters
        ----------
        target : float
            Target position in contracts (e.g., +1, 0, -1).
        last_price : float
            Current market price for PnL estimation.

        Returns
        -------
        TradeRecord or None if no action needed.
        """
        if self._halted:
            logger.warning("Trading halted due to daily loss limit")
            return None

        # Clamp target
        target = max(-self.max_contracts, min(self.max_contracts, target))

        delta = int(round(target - self.current_position))
        if delta == 0:
            return None

        # Estimate PnL on position close/reduction
        pnl = 0.0
        if self.current_position != 0 and abs(target) < abs(self.current_position):
            contracts_closed = abs(self.current_position) - abs(target)
            price_change = last_price - self.entry_price
            pnl = contracts_closed * price_change * (1 if self.current_position > 0 else -1)
            # NQ multiplier: $20 per point
            pnl *= 20.0
            # Subtract costs
            pnl -= contracts_closed * self.cost_per_side * 2  # round trip

        # Submit order
        trade = self.ib.place_order(delta)
        if trade is None:
            return None

        action = "BUY" if delta > 0 else "SELL"
        self.daily_pnl += pnl
        self.total_pnl += pnl

        # Update position tracking
        old_pos = self.current_position
        self.current_position = target
        if target != 0:
            self.entry_price = last_price

        record = TradeRecord(
            timestamp=time.time(),
            action=action,
            quantity=abs(delta),
            price=last_price,
            position_after=target,
            pnl_realized=pnl,
        )
        self.trades.append(record)
        logger.info(f"Trade: {action} {abs(delta)} @ ~{last_price:.2f} | "
                     f"pos: {old_pos} → {target} | realized: ${pnl:.2f} | "
                     f"daily P&L: ${self.daily_pnl:.2f}")

        # Check daily loss limit
        if self.daily_pnl < -self.daily_loss_limit:
            self._halted = True
            logger.warning(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:.2f} "
                           f"(limit: -${self.daily_loss_limit:.2f})")

        return record

    def reset_daily(self):
        """Reset daily P&L tracking (call at session start)."""
        self.daily_pnl = 0.0
        self._halted = False
        logger.info("Daily P&L reset")

    @property
    def halted(self) -> bool:
        return self._halted

    @property
    def n_trades(self) -> int:
        return len(self.trades)
