"""
Interactive Brokers Connector — wraps ib_insync for NQ futures.

Handles:
  - Connection to TWS / IB Gateway
  - NQ futures contract resolution
  - Real-time tick subscription
  - Position queries
  - Market order submission
  - Account equity queries
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Callable, Optional

logger = logging.getLogger(__name__)

try:
    from ib_insync import IB, Future, MarketOrder, Trade, Ticker, util
    HAS_IB = True
except ImportError:
    HAS_IB = False
    logger.warning("ib_insync not installed — IB connector will not work. "
                   "Install with: pip install ib_insync")


class IBConnector:
    """
    Manages IB TWS/Gateway connection for NQ futures trading.

    Parameters
    ----------
    host : str
        TWS/Gateway host (default: 127.0.0.1).
    port : int
        TWS/Gateway port (7497=paper TWS, 7496=live TWS,
        4002=paper Gateway, 4001=live Gateway).
    client_id : int
        Unique client identifier.
    nq_expiry : str
        NQ futures contract expiry (e.g., '20260618' or '202606').
    on_tick : callable, optional
        Callback for tick data: (price, size, is_uptick, timestamp).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        nq_expiry: str = "",
        on_tick: Optional[Callable] = None,
    ):
        if not HAS_IB:
            raise ImportError("ib_insync is required. Install: pip install ib_insync")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.nq_expiry = nq_expiry
        self.on_tick = on_tick

        self.ib = IB()
        self.contract: Optional[Future] = None
        self.ticker: Optional[Ticker] = None
        self._last_price: float = 0.0

    def connect(self):
        """Connect to TWS/IB Gateway."""
        logger.info(f"Connecting to IB at {self.host}:{self.port} "
                     f"(client_id={self.client_id})")
        self.ib.connect(self.host, self.port, clientId=self.client_id)
        logger.info("Connected to IB")

        # Resolve NQ contract
        self.contract = Future("NQ", self.nq_expiry, "CME")
        qualified = self.ib.qualifyContracts(self.contract)
        if not qualified:
            raise RuntimeError(f"Could not qualify NQ contract: {self.contract}")
        self.contract = qualified[0]
        logger.info(f"Qualified contract: {self.contract}")

    def subscribe_ticks(self):
        """Subscribe to real-time tick-by-tick trade data."""
        if self.contract is None:
            raise RuntimeError("Must connect() first")

        self.ticker = self.ib.reqTickByTickData(
            self.contract, "AllLast", numberOfTicks=0, ignoreSize=False
        )
        # Register callback
        self.ticker.updateEvent += self._on_tick_event
        logger.info(f"Subscribed to tick data for {self.contract.localSymbol}")

    def _on_tick_event(self, ticker: Ticker):
        """Handle incoming tick data."""
        ticks = ticker.tickByTicks
        if not ticks:
            return
        tick = ticks[-1]  # latest tick

        price = tick.price
        size = tick.size
        timestamp = tick.time.timestamp() if tick.time else 0.0

        # Determine uptick/downtick from tick type
        # ib_insync: tickType '' for AllLast doesn't distinguish bid/ask
        # Use price vs last price as heuristic
        is_uptick = price >= self._last_price
        self._last_price = price

        if self.on_tick:
            self.on_tick(price, size, is_uptick, timestamp)

    def get_position(self) -> float:
        """Get current NQ position (signed contracts)."""
        positions = self.ib.positions()
        for pos in positions:
            if (pos.contract.symbol == "NQ" and
                pos.contract.secType == "FUT"):
                return float(pos.position)
        return 0.0

    def get_account_value(self) -> float:
        """Get current account net liquidation value."""
        for av in self.ib.accountValues():
            if av.tag == "NetLiquidation" and av.currency == "USD":
                return float(av.value)
        return 0.0

    def place_order(self, quantity: int) -> Optional[Trade]:
        """
        Submit a market order.

        Parameters
        ----------
        quantity : int
            Positive = buy, negative = sell. Zero = no action.

        Returns
        -------
        Trade or None
        """
        if quantity == 0:
            return None
        action = "BUY" if quantity > 0 else "SELL"
        order = MarketOrder(action, abs(quantity))
        trade = self.ib.placeOrder(self.contract, order)
        logger.info(f"Order placed: {action} {abs(quantity)} NQ → "
                     f"order_id={trade.order.orderId}")
        return trade

    def disconnect(self):
        """Disconnect from IB."""
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("Disconnected from IB")

    def sleep(self, seconds: float = 0.0):
        """Process IB messages for a duration."""
        self.ib.sleep(seconds)

    @property
    def connected(self) -> bool:
        return self.ib.isConnected()
