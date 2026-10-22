# executor.py -- Hyperliquid order placement with retries and slippage guard.
"""The private key is read from .env ONLY. It is never logged, never printed,
never written to state.json. All failures raise after N retries; nothing is
silently ignored, because a silently failed order is the worst outcome here.

DRY_RUN=true routes every order to _simulate_order(), so the whole loop can run
for a week without a funded wallet.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()                      # reads .env in the working directory
log = logging.getLogger(__name__)


@dataclass
class Fill:
    coin: str
    side: str                      # "buy" | "sell"
    size: float
    price: float
    order_id: str | None = None
    simulated: bool = False


class HyperliquidExecutor:
    def __init__(self, dry_run: bool = True, retries: int = 3, backoff: float = 2.0):
        self.dry_run = dry_run
        self.retries = retries
        self.backoff = backoff
        self.client = None
        if not dry_run:
            self.client = self._connect()

    # ------------------------------------------------------------------ connect
    def _connect(self):
        key = os.getenv("HL_PRIVATE_KEY")
        wallet = os.getenv("HL_WALLET_ADDRESS")
        if not key or not wallet:
            raise RuntimeError("HL_PRIVATE_KEY / HL_WALLET_ADDRESS missing from .env")
        from eth_account import Account
        from hyperliquid.exchange import Exchange
        from hyperliquid.utils import constants

        acct = Account.from_key(key)          # key never leaves this scope
        if acct.address.lower() != wallet.lower():
            raise RuntimeError("private key does not match HL_WALLET_ADDRESS")
        log.info("connected to Hyperliquid as %s", acct.address)
        return Exchange(acct, constants.MAINNET_API_URL)

    # ------------------------------------------------------------------ helpers
    def _retry(self, fn, *args, **kwargs):
        last = None
        for attempt in range(1, self.retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                last = exc
                log.warning("%s attempt %d/%d failed: %s",
                            getattr(fn, "__name__", "call"), attempt, self.retries, exc)
                time.sleep(self.backoff * attempt)
        raise RuntimeError(f"giving up after {self.retries} attempts") from last

    def _round(self, coin: str, size: float, price: float):
        """Hyperliquid tick/size rules: 5 significant figures, 6 dp on size."""
        sz = round(size, 6)
        px = float(f"{price:.6g}")
        return sz, px

    # ------------------------------------------------------------------ orders
    def market_order(self, coin: str, side: str, size: float, mid: float,
                     slippage_pct: float = 0.0002) -> Fill:
        """IOC limit at mid +/- slippage: protects against a bad fill on a thin book."""
        is_buy = side == "buy"
        px = mid * (1 + slippage_pct) if is_buy else mid * (1 - slippage_pct)
        sz, px = self._round(coin, size, px)

        if self.dry_run:
            return self._simulate_order(coin, side, sz, mid)

        def _place():
            return self.client.order(
                coin=coin, is_buy=is_buy, sz=sz, limit_px=px,
                order_type={"limit": {"tif": "Ioc"}}, reduce_only=False)

        resp = self._retry(_place)
        if not isinstance(resp, dict) or resp.get("status") != "ok":
            raise RuntimeError(f"order rejected: {resp}")
        statuses = resp["response"]["data"]["statuses"]
        st = statuses[0]
        if "error" in st:
            raise RuntimeError(f"order error: {st['error']}")
        filled = st.get("filled") or {}
        if filled.get("totalSz") in (None, "0", "0.0"):
            raise RuntimeError(f"order not filled (IOC): {st}")
        log.info("FILLED %s %s %.6f @ %s", side, coin,
                 float(filled["totalSz"]), filled.get("avgPx"))
        return Fill(coin=coin, side=side, size=float(filled["totalSz"]),
                    price=float(filled.get("avgPx") or px),
                    order_id=str(st.get("oid")), simulated=False)

    def _simulate_order(self, coin: str, side: str, size: float, price: float) -> Fill:
        log.info("[DRY] %s %s %.6f @ %.6g", side.upper(), coin, size, price)
        return Fill(coin=coin, side=side, size=size, price=price,
                    order_id="dry", simulated=True)

    # ------------------------------------------------------------------ account
    def equity(self) -> float:
        if self.dry_run:
            return float(os.getenv("DRY_EQUITY", "1000"))
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        info = self._retry(Info, constants.MAINNET_API_URL, skip_ws=True)
        state = self._retry(info.user_state, os.getenv("HL_WALLET_ADDRESS"))
        return float(state["marginSummary"]["accountValue"])
