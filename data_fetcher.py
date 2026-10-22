# data_fetcher.py -- Hyperliquid public info API (no auth, no third-party deps).
"""Stdlib only (urllib) so the server needs nothing beyond numpy/pandas/scipy.

Every call retries with backoff; failures raise, they are never silently
swallowed -- silent data gaps are how backtests lie.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request

import pandas as pd

INFO_URL = "https://api.hyperliquid.xyz/info"
UA = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
log = logging.getLogger(__name__)


class HyperliquidData:
    def __init__(self, retries: int = 3, backoff: float = 2.0, timeout: int = 15):
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout

    def _post(self, payload: dict):
        last = None
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(INFO_URL, data=body, headers=UA)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as exc:                      # network / 5xx / timeout
                last = exc
                log.warning("info API attempt %d/%d failed: %s",
                            attempt, self.retries, exc)
                time.sleep(self.backoff * attempt)
        raise RuntimeError(
            f"Hyperliquid info API failed after {self.retries} tries") from last

    # ------------------------------------------------------------------ candles
    def candles(self, coin: str, interval: str = "1h", bars: int = 400) -> pd.DataFrame:
        now = int(time.time() * 1000)
        ms_per_bar = {"1h": 3_600_000, "4h": 14_400_000}[interval]
        start = now - (bars + 5) * ms_per_bar
        raw = self._post({"type": "candleSnapshot",
                          "req": {"coin": coin, "interval": interval,
                                  "startTime": start, "endTime": now}})
        if not raw:
            raise RuntimeError(f"empty candle response for {coin}")
        df = pd.DataFrame(raw)
        df["open_time"] = df["t"].astype("int64")
        for a, b in (("o", "open"), ("h", "high"), ("l", "low"),
                     ("c", "close"), ("v", "volume")):
            df[b] = df[a].astype("float64")
        df = df.sort_values("open_time").drop_duplicates("open_time")
        # no taker-buy split on this endpoint; neutralise it (unused by FEATURES)
        df["taker_buy_base"] = df["volume"] / 2.0
        df["trades"] = 0
        return df[["open_time", "open", "high", "low", "close", "volume",
                   "taker_buy_base", "trades"]].reset_index(drop=True)

    # ------------------------------------------------------------------ mid px
    def mids(self) -> dict[str, float]:
        raw = self._post({"type": "allMids"})
        return {k: float(v) for k, v in raw.items()}

    def spot(self, coin: str) -> float:
        m = self.mids()
        if coin not in m:
            raise KeyError(f"{coin} not in allMids")
        return m[coin]
