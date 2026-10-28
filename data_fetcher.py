# data_fetcher.py -- Binance public market data API.
"""Binance public data adapter with the same high-level API as the previous
Hyperliquid version so the strategy loop can remain unchanged.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

import pandas as pd

BASE_URL = "https://api.binance.com"
UA = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
log = logging.getLogger(__name__)


class BinanceData:
    def __init__(self, retries: int = 3, backoff: float = 2.0, timeout: int = 15):
        self.retries = retries
        self.backoff = backoff
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None):
        last = None
        url = BASE_URL + path
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        for attempt in range(1, self.retries + 1):
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode("utf-8"))
            except Exception as exc:
                last = exc
                log.warning("Binance API attempt %d/%d failed: %s",
                            attempt, self.retries, exc)
                time.sleep(self.backoff * attempt)
        raise RuntimeError(
            f"Binance public API failed after {self.retries} tries") from last

    @staticmethod
    def _normalize_symbol(coin: str) -> str:
        symbol = str(coin).strip().upper().replace(" ", "")
        if symbol.endswith("USD") and not symbol.endswith("USDT"):
            symbol = symbol[:-3] + "USDT"
        if not symbol.endswith("USDT"):
            symbol = f"{symbol}USDT"
        return symbol

    def candles(self, coin: str, interval: str = "1h", bars: int = 400) -> pd.DataFrame:
        symbol = self._normalize_symbol(coin)
        interval_key = {"1h": "1h", "4h": "4h"}.get(interval)
        if interval_key is None:
            raise ValueError(f"unsupported interval: {interval!r}")
        raw = self._get(
            "/api/v3/klines",
            {"symbol": symbol, "interval": interval_key, "limit": bars},
        )
        if not raw:
            raise RuntimeError(f"empty candle response for {symbol}")

        rows = []
        for item in raw:
            if len(item) < 6:
                continue
            rows.append({
                "open_time": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
            })

        if not rows:
            raise RuntimeError(f"no valid candle rows for {symbol}")

        df = pd.DataFrame(rows)
        df = df.sort_values("open_time").drop_duplicates("open_time")
        df["taker_buy_base"] = df["volume"] / 2.0
        df["trades"] = 0
        return df[["open_time", "open", "high", "low", "close", "volume",
                   "taker_buy_base", "trades"]].reset_index(drop=True)

    def mids(self) -> dict[str, float]:
        raw = self._get("/api/v3/ticker/price")
        return {str(item["symbol"]): float(item["price"]) for item in raw
                if "symbol" in item and "price" in item}

    def spot(self, coin: str) -> float:
        symbol = self._normalize_symbol(coin)
        raw = self._get("/api/v3/ticker/price", {"symbol": symbol})
        if isinstance(raw, dict):
            price = raw.get("price")
        else:
            price = None
        if price is None:
            raise KeyError(f"{symbol} not in ticker/price response")
        return float(price)


# Backward-compatible aliases for the rest of the program.
HyperliquidData = BinanceData
