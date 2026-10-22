# run_live.py -- main loop: data -> signal -> sizing -> order -> state.json
"""Restart-safe: state.json is written after EVERY change (entry, exit, tick).
On startup the open positions in state.json are adopted, not re-opened.

    python run_live.py --config config.yaml
    python run_live.py --once            # single pass, for cron/systemd timer
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from data_fetcher import HyperliquidData
from executor import HyperliquidExecutor
from strategy import exit_due, latest_signal

log = logging.getLogger("run_live")


# ---------------------------------------------------------------------- state
def load_state(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"equity": None, "positions": {}, "last_bar": {}, "trades": []}


def save_state(path: str, state: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp, path)            # atomic: never leave a half-written state


# ---------------------------------------------------------------------- sizing
def size_for(symbol: str, cfg: dict, state: dict, equity: float) -> float:
    """20% of equity by default, 30% hard ceiling, respect the concurrency cap."""
    if len(state["positions"]) >= cfg["max_concurrent_trades"]:
        log.info("skip %s: %d positions open (cap %d)", symbol,
                 len(state["positions"]), cfg["max_concurrent_trades"])
        return 0.0
    if symbol in state["positions"]:
        return 0.0
    pct = min(cfg["position_size_pct"], cfg["max_position_size_pct"])
    notional = equity * pct
    free = equity * (1 - cfg["reserve_cash_pct"])
    used = sum(p["notional"] for p in state["positions"].values())
    notional = max(0.0, min(notional, free - used))
    return notional


# ---------------------------------------------------------------------- main
def run_once(cfg: dict, data: HyperliquidData, ex: HyperliquidExecutor, state: dict) -> None:
    equity = ex.equity()
    state["equity"] = equity
    now_ms = int(time.time() * 1000)

    # ---- 1. exits first (time-based, matches the backtest)
    for sym in list(state["positions"]):
        pos = state["positions"][sym]
        if exit_due(pos["entry_time_ms"], now_ms, cfg["hold_bars"]):
            mid = data.spot(sym)
            fill = ex.market_order(sym, "sell", pos["size"], mid, cfg["slippage_pct"])
            pnl = (fill.price - pos["entry_price"]) * pos["size"]
            state["trades"].append({
                "symbol": sym, "entry_time": pos["entry_time_utc"],
                "exit_time": datetime.now(timezone.utc).isoformat(),
                "entry": pos["entry_price"], "exit": fill.price,
                "size": pos["size"], "pnl": pnl, "reason": "time"})
            del state["positions"][sym]
            save_state(cfg["state_file"], state)
            log.info("EXIT %s @ %.6g  pnl %+.2f", sym, fill.price, pnl)

    # ---- 2. entries
    for sym in cfg["symbols"]:
        try:
            candles = data.candles(sym, cfg["interval"], cfg["history_bars"])
            sig = latest_signal(candles, ex.model, cfg["entry_prob"])
        except Exception as exc:
            log.error("signal failed for %s: %s", sym, exc)
            continue
        if sig is None:
            continue
        if state["last_bar"].get(sym) == sig["open_time"]:
            continue                       # already acted on this bar
        state["last_bar"][sym] = sig["open_time"]
        save_state(cfg["state_file"], state)
        if not sig["signal"]:
            continue
        notional = size_for(sym, cfg, state, equity)
        if notional <= 0:
            continue
        size = notional / sig["close"]
        fill = ex.market_order(sym, "buy", size, sig["close"], cfg["slippage_pct"])
        state["positions"][sym] = {
            "entry_price": fill.price, "size": fill.size,
            "notional": fill.size * fill.price,
            "entry_time_ms": now_ms,
            "entry_time_utc": datetime.now(timezone.utc).isoformat(),
            "prob": sig["prob"], "signal_bar": sig["open_time"]}
        save_state(cfg["state_file"], state)
        log.info("ENTRY %s prob %.3f size %.6f @ %.6g",
                 sym, sig["prob"], fill.size, fill.price)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[logging.FileHandler(cfg["log_file"], encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)])

    import pickle

    with open(cfg["model_file"], "rb") as f:
        model = pickle.load(f)

    data = HyperliquidData(retries=cfg["retry_attempts"],
                           backoff=cfg["retry_backoff_sec"])
    ex = HyperliquidExecutor(dry_run=cfg["dry_run"],
                             retries=cfg["retry_attempts"],
                             backoff=cfg["retry_backoff_sec"])
    ex.model = model                      # reuse one loaded model everywhere

    state = load_state(cfg["state_file"])
    log.info("start | dry_run=%s | symbols=%s | open=%d | equity=%.2f",
             cfg["dry_run"], cfg["symbols"], len(state["positions"]),
             state["equity"] or 0)

    while True:
        try:
            run_once(cfg, data, ex, state)
        except Exception:
            log.exception("loop iteration failed; sleeping and retrying")
        if args.once:
            break
        time.sleep(cfg["poll_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
