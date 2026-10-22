# run_live.py -- main loop: data -> signal -> sizing -> order -> state.json
"""Restart-safe: state.json is written after EVERY change (entry, exit, tick).

Three things this version fixes:
  1. dry-run equity is REAL: fetched at startup and logged, never 0.00
  2. a heartbeat line is logged on EVERY poll, so silence always means "dead"
  3. logging goes to BOTH stdout and run_live.log (force=True so library
     logging cannot steal the root logger)

    python run_live.py --once            # single pass (systemd timer / cron)
    python run_live.py                   # poll forever
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from data_fetcher import HyperliquidData
from executor import HyperliquidExecutor
from strategy import exit_due, latest_signal

log = logging.getLogger("run_live")
HEARTBEAT_EVERY = 1          # log a heartbeat line every N polls


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
    """20% of equity by default, 30% hard ceiling, respects the concurrency cap."""
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
    return max(0.0, min(notional, free - used))


# ---------------------------------------------------------------------- main
def run_once(cfg: dict, data: HyperliquidData, ex: HyperliquidExecutor,
             state: dict, poll_id: int = 0) -> None:
    equity = ex.equity()
    if cfg["dry_run"]:
        # the paper account must track realised PnL, else every run reports 1000
        equity += sum(t.get("pnl", 0.0) for t in state["trades"])
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
            log.info("EXIT %s @ %.6g  pnl %+.2f  equity %.2f", sym, fill.price, pnl, equity)

    # ---- 2. entries + heartbeat
    hb = []
    for sym in cfg["symbols"]:
        try:
            candles = data.candles(sym, cfg["interval"], cfg["history_bars"])
            sig = latest_signal(candles, ex.model, cfg["entry_prob"])
        except Exception as exc:
            log.error("signal failed for %s: %s", sym, exc)
            hb.append(f"{sym}=ERR")
            continue
        if sig is None:
            log.warning("HB  %-4s bars=%d  features not ready (need >= 72 bars)",
                        sym, len(candles))
            hb.append(f"{sym}=NA")
            continue

        hb.append(f"{sym}={sig['prob']:.3f}")
        new_bar = state["last_bar"].get(sym) != sig["open_time"]
        if new_bar:
            log.info("BAR %-4s close=%-12.6g prob=%.4f vol24=%.0f%%  %s",
                     sym, sig["close"], sig["prob"], sig["vol_24"] * 100,
                     "SIGNAL" if sig["signal"] else "")
            state["last_bar"][sym] = sig["open_time"]
            save_state(cfg["state_file"], state)
        if not sig["signal"] or not new_bar:
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
        log.info("ENTRY %s prob %.3f size %.6f @ %.6g (notional %.2f)",
                 sym, sig["prob"], fill.size, fill.price, notional)

    # ---- 3. the line that proves the process is alive
    if poll_id % HEARTBEAT_EVERY == 0:
        log.info("HB  %s | eq=%.2f pos=%d | %s",
                 datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                 equity, len(state["positions"]), "  ".join(hb))


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
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.FileHandler(cfg["log_file"], encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
        force=True)                     # take over the root logger, no exceptions

    with open(cfg["model_file"], "rb") as f:
        model = pickle.load(f)
    log.info("model loaded from %s  (%s)", cfg["model_file"], type(model).__name__)

    data = HyperliquidData(retries=cfg["retry_attempts"],
                           backoff=cfg["retry_backoff_sec"])
    ex = HyperliquidExecutor(dry_run=cfg["dry_run"],
                             retries=cfg["retry_attempts"],
                             backoff=cfg["retry_backoff_sec"])
    ex.model = model

    state = load_state(cfg["state_file"])
    equity = ex.equity()                # <- real number, fetched BEFORE logging
    state["equity"] = equity
    save_state(cfg["state_file"], state)
    log.info("start | dry_run=%s | symbols=%s | open=%d | equity=%.2f | log=%s",
             cfg["dry_run"], ",".join(cfg["symbols"]),
             len(state["positions"]), equity, cfg["log_file"])

    poll = 0
    while True:
        try:
            run_once(cfg, data, ex, state, poll_id=poll)
        except Exception:
            log.exception("loop iteration failed; sleeping and retrying")
        poll += 1
        if args.once:
            break
        time.sleep(cfg["poll_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
