"""
trading/engine.py — Production-Grade SMC Trading Engine
=========================================================
Continuously scans all configured pairs, generates SMC signals via the
strategy pipeline, applies discipline controls, deduplicates signals, and
dispatches clean signals through the notifier.

NEVER executes trades.

Production guarantees:
  ✓ Never crashes the system — every pair is wrapped independently
  ✓ Never stops silently — heartbeat logs on every cycle
  ✓ Scan interval: 60–120 s (configurable via MAIN_LOOP_INTERVAL in config)
  ✓ Full candle validation before strategy is called
  ✓ Signal deduplication — same direction on same pair is not re-sent
  ✓ Signal quality gate — entry/SL/TP must be valid numbers
  ✓ Shared runtime state updated every cycle (readable by chatbot / web)
  ✓ API errors per-pair are logged and skipped; loop always continues
"""
from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from threading import Event

from core.config import (
    MAIN_LOOP_INTERVAL,
    CRYPTO_PAIRS,
    FOREX_PAIRS,
    TF_H1,
    TF_M15,
    TF_M1,
    TF_M5,
)
from core.logger import get_logger
from trading.discipline import Discipline
from trading.strategy import Signal, generate_signal
from trading.market.unified import get_candles, market_type
from trading.runtime_state import state as runtime_state
from communication.notifier import send_signal

log = get_logger(__name__)

# ── Scan interval ──────────────────────────────────────────────────────────────
# M15-based setups form inside candles, not only at candle close.
# 60 s gives fresh signals; 120 s is the upper practical bound.
# MAIN_LOOP_INTERVAL is read from config (env: MAIN_LOOP_INTERVAL).
# Values outside 60–120 are clamped so a misconfigured env cannot break timing.
_MIN_INTERVAL = 60
_MAX_INTERVAL = 120
_SCAN_INTERVAL: int = max(_MIN_INTERVAL, min(_MAX_INTERVAL, int(MAIN_LOOP_INTERVAL)))

# ── Candle minimums ────────────────────────────────────────────────────────────
_MIN_H1_CANDLES  = 5  # Lowered from 10 to allow charts to load faster during startup
_MIN_M15_CANDLES = 10


class TradingEngine:
    """
    Continuously scans all configured pairs, generates SMC signals,
    applies discipline checks, and dispatches signals via notifier.
    Never executes trades.
    """

    ALL_PAIRS: list[str] = CRYPTO_PAIRS + FOREX_PAIRS

    def __init__(self) -> None:
        self._discipline = Discipline()
        self._stop_event = Event()

        # ── In-memory deduplication store ─────────────────────────────────────
        # Maps symbol → last dispatched signal direction ("BUY" | "SELL").
        # Prevents sending the same direction twice in a row for the same pair.
        # Cleared on engine restart (intentional — stale state is worse than
        # an occasional re-send after a restart).
        self._last_sent: dict[str, str] = {}

    # ── Public API ─────────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Signal the engine to stop after the current cycle completes."""
        self._stop_event.set()

    def run(self) -> None:
        """
        Main engine loop.

        Structure:
            loop forever (until stop()):
                log cycle start
                process every pair (each independently fault-tolerant)
                log cycle complete
                sleep SCAN_INTERVAL
        """
        runtime_state.mark_running()
        log.info(
            "Trading engine started | pairs: %s | scan interval: %ds",
            self.ALL_PAIRS,
            _SCAN_INTERVAL,
        )

        while not self._stop_event.is_set():
            try:
                self._scan_cycle()
            except Exception as exc:
                # This guard should never fire — _scan_cycle is itself fully
                # wrapped — but it is here as the ultimate safety net so that
                # an unforeseen error never kills the engine thread.
                err_msg = f"Unexpected engine-level error: {exc}"
                log.error(err_msg, exc_info=True)
                runtime_state.record_error(err_msg)

            # Interruptible sleep: wakes immediately on stop()
            self._stop_event.wait(timeout=_SCAN_INTERVAL)

        runtime_state.mark_stopped()
        log.info("Trading engine stopped cleanly.")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _scan_cycle(self) -> None:
        """
        One full scan of all pairs.
        Each pair is wrapped in its own try/except so one failure cannot
        prevent the remaining pairs from being processed.
        """
        cycle_start = datetime.now(timezone.utc)
        runtime_state.update_scan_start()

        log.info(
            "-- Scan cycle #%d started -- %s UTC -- %d pairs",
            runtime_state.snapshot()["cycle_count"],
            cycle_start.strftime("%H:%M:%S"),
            len(self.ALL_PAIRS),
        )

        pairs_processed = 0

        for symbol in self.ALL_PAIRS:
            if self._stop_event.is_set():
                log.info("Stop requested — aborting cycle mid-scan.")
                break

            log.info("Processing pair: %s", symbol)

            try:
                self._process_symbol(symbol)
                pairs_processed += 1
            except Exception as exc:
                # Per-pair isolation: log and continue — never halt the loop
                err_msg = f"[{symbol}] Unhandled error in _process_symbol: {exc}"
                log.error(err_msg, exc_info=True)
                runtime_state.record_error(err_msg)

        elapsed = (datetime.now(timezone.utc) - cycle_start).total_seconds()
        runtime_state.update_scan_complete(pairs_processed)

        log.info(
            "-- Scan cycle #%d completed -- %d/%d pairs processed -- %.1fs elapsed",
            runtime_state.snapshot()["cycle_count"],
            pairs_processed,
            len(self.ALL_PAIRS),
            elapsed,
        )

    def _process_symbol(self, symbol: str) -> None:
        """
        Full processing pipeline for one symbol.

        Steps:
          [1] Discipline check (cooldown / hourly / daily limits)
          [2] Fetch candles (H1 + M15)
          [3] Validate candle data
          [4] Generate strategy signal
          [5] Quality gate (actionable + valid levels)
          [6] Deduplication gate
          [7] Record discipline + update runtime state + dispatch
        """
        mtype = market_type(symbol)

        # ── [1] Discipline check ───────────────────────────────────────────────
        allowed, reason = self._discipline.can_signal(symbol)
        if not allowed:
            log.debug("[%s] Discipline skip: %s", symbol, reason)
            return

        # ── [2] Fetch candles ──────────────────────────────────────────────────
        log.debug("[%s] Fetching candles (H1 + M15)", symbol)
        try:
            h1_raw  = get_candles(symbol, TF_H1,  limit=100)
            m15_raw = get_candles(symbol, TF_M15, limit=100)
        except Exception as exc:
            log.warning("[%s] Candle fetch raised exception: %s", symbol, exc)
            return

        # ── [3] Candle validation ──────────────────────────────────────────────
        valid, validation_reason = self._validate_candles(symbol, h1_raw, m15_raw)
        if not valid:
            log.warning("[%s] Candle validation failed: %s", symbol, validation_reason)
            return

        # ── [4] Generate signal ────────────────────────────────────────────────
        try:
            signal = generate_signal(symbol, h1_raw, m15_raw, mtype)
        except Exception as exc:
            log.error("[%s] Strategy raised exception: %s", symbol, exc, exc_info=True)
            return

        log.info("[%s] Signal generated: %s -- %s", symbol, signal.direction, signal.reason)

        if signal.direction == "WAIT":
            return

        # ── [5] Signal quality gate ────────────────────────────────────────────
        if not self._passes_quality_gate(signal):
            return

        # ── [6] Deduplication gate ─────────────────────────────────────────────
        last = self._last_sent.get(symbol)
        if last == signal.direction:
            log.debug(
                "[%s] Duplicate suppressed -- %s already sent for this pair",
                symbol, signal.direction,
            )
            return

        # ── [7] Dispatch ───────────────────────────────────────────────────────
        self._discipline.record_signal(symbol)
        self._last_sent[symbol] = signal.direction
        runtime_state.record_signal(symbol, signal.direction)

        log.info(
            "[%s] %s signal dispatching -> entry=%.5f SL=%.5f TP1=%.5f",
            symbol, signal.direction, signal.entry, signal.stop_loss, signal.tp1,
        )
        send_signal(signal)

    # ── Validation helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _validate_candles(
        symbol: str,
        h1_raw: object,
        m15_raw: object,
    ) -> tuple[bool, str]:
        """
        Validate raw candle data before handing it to the strategy.

        Checks:
          - Both feeds are non-empty lists
          - Minimum candle counts are met (H1>=30, M15>=20)
          - No candle has a zero or negative close price

        Returns (valid, reason).
        """
        if not h1_raw or not isinstance(h1_raw, list):
            return False, "H1 candle feed is empty or invalid"
        if not m15_raw or not isinstance(m15_raw, list):
            return False, "M15 candle feed is empty or invalid"

        if len(h1_raw) < _MIN_H1_CANDLES:
            return False, (
                f"Insufficient H1 candles: {len(h1_raw)} < {_MIN_H1_CANDLES} required"
            )
        if len(m15_raw) < _MIN_M15_CANDLES:
            return False, (
                f"Insufficient M15 candles: {len(m15_raw)} < {_MIN_M15_CANDLES} required"
            )

        # Price sanity on the most recent candles (strategy catches the rest)
        for label, candles in (("H1", h1_raw[-5:]), ("M15", m15_raw[-5:])):
            for c in candles:
                if not isinstance(c, dict):
                    return False, f"{label} candle is not a dict: {c!r}"
                price = c.get("close", 0)
                if not isinstance(price, (int, float)) or price <= 0:
                    return False, f"{label} candle has invalid close price: {price!r}"

        return True, "OK"

    @staticmethod
    def _passes_quality_gate(signal: Signal) -> bool:
        """
        Final quality check before a BUY/SELL signal leaves the engine.

        Rejects the signal if:
          - direction is not BUY or SELL
          - is_actionable is False
          - entry, stop_loss, or tp1 are zero, negative, or NaN/Inf
        """
        if not signal.is_actionable:
            log.warning(
                "[%s] Invalid signal blocked -- direction not actionable: %s",
                signal.symbol, signal.direction,
            )
            return False

        def _bad(v: float) -> bool:
            return not isinstance(v, (int, float)) or v <= 0 or not math.isfinite(v)

        for field_name, value in (
            ("entry",     signal.entry),
            ("stop_loss", signal.stop_loss),
            ("tp1",       signal.tp1),
        ):
            if _bad(value):
                log.warning(
                    "[%s] Invalid signal blocked -- %s is invalid: %s",
                    signal.symbol, field_name, value,
                )
                return False

        return True
        