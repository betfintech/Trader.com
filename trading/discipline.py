"""
trading/discipline.py — Signal rate limiting (HARDENED)
=========================================================
HARDENING CHANGES:
  - _load: uses load_json_safe for corrupted/missing file handling
  - _save: already had atomic write (retained) + error handling improved
  - isoformat parsing wrapped per-timestamp to skip malformed entries
  - _prune: defensive — any parse error in stored timestamps just drops them
  - can_signal: fully wrapped in try/except; returns (True, "OK") on unexpected error
    to avoid blocking engine on discipline module crash
  - record_signal: wrapped in try/except so signal still dispatches if recording fails
"""
from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from threading import Lock

from core.config import MAX_TRADES_PER_HOUR, MAX_TRADES_PER_DAY
from core.logger import get_logger
from core.utils import atomic_write_json, load_json_safe

log = get_logger(__name__)

_TRADES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.json")


def _parse_dt(s: str) -> datetime | None:
    """Parse an ISO datetime string safely. Returns None on error."""
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class Discipline:
    """
    Enforces per-symbol cooldowns and global hourly/daily trade limits.
    Thread-safe. Never crashes callers.
    """

    SYMBOL_COOLDOWN_MINUTES: int = 60

    def __init__(self) -> None:
        self._lock = Lock()
        self._signal_times: dict[str, list[datetime]] = defaultdict(list)
        self._all_times: list[datetime] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        data = load_json_safe(_TRADES_FILE, default={})
        try:
            for sym, times in data.get("symbol_times", {}).items():
                parsed = [dt for t in times if (dt := _parse_dt(t)) is not None]
                self._signal_times[sym] = parsed

            self._all_times = [
                dt for t in data.get("all_times", [])
                if (dt := _parse_dt(t)) is not None
            ]
        except Exception as exc:
            log.warning("Discipline._load: error parsing trades.json: %s", exc)
            self._signal_times = defaultdict(list)
            self._all_times = []

    def _save(self) -> None:
        """Atomic write via temp-file + rename."""
        try:
            os.makedirs(os.path.dirname(_TRADES_FILE), exist_ok=True)
        except OSError:
            pass

        try:
            data = {
                "symbol_times": {
                    sym: [t.isoformat() for t in times]
                    for sym, times in self._signal_times.items()
                },
                "all_times": [t.isoformat() for t in self._all_times],
            }
            success = atomic_write_json(_TRADES_FILE, data)
            if not success:
                log.error("Discipline._save: atomic write failed for %s", _TRADES_FILE)
        except Exception as exc:
            log.error("Discipline._save: unexpected error: %s", exc)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)

    def _prune(self) -> None:
        """Remove timestamps older than 24 h."""
        try:
            cutoff = self._now() - timedelta(hours=24)
            self._all_times = [t for t in self._all_times if isinstance(t, datetime) and t > cutoff]
            for sym in list(self._signal_times):
                self._signal_times[sym] = [
                    t for t in self._signal_times[sym]
                    if isinstance(t, datetime) and t > cutoff
                ]
        except Exception as exc:
            log.warning("Discipline._prune error: %s", exc)

    # ── Public API ─────────────────────────────────────────────────────────────

    def can_signal(self, symbol: str) -> tuple[bool, str]:
        """
        Return (allowed, reason).
        Returns (True, "OK") on unexpected internal error — engine safety net.
        """
        try:
            with self._lock:
                self._prune()
                now = self._now()

                # Symbol cooldown
                sym_times = self._signal_times.get(symbol, [])
                if sym_times:
                    last = max(sym_times)
                    elapsed = (now - last).total_seconds() / 60
                    if elapsed < self.SYMBOL_COOLDOWN_MINUTES:
                        remaining = int(self.SYMBOL_COOLDOWN_MINUTES - elapsed)
                        return False, f"Symbol cooldown: {remaining}m remaining for {symbol}"

                # Hourly limit
                hour_cutoff = now - timedelta(hours=1)
                hourly_count = sum(1 for t in self._all_times if t > hour_cutoff)
                if hourly_count >= MAX_TRADES_PER_HOUR:
                    return False, f"Hourly limit reached ({MAX_TRADES_PER_HOUR}/h)"

                # Daily limit
                day_cutoff = now - timedelta(hours=24)
                daily_count = sum(1 for t in self._all_times if t > day_cutoff)
                if daily_count >= MAX_TRADES_PER_DAY:
                    return False, f"Daily limit reached ({MAX_TRADES_PER_DAY}/d)"

                return True, "OK"
        except Exception as exc:
            log.error("Discipline.can_signal unexpected error for %s: %s", symbol, exc)
            return True, "OK"  # Fail open — let engine continue

    def record_signal(self, symbol: str) -> None:
        """Record that a signal was sent for this symbol."""
        try:
            with self._lock:
                now = self._now()
                self._signal_times[symbol].append(now)
                self._all_times.append(now)
                self._save()
                log.debug(
                    "Signal recorded for %s | total today: %d",
                    symbol, len(self._all_times),
                )
        except Exception as exc:
            log.error("Discipline.record_signal failed for %s: %s", symbol, exc)
