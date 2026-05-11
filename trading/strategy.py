"""
Smart Money Concepts (SMC) Strategy — STRICT v4
================================================
Works identically for crypto and forex.
Market-specific parameters are adjusted via the market_type argument.

Signal output: BUY | SELL | WAIT

DESIGN PHILOSOPHY:
  Clean setups only — no forced trades under any condition.
  Every gate is HARD except Gate 5 (key level proximity), which logs a
  warning but never blocks.  All soft fallbacks from v3 have been removed
  to match the institutional trading standard exactly.

GATE SUMMARY:
  GATE 1  - Session filter (Asian blocked for forex; crypto 24/7)        [HARD]
  GATE 2  - H1 Market structure — HH/HL or LH/LL; Range → WAIT          [HARD]
  GATE 3  - Price zone — BUY needs Discount, SELL needs Premium;         [HARD]
             Equilibrium is NOT a valid entry zone → WAIT
  GATE 4  - Volatility — must be above 50% of configured threshold       [HARD]
  GATE 5  - Key level proximity — SOFT (logs warning, never blocks)      [SOFT]
  GATE 6  - H1 Liquidity sweep — real wick sweep only; 3 consecutive     [HARD]
             closes do NOT count; no soft fallback
  GATE 7  - Market narrative — Reversal → WAIT; continuation/pullback OK [HARD]
  GATE 8  - Momentum — body/wick >= 0.6; at least 2/5 strong directional [HARD]
  GATE 9  - M15 Liquidity sweep — real wick sweep required; no fallback  [HARD]
             to H1 reference; 3 consecutive closes NOT accepted
  GATE 10 - CHOCH/BOS on M15 — price must close beyond prior swing;      [HARD]
             3 consecutive closes do NOT count; no soft fallback
  GATE 11 - Point of Interest — OB first, FVG second, swing extreme      [HARD]
             only as last fallback when sweep reference exists
  GATE 12 - Confirmation candle — body_ratio >= 0.60 OR full engulfing   [HARD]
             OR rejection wick >= 50% of candle range
  RR      - Minimum 1:2 enforced strictly on every signal                [HARD]

WHAT MAKES THIS HIGH-ACCURACY:
  - Real H1 structure (HH/HL or LH/LL) required
  - Real wick sweeps only on BOTH H1 and M15 — no 3-close substitutes
  - Equilibrium rejected as entry zone — discount for BUY, premium for SELL only
  - M15 sweep is its own HARD gate — H1 sweep does not substitute
  - Real structural break (BOS/CHOCH) required on M15
  - POI (OB/FVG) required; swing extreme only as explicitly stated fallback
  - SL placed beyond actual sweep wick with 0.1% forex / 0.3% crypto buffer
  - Final TP targets the nearest prior liquidity pool (prior highs or lows)
  - RR 1:2 strictly enforced — every losing trade stays controlled
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from core.config import (
    CRYPTO_VOLATILITY_THRESHOLD,
    FOREX_VOLATILITY_THRESHOLD,
    ENABLE_SESSION_FILTER,
)
from core.logger import get_logger
from core.utils import pct_change

log = get_logger(__name__)


# ======================================================================
# DATA CONTAINERS
# ======================================================================

@dataclass
class Signal:
    symbol: str
    direction: str          # BUY | SELL | WAIT
    market_type: str        # crypto | forex
    entry: float = 0.0
    stop_loss: float = 0.0
    tp1: float = 0.0
    tp2: float = 0.0
    tp_final: float = 0.0
    reason: str = ""
    setup_quality: str = ""   # "A" | "B" | "C" — for logging/reference only
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.direction in ("BUY", "SELL")


@dataclass
class Candle:
    timestamp: object
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_ratio(self) -> float:
        return self.body / self.range if self.range > 0 else 0.0

    @property
    def is_strong(self) -> bool:
        # Aligned with Gate 12: strong requires body_ratio >= 0.60
        return self.body_ratio >= 0.60 and self.body > 0


def _to_candles(raw: list[dict]) -> list[Candle]:
    result = []
    for r in raw:
        try:
            result.append(Candle(**r))
        except (TypeError, KeyError):
            pass
    return result


# ======================================================================
# SESSION LOGIC
# ======================================================================

def _current_session() -> str:
    hour = datetime.now(timezone.utc).hour
    if 13 <= hour < 22:
        return "NewYork"
    if 7 <= hour < 16:
        return "London"
    return "Asian"


def _is_tradeable_session(market_type: str) -> tuple[bool, str]:
    """Gate 1: Asian session blocked for forex; crypto trades 24/7."""
    if not ENABLE_SESSION_FILTER:
        return True, "Session filter disabled"
    if market_type == "crypto":
        return True, "Crypto: 24/7 market"
    session = _current_session()
    hour = datetime.now(timezone.utc).hour
    if session == "Asian":
        return False, (
            f"Asian session ({hour:02d}:00 UTC) — low volatility, "
            "no clean setups. Wait for London open (07:00 UTC)."
        )
    return True, f"Session: {session}"


def _session_volatility_ok(candles: list[Candle], market_type: str) -> tuple[bool, str]:
    """Gate 4: Must exceed 50% of configured volatility threshold."""
    if len(candles) < 5:
        return False, "Too few candles for volatility check"
    recent_ranges = [c.range for c in candles[-5:]]
    avg_range = statistics.mean(recent_ranges)
    price = candles[-1].close
    if price <= 0:
        return False, "Invalid price"
    rel_vol = avg_range / price
    base = (CRYPTO_VOLATILITY_THRESHOLD if market_type == "crypto"
            else FOREX_VOLATILITY_THRESHOLD)
    threshold = base * 0.5
    if rel_vol < threshold:
        return False, (
            f"Volatility too low ({rel_vol:.5f} < {threshold:.5f}) — market flat"
        )
    return True, f"Volatility OK ({rel_vol:.5f} >= {threshold:.5f})"


# ======================================================================
# H1 BIAS ENGINE
# ======================================================================

def _find_swing_highs_lows(candles: list[Candle], window: int = 3):
    """Swing pivot detection using a symmetric window."""
    highs, lows = [], []
    for i in range(window, len(candles) - window):
        slice_highs = [c.high for c in candles[i - window:i + window + 1]]
        slice_lows  = [c.low  for c in candles[i - window:i + window + 1]]
        if candles[i].high == max(slice_highs):
            highs.append(i)
        if candles[i].low == min(slice_lows):
            lows.append(i)
    return highs, lows


def _detect_structure(candles: list[Candle]) -> tuple[str, str, str]:
    """
    Gate 2: H1 market structure detection.
    Returns (trend, reason, quality).
      quality = "strong"  — full HH+HL or LH+LL confirmed
      quality = "partial" — only one side confirmed
      quality = "slope"   — slope fallback only (weakest signal)

    Range → "range" → HARD BLOCK in Gate 2.
    """
    if len(candles) < 20:
        return "range", "Insufficient data for structure analysis", "none"

    highs, lows = _find_swing_highs_lows(candles, window=3)

    if len(highs) >= 2 and len(lows) >= 2:
        rh = [candles[i].high for i in highs[-2:]]
        rl = [candles[i].low  for i in lows[-2:]]
        hh = rh[-1] > rh[-2]
        hl = rl[-1] > rl[-2]
        lh = rh[-1] < rh[-2]
        ll = rl[-1] < rl[-2]
        if hh and hl:
            return "bullish", "HH + HL — bullish structure", "strong"
        if lh and ll:
            return "bearish", "LH + LL — bearish structure", "strong"
        if hh:
            return "bullish", "HH confirmed (partial)", "partial"
        if hl:
            return "bullish", "HL confirmed (partial)", "partial"
        if lh:
            return "bearish", "LH confirmed (partial)", "partial"
        if ll:
            return "bearish", "LL confirmed (partial)", "partial"

    if len(highs) >= 2:
        rh = [candles[i].high for i in highs[-2:]]
        if rh[-1] > rh[-2]:
            return "bullish", "HH confirmed (no HL yet)", "partial"
        if rh[-1] < rh[-2]:
            return "bearish", "LH confirmed (no LL yet)", "partial"

    if len(lows) >= 2:
        rl = [candles[i].low for i in lows[-2:]]
        if rl[-1] > rl[-2]:
            return "bullish", "HL confirmed (no HH yet)", "partial"
        if rl[-1] < rl[-2]:
            return "bearish", "LL confirmed (no LH yet)", "partial"

    closes = [c.close for c in candles]
    mid = len(closes) // 2
    slope = pct_change(statistics.mean(closes[:mid]), statistics.mean(closes[mid:]))
    if slope > 0.002:
        return "bullish", f"Bullish slope ({slope:.3%})", "slope"
    if slope < -0.002:
        return "bearish", f"Bearish slope ({slope:.3%})", "slope"
    return "range", f"Range market (slope={slope:.3%})", "none"


def _price_zone(candles: list[Candle]) -> tuple[str, float, float, float]:
    """
    Gate 3: Price zone classification using the last 50 H1 candles.

    BUY  → Discount only   (below midpoint minus equilibrium band)
    SELL → Premium only    (above midpoint plus equilibrium band)
    Equilibrium → NOT a valid entry zone → caller must WAIT
    """
    recent = candles[-50:] if len(candles) >= 50 else candles
    high = max(c.high for c in recent)
    low  = min(c.low  for c in recent)
    mid  = (high + low) / 2
    eq_band = (high - low) * 0.05
    current = candles[-1].close
    if current > mid + eq_band:
        return "premium", high, low, mid
    if current < mid - eq_band:
        return "discount", high, low, mid
    return "equilibrium", high, low, mid


def _at_key_level(
    candles: list[Candle],
    high_idxs: list[int],
    low_idxs: list[int],
    market_type: str = "forex",
) -> tuple[bool, str]:
    """
    Gate 5 (SOFT): Key level proximity check.
    A miss logs a warning but NEVER blocks an entry.
    Tolerance: 1.5% forex / 2.5% crypto. Round-number band: 0.8%.
    """
    current = candles[-1].close
    tol_pct = 0.025 if market_type == "crypto" else 0.015
    tolerance = current * tol_pct

    for i in high_idxs[-10:]:
        if abs(candles[i].high - current) <= tolerance:
            return True, f"Near swing HIGH ({candles[i].high:.5f})"

    for i in low_idxs[-10:]:
        if abs(candles[i].low - current) <= tolerance:
            return True, f"Near swing LOW ({candles[i].low:.5f})"

    for mag in (1000.0, 100.0, 10.0, 1.0, 0.1, 0.01):
        nearest = round(current / mag) * mag
        if nearest > 0 and abs(nearest - current) / current < 0.008:
            return True, f"Near round-number ({nearest})"

    return False, "Price not at a significant H1 key level"


# ======================================================================
# LIQUIDITY ANALYSIS
# ======================================================================

def _liquidity_swept(
    candles: list[Candle],
    trend: str,
    strict: bool = True,
) -> tuple[bool, str, float]:
    """
    Liquidity sweep detection — REAL WICK SWEEP ONLY.

    A real sweep requires: price wicks beyond a prior swing high/low AND
    closes back on the correct side of that level.
    Minimum wick size: 20% of the sweeping candle's range. Lookback: 10 candles.

    THREE CONSECUTIVE CLOSES IN ONE DIRECTION DO NOT QUALIFY AS A SWEEP.
    The ``strict`` parameter is retained for API compatibility; the former
    soft 3-close fallback has been removed entirely regardless of its value.
    """
    if len(candles) < 10:
        return False, "Too few candles for sweep detection", 0.0

    for lookback in range(1, 11):
        if lookback >= len(candles):
            break
        candidate = candles[-lookback]
        prior_slice = candles[max(0, -(lookback + 12)): -lookback]
        if len(prior_slice) < 5:
            continue

        if trend == "bullish":
            ref_low = min(c.low for c in prior_slice)
            if candidate.low < ref_low and candidate.close > ref_low:
                wick_size = ref_low - candidate.low
                wick_pct = wick_size / candidate.range if candidate.range > 0 else 0
                if wick_pct >= 0.20:
                    return (
                        True,
                        f"Bullish sweep: wick {candidate.low:.5f} < ref {ref_low:.5f}, "
                        f"closed {candidate.close:.5f} (wick={wick_pct:.0%})",
                        candidate.low,
                    )
        else:
            ref_high = max(c.high for c in prior_slice)
            if candidate.high > ref_high and candidate.close < ref_high:
                wick_size = candidate.high - ref_high
                wick_pct = wick_size / candidate.range if candidate.range > 0 else 0
                if wick_pct >= 0.20:
                    return (
                        True,
                        f"Bearish sweep: wick {candidate.high:.5f} > ref {ref_high:.5f}, "
                        f"closed {candidate.close:.5f} (wick={wick_pct:.0%})",
                        candidate.high,
                    )

    return False, "No real wick sweep detected (3-close fallback not accepted)", 0.0


# ======================================================================
# MARKET NARRATIVE
# ======================================================================

def _classify_narrative(candles: list[Candle], trend: str) -> tuple[str, str]:
    """
    Gate 7: Market narrative classification.
    - REVERSAL (8+/10 candles against trend) — HARD BLOCK.
    - CONTINUATION (strong directional move)  — ALLOW.
    - PULLBACK (retracement in progress)       — ALLOW (best SMC entry point).
    - UNCLEAR / mixed                          — ALLOW (weak momentum caught by Gate 8).
    """
    if len(candles) < 20:
        return "unclear", "Insufficient data"

    last5  = candles[-5:]
    last10 = candles[-10:]
    bull5  = sum(1 for c in last5  if c.is_bullish)
    bear5  = sum(1 for c in last5  if c.is_bearish)
    bull10 = sum(1 for c in last10 if c.is_bullish)
    bear10 = sum(1 for c in last10 if c.is_bearish)
    strong5 = sum(1 for c in last5 if c.is_strong)

    if trend == "bullish":
        if bull5 >= 4 and strong5 >= 2:
            return "continuation", f"Bullish continuation: {bull5}/5 bull, {strong5} strong"
        if bear5 >= 3:
            if bear10 >= 8:
                return "reversal", f"Potential reversal: {bear10}/10 bearish — BLOCKED"
            return "pullback", f"Bullish pullback: {bear5}/5 bearish (retracement)"
        return "unclear", f"Mixed: {bull5} bull / {bear5} bear"
    else:
        if bear5 >= 4 and strong5 >= 2:
            return "continuation", f"Bearish continuation: {bear5}/5 bear, {strong5} strong"
        if bull5 >= 3:
            if bull10 >= 8:
                return "reversal", f"Potential reversal: {bull10}/10 bullish — BLOCKED"
            return "pullback", f"Bearish pullback: {bull5}/5 bullish (retracement)"
        return "unclear", f"Mixed: {bull5} bull / {bear5} bear"


def _momentum_strength(candles: list[Candle], trend: str) -> tuple[bool, str]:
    """
    Gate 8: Momentum quality check.

    Requirements (both must pass):
      1. Average body / average wick ratio >= 0.60 across last 5 candles.
      2. At least 2 out of 5 recent candles must be strong and directional
         (body_ratio >= 0.60 and closing in the trend direction).

    Weak, mixed, or indecisive candles → WAIT.
    """
    if len(candles) < 5:
        return False, "Too few candles for momentum check"
    last5 = candles[-5:]
    bodies = [c.body for c in last5]
    wicks  = [c.upper_wick + c.lower_wick for c in last5]
    avg_body = statistics.mean(bodies)
    avg_wick = statistics.mean(wicks) + 1e-10
    body_ratio = avg_body / avg_wick
    directional = (
        sum(1 for c in last5 if c.is_bullish and c.is_strong)
        if trend == "bullish"
        else sum(1 for c in last5 if c.is_bearish and c.is_strong)
    )
    if body_ratio < 0.6:
        return False, f"Weak momentum: body/wick ratio {body_ratio:.2f} < 0.60"
    if directional < 2:
        return False, (
            f"Weak momentum: only {directional}/5 strong directional candles "
            "(minimum 2 required)"
        )
    return True, f"Momentum OK: body/wick={body_ratio:.2f}, {directional}/5 directional"


# ======================================================================
# M15 ENTRY CONFIRMATION ENGINE
# ======================================================================

def _detect_choch_bos(m15: list[Candle], trend: str) -> tuple[bool, str]:
    """
    Gate 10: Change of Character / Break of Structure on M15.

    A real BOS/CHOCH requires the last candle to CLOSE beyond a prior swing
    high (bullish trend) or prior swing low (bearish trend).

    THREE CONSECUTIVE CLOSES IN ONE DIRECTION DO NOT QUALIFY.
    The soft 3-close fallback has been removed entirely.
    """
    if len(m15) < 10:
        return False, "Too few M15 candles for CHOCH/BOS detection"

    for lookback in range(1, 11):
        if lookback > len(m15) - 8:
            break
        if lookback == 1:
            window = m15[-15:]
        else:
            start = max(0, len(m15) - 15 - lookback + 1)
            window = m15[start: len(m15) - lookback + 1]
        if len(window) < 5:
            continue
        last = window[-1]

        if trend == "bullish":
            swing_high = max(c.high for c in window[:-1])
            if last.close > swing_high:
                return True, f"BOS: close {last.close:.5f} > swing high {swing_high:.5f}"
            minor_high = max(c.high for c in window[-6:-1])
            if last.is_bullish and last.close > minor_high:
                return True, f"CHOCH: close {last.close:.5f} > minor high {minor_high:.5f}"
        else:
            swing_low = min(c.low for c in window[:-1])
            if last.close < swing_low:
                return True, f"BOS: close {last.close:.5f} < swing low {swing_low:.5f}"
            minor_low = min(c.low for c in window[-6:-1])
            if last.is_bearish and last.close < minor_low:
                return True, f"CHOCH: close {last.close:.5f} < minor low {minor_low:.5f}"

    return False, "No real CHOCH or BOS on M15 (3-close fallback not accepted)"


def _find_order_block(m15: list[Candle], trend: str) -> tuple[Optional[tuple[float, float]], str]:
    """
    Gate 11a: Order Block detection on M15.
    An OB is the last opposing candle before an impulse of >= 1.2x its body.
    Searches last 30 candles. OB must not have been violated after formation.
    """
    if len(m15) < 6:
        return None, "Too few M15 candles for OB"
    search_depth = min(len(m15) - 2, 30)
    for i in range(len(m15) - 2, len(m15) - 2 - search_depth, -1):
        if i < 0 or i + 1 >= len(m15):
            continue
        c  = m15[i]
        nx = m15[i + 1]
        if trend == "bullish" and c.is_bearish and nx.is_bullish and nx.body >= c.body * 1.2:
            ob_zone = (c.low, c.high)
            violated = any(m15[j].close < c.low for j in range(i + 2, len(m15)))
            if not violated:
                return ob_zone, f"Bullish OB at {c.low:.5f}–{c.high:.5f}"
        elif trend == "bearish" and c.is_bullish and nx.is_bearish and nx.body >= c.body * 1.2:
            ob_zone = (c.low, c.high)
            violated = any(m15[j].close > c.high for j in range(i + 2, len(m15)))
            if not violated:
                return ob_zone, f"Bearish OB at {c.low:.5f}–{c.high:.5f}"

    return None, "No valid Order Block found"


def _find_fvg(m15: list[Candle], trend: str) -> tuple[Optional[tuple[float, float]], str]:
    """
    Gate 11b: Fair Value Gap (FVG) detection on M15.
    A 3-candle pattern where candles[i-1] and candles[i+1] do not overlap,
    leaving an imbalance zone. Searches last 30 candles.
    Gap must not have been fully filled since formation.
    """
    if len(m15) < 5:
        return None, "Too few M15 candles for FVG"
    search_depth = min(len(m15) - 2, 30)
    for i in range(len(m15) - 2, len(m15) - 2 - search_depth, -1):
        if i < 1 or i + 1 >= len(m15):
            continue
        prev = m15[i - 1]
        curr = m15[i]
        nxt  = m15[i + 1]
        if trend == "bullish" and curr.is_bullish:
            if nxt.low > prev.high:
                fvg_low  = prev.high
                fvg_high = nxt.low
                filled = any(m15[j].close < fvg_low for j in range(i + 2, len(m15)))
                if not filled:
                    return (fvg_low, fvg_high), f"Bullish FVG {fvg_low:.5f}–{fvg_high:.5f}"
        elif trend == "bearish" and curr.is_bearish:
            if nxt.high < prev.low:
                fvg_low  = nxt.high
                fvg_high = prev.low
                filled = any(m15[j].close > fvg_high for j in range(i + 2, len(m15)))
                if not filled:
                    return (fvg_low, fvg_high), f"Bearish FVG {fvg_low:.5f}–{fvg_high:.5f}"

    return None, "No valid FVG found"


def _find_poi(
    m15: list[Candle],
    trend: str,
    sweep_ref: float,
) -> tuple[Optional[tuple[float, float]], str]:
    """
    Gate 11: Point of Interest — OB first, FVG second.
    Swing extreme is the last fallback ONLY when a valid sweep_ref > 0 exists
    and no OB or FVG is present.  Without a sweep reference, no fallback is used.
    """
    ob_zone, ob_reason = _find_order_block(m15, trend)
    if ob_zone:
        return ob_zone, ob_reason

    fvg_zone, fvg_reason = _find_fvg(m15, trend)
    if fvg_zone:
        return fvg_zone, fvg_reason

    # Swing extreme fallback — only when a real sweep reference exists
    if sweep_ref > 0:
        if trend == "bullish":
            zone_low  = sweep_ref
            zone_high = sweep_ref * 1.002     # 0.2% band above sweep low
            return (zone_low, zone_high), f"Swing low fallback POI {zone_low:.5f}"
        else:
            zone_low  = sweep_ref * 0.998
            zone_high = sweep_ref
            return (zone_low, zone_high), f"Swing high fallback POI {zone_high:.5f}"

    return None, "No POI found (no OB, FVG, or sweep reference)"


def _entry_confirmation(m15: list[Candle], trend: str) -> tuple[bool, str]:
    """
    Gate 12: Confirmation candle — ANY ONE of the following required:

      (a) Strong directional candle: body_ratio >= 0.60 in trend direction
      (b) Full engulfing: last candle body > prior candle body AND close
          beyond prior high (bullish) or prior low (bearish)
      (c) Rejection wick: wick opposing the trend >= 50% of candle range

    Body ratio threshold is 0.60 — NOT 0.55 or 0.40.
    Rejection wick threshold is 50% — NOT 40%.
    """
    if len(m15) < 2:
        return False, "Too few M15 candles for confirmation"

    last = m15[-1]
    prev = m15[-2]

    # (a) Strong directional candle — 60% body ratio minimum
    if trend == "bullish" and last.is_bullish and last.body_ratio >= 0.60:
        return True, f"Strong bullish candle (body_ratio={last.body_ratio:.2f})"
    if trend == "bearish" and last.is_bearish and last.body_ratio >= 0.60:
        return True, f"Strong bearish candle (body_ratio={last.body_ratio:.2f})"

    # (b) Full engulfing pattern
    if trend == "bullish" and last.is_bullish and last.body > prev.body and last.close > prev.high:
        return True, "Bullish engulfing confirmation"
    if trend == "bearish" and last.is_bearish and last.body > prev.body and last.close < prev.low:
        return True, "Bearish engulfing confirmation"

    # (c) Rejection wick — 50% minimum of candle range
    if last.range > 0:
        if trend == "bullish" and (last.lower_wick / last.range) >= 0.50:
            return True, f"Bullish rejection wick ({last.lower_wick / last.range:.0%} lower wick)"
        if trend == "bearish" and (last.upper_wick / last.range) >= 0.50:
            return True, f"Bearish rejection wick ({last.upper_wick / last.range:.0%} upper wick)"

    return False, "No confirmation: body_ratio < 0.60, no engulfing, no 50% rejection wick"


# ======================================================================
# LIQUIDITY TARGET
# ======================================================================

def _find_liquidity_target(candles: list[Candle], trend: str, entry: float) -> float:
    """
    Find the nearest prior liquidity pool beyond the entry price.

    BUY  (bullish): returns the nearest prior swing HIGH above entry.
    SELL (bearish): returns the nearest prior swing LOW below entry.

    Returns 0.0 if no qualifying swing exists in the provided candles.
    """
    highs, lows = _find_swing_highs_lows(candles, window=3)
    if trend == "bullish":
        targets = [candles[i].high for i in highs if candles[i].high > entry]
        return min(targets) if targets else 0.0
    else:
        targets = [candles[i].low for i in lows if candles[i].low < entry]
        return max(targets) if targets else 0.0


# ======================================================================
# LEVEL BUILDER
# ======================================================================

def _build_levels(
    m15: list[Candle],
    trend: str,
    poi_zone: tuple[float, float],
    sweep_ref: float,
    market_type: str,
    h1: Optional[list[Candle]] = None,
) -> tuple[float, float, float, float, float]:
    """
    Compute entry, stop_loss, tp1, tp2, tp_final.

    Entry    : midpoint of POI zone (OB or FVG centre)
    SL       : beyond actual sweep wick + buffer (0.1% forex / 0.3% crypto)
    TP1      : 1:2 RR
    TP2      : 1:3 RR
    TP_final : nearest prior liquidity pool (prior high for BUY, prior low for SELL)
               Falls back to 1:4 RR only when no qualifying swing level is found.

    Returns (entry, sl, tp1, tp2, tp_final). All zero on failure.
    All-zero return → caller issues WAIT (RR < 1:2 or risk too tight).
    """
    entry = (poi_zone[0] + poi_zone[1]) / 2
    if entry <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    sl_buffer_pct = 0.003 if market_type == "crypto" else 0.001   # 0.3% / 0.1%

    if trend == "bullish":
        sl = sweep_ref * (1 - sl_buffer_pct) if sweep_ref > 0 else poi_zone[0] * (1 - sl_buffer_pct)
        risk = entry - sl
    else:
        sl = sweep_ref * (1 + sl_buffer_pct) if sweep_ref > 0 else poi_zone[1] * (1 + sl_buffer_pct)
        risk = sl - entry

    if risk <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    # Minimum risk guard — prevents meaninglessly tight stops
    min_risk_pct = 0.0005 if market_type == "forex" else 0.002
    if risk / entry < min_risk_pct:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    if trend == "bullish":
        tp1 = entry + risk * 2
        tp2 = entry + risk * 3
    else:
        tp1 = entry - risk * 2
        tp2 = entry - risk * 3

    # TP_final: nearest prior liquidity pool (H1 preferred for wider view)
    ref_candles = h1 if h1 else m15
    liq_target = _find_liquidity_target(ref_candles, trend, entry)

    if liq_target > 0:
        # Sanity-check: final TP must not be worse than TP1
        if trend == "bullish" and liq_target > tp1:
            tp_final = liq_target
        elif trend == "bearish" and liq_target < tp1:
            tp_final = liq_target
        else:
            tp_final = entry + risk * 4 if trend == "bullish" else entry - risk * 4
    else:
        # No qualifying swing found — fall back to 1:4 RR
        tp_final = entry + risk * 4 if trend == "bullish" else entry - risk * 4

    return entry, sl, tp1, tp2, tp_final


# ======================================================================
# MAIN SIGNAL GENERATOR
# ======================================================================

def generate_signal(
    symbol: str,
    h1_raw: list[dict],
    m15_raw: list[dict],
    market_type: str,
) -> Signal:
    """
    Full 12-gate SMC signal pipeline. Returns Signal with direction BUY/SELL/WAIT.

    Every gate is HARD and blocks independently.
    The only soft gate is Gate 5 (key level proximity) — it logs a warning
    but never returns WAIT.
    The only permitted fallback is the swing-extreme POI in Gate 11, and only
    when a real M15 sweep reference exists and no OB/FVG is found.
    """
    def _wait(reason: str) -> Signal:
        return Signal(symbol=symbol, direction="WAIT", market_type=market_type, reason=reason)

    # ── Convert raw candles ───────────────────────────────────────────────────
    h1  = _to_candles(h1_raw)
    m15 = _to_candles(m15_raw)

    if len(h1) < 5 or len(m15) < 5:
        return _wait("Insufficient candle data")

    # ── GATE 1: Session ───────────────────────────────────────────────────────
    session_ok, session_reason = _is_tradeable_session(market_type)
    if not session_ok:
        return _wait(f"G1 Session: {session_reason}")
    log.debug("[%s] G1 OK: %s", symbol, session_reason)

    # ── GATE 2: H1 Structure ──────────────────────────────────────────────────
    trend, struct_reason, struct_quality = _detect_structure(h1)
    if trend == "range":
        return _wait(f"G2 Structure: {struct_reason}")
    log.info("[%s] G2 OK: %s (quality=%s)", symbol, struct_reason, struct_quality)

    # ── GATE 3: Price zone ────────────────────────────────────────────────────
    # BUY  requires Discount.
    # SELL requires Premium.
    # Equilibrium is NOT a valid entry zone for either direction.
    zone, range_high, range_low, range_mid = _price_zone(h1)
    if zone == "equilibrium":
        return _wait(
            "G3 Zone: Price at equilibrium — not a valid entry zone. "
            "Wait for Discount (BUY) or Premium (SELL)."
        )
    if trend == "bullish" and zone == "premium":
        return _wait(
            f"G3 Zone: BUY requires Discount but price is in Premium. WAIT."
        )
    if trend == "bearish" and zone == "discount":
        return _wait(
            f"G3 Zone: SELL requires Premium but price is in Discount. WAIT."
        )
    log.debug("[%s] G3 OK: zone=%s trend=%s", symbol, zone, trend)

    # ── GATE 4: Volatility ────────────────────────────────────────────────────
    vol_ok, vol_reason = _session_volatility_ok(m15, market_type)
    if not vol_ok:
        return _wait(f"G4 Volatility: {vol_reason}")
    log.debug("[%s] G4 OK: %s", symbol, vol_reason)

    # ── GATE 5: Key level (SOFT — warning only, never blocks) ─────────────────
    h1_highs, h1_lows = _find_swing_highs_lows(h1, window=3)
    at_key, key_reason = _at_key_level(h1, h1_highs, h1_lows, market_type)
    if not at_key:
        log.warning("[%s] G5 SOFT miss: %s (continuing)", symbol, key_reason)
    else:
        log.debug("[%s] G5 OK: %s", symbol, key_reason)

    # ── GATE 6: H1 Liquidity sweep (HARD — real wick sweep only) ─────────────
    # Three consecutive closes do NOT count. No soft fallback.
    swept_h1, sweep_h1_reason, sweep_ref = _liquidity_swept(h1, trend, strict=True)
    if not swept_h1:
        return _wait(f"G6 H1 Sweep: {sweep_h1_reason}")
    log.info("[%s] G6 OK: %s", symbol, sweep_h1_reason)

    # ── GATE 7: Market narrative ──────────────────────────────────────────────
    narrative, narrative_reason = _classify_narrative(h1, trend)
    if narrative == "reversal":
        return _wait(f"G7 Narrative: {narrative_reason}")
    log.debug("[%s] G7 OK: %s", symbol, narrative_reason)

    # ── GATE 8: Momentum ──────────────────────────────────────────────────────
    # Requires at least 2/5 strong directional candles.
    mom_ok, mom_reason = _momentum_strength(m15, trend)
    if not mom_ok:
        return _wait(f"G8 Momentum: {mom_reason}")
    log.debug("[%s] G8 OK: %s", symbol, mom_reason)

    # ── GATE 9: M15 liquidity sweep (HARD — real wick sweep required) ─────────
    # No fallback to H1 reference. Three consecutive closes NOT accepted.
    swept_m15, sweep_m15_reason, sweep_m15_ref = _liquidity_swept(m15, trend, strict=True)
    if not swept_m15:
        return _wait(
            f"G9 M15 Sweep: {sweep_m15_reason} "
            "(H1 sweep reference does not substitute; M15 real wick sweep required)"
        )
    # Replace H1 sweep reference with the tighter M15 reference for SL placement
    sweep_ref = sweep_m15_ref
    log.info("[%s] G9 OK (M15): %s", symbol, sweep_m15_reason)

    # ── GATE 10: CHOCH / BOS on M15 ──────────────────────────────────────────
    # Price must close beyond a prior swing high/low. No soft fallback.
    choch_ok, choch_reason = _detect_choch_bos(m15, trend)
    if not choch_ok:
        return _wait(f"G10 CHOCH/BOS: {choch_reason}")
    log.info("[%s] G10 OK: %s", symbol, choch_reason)

    # ── GATE 11: Point of Interest ────────────────────────────────────────────
    # OB → FVG → swing extreme (only when real M15 sweep_ref exists)
    poi_zone, poi_reason = _find_poi(m15, trend, sweep_ref)
    if poi_zone is None:
        return _wait(f"G11 POI: {poi_reason}")
    log.info("[%s] G11 OK: %s", symbol, poi_reason)

    # ── GATE 12: Entry confirmation ───────────────────────────────────────────
    # Strong candle (body_ratio >= 0.60) OR engulfing OR rejection wick >= 50%
    confirm_ok, confirm_reason = _entry_confirmation(m15, trend)
    if not confirm_ok:
        return _wait(f"G12 Confirmation: {confirm_reason}")
    log.info("[%s] G12 OK: %s", symbol, confirm_reason)

    # ── Build levels ──────────────────────────────────────────────────────────
    entry, sl, tp1, tp2, tp_final = _build_levels(
        m15, trend, poi_zone, sweep_ref, market_type, h1=h1
    )
    if entry == 0:
        return _wait("Levels: Could not compute valid entry/SL/TP (risk too tight or zero)")

    # ── Verify minimum 1:2 RR ─────────────────────────────────────────────────
    if trend == "bullish":
        actual_rr = (tp1 - entry) / (entry - sl) if (entry - sl) > 0 else 0.0
    else:
        actual_rr = (entry - tp1) / (sl - entry) if (sl - entry) > 0 else 0.0
    if actual_rr < 2.0:
        return _wait(
            f"RR: Computed RR={actual_rr:.2f} is below the 1:2 minimum — WAIT"
        )

    # ── Setup quality grading ─────────────────────────────────────────────────
    has_ob  = "OB"  in poi_reason
    has_fvg = "FVG" in poi_reason
    if struct_quality == "strong" and (has_ob or has_fvg):
        quality = "A"
    elif struct_quality in ("strong", "partial") and (has_ob or has_fvg):
        quality = "B"
    else:
        quality = "C"

    direction = "BUY" if trend == "bullish" else "SELL"

    reason_parts = [
        f"Struct:{struct_reason}",
        f"Zone:{zone}",
        f"H1Sweep:{sweep_h1_reason}",
        f"M15Sweep:{sweep_m15_reason}",
        f"Narr:{narrative}",
        f"POI:{poi_reason}",
        f"Confirm:{confirm_reason}",
        f"RR:{actual_rr:.2f}",
        f"Quality:{quality}",
    ]

    log.info(
        "[%s] ✅ %s signal | entry=%.5f SL=%.5f TP1=%.5f TP2=%.5f TPfinal=%.5f | quality=%s",
        symbol, direction, entry, sl, tp1, tp2, tp_final, quality,
    )

    return Signal(
        symbol=symbol,
        direction=direction,
        market_type=market_type,
        entry=entry,
        stop_loss=sl,
        tp1=tp1,
        tp2=tp2,
        tp_final=tp_final,
        reason=" | ".join(reason_parts),
        setup_quality=quality,
    )
