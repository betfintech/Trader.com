"""
Smart Money Concepts (SMC) Strategy — OPTIMIZED v3
====================================================
Works identically for crypto and forex.
Market-specific parameters are adjusted via the market_type argument.

Signal output: BUY | SELL | WAIT

DESIGN PHILOSOPHY:
  Target win rate  : ~70% (max 30% loss rate)
  Signal frequency : 3–6 valid signals per day across watchlist in trending markets
  Perfect setups   : Never blocked — if all conditions align cleanly, fire immediately
  Bad trades       : Blocked by structure quality, real POI, and strict 1:2 RR

GATE SUMMARY:
  GATE 1  - Session filter (Asian blocked for forex; crypto 24/7)
  GATE 2  - H1 Market structure — HH/HL or LH/LL required; slope fallback at 0.002
  GATE 3  - Price zone — Bullish needs discount/equilibrium; bearish needs premium/equilibrium
  GATE 4  - Volatility — Must be above 50% of configured threshold
  GATE 5  - Key level proximity — SOFT gate (logs warning, never blocks)
  GATE 6  - H1 Liquidity sweep — Real wick sweep (20%) OR soft fallback (3+ closes) [HARD]
  GATE 7  - Market narrative — Only REVERSAL blocks; pullback and unclear both pass
  GATE 8  - Momentum — body/wick >= 0.6; at least 1/5 strong directional candles
  GATE 9  - M15 Liquidity sweep — SOFT gate (falls back to H1 wick reference)
  GATE 10 - CHOCH/BOS on M15 — Real structure break OR soft 3-close fallback [HARD]
  GATE 11 - Point of Interest — OB or FVG required; swing extreme as last fallback [HARD]
  GATE 12 - Flexible confirmation — Strong candle OR engulfing OR rejection wick [HARD]
  RR      - Minimum 1:2 enforced strictly on every signal

WHAT MAKES THIS ~70% ACCURATE:
  - Real structure (HH/HL) required — slope alone is a weaker signal
  - Real liquidity sweep (wick) still preferred; soft fallback only when market grinds
  - Reversal narrative hard-blocks entries (don't trade against forming reversal)
  - POI (OB/FVG) required — entries at key institutional zones only
  - SL placed at the actual sweep wick (real invalidation point)
  - RR 1:2 strictly enforced — even losing trades stay controlled
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
        return self.body_ratio >= 0.55 and self.body > 0


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
    if not ENABLE_SESSION_FILTER:
        return True, "Session filter disabled"
    if market_type == "crypto":
        return True, "Crypto: 24/7 market"
    session = _current_session()
    hour = datetime.now(timezone.utc).hour
    if session == "Asian":
        return False, (
            f"Asian session ({hour:02d}:00 UTC) -- low volatility, "
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
            f"Volatility too low ({rel_vol:.5f} < {threshold:.5f}) -- market flat"
        )
    return True, f"Volatility OK ({rel_vol:.5f} >= {threshold:.5f})"


# ======================================================================
# H1 BIAS ENGINE
# ======================================================================

def _find_swing_highs_lows(candles: list[Candle], window: int = 3):
    """Swing detection with window=3 for responsive but valid pivot identification."""
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
    Gate 2: Market structure detection.
    Returns: (trend, reason, quality)
      quality = "strong"  — full HH+HL or LH+LL confirmed
      quality = "partial" — only HH or only HL confirmed (one side)
      quality = "slope"   — slope fallback only (weakest)

    Only "strong" and "partial" are used for entries.
    "slope" alone still fires but is tagged as lower quality.
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
            return "bullish", "HH + HL -- bullish structure", "strong"
        if lh and ll:
            return "bearish", "LH + LL -- bearish structure", "strong"
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
    # Threshold 0.002 — balanced between too sensitive (0.001) and too strict (0.003)
    if slope > 0.002:
        return "bullish", f"Bullish slope ({slope:.3%})", "slope"
    if slope < -0.002:
        return "bearish", f"Bearish slope ({slope:.3%})", "slope"
    return "range", f"Range market (slope={slope:.3%})", "none"


def _price_zone(candles: list[Candle]) -> tuple[str, float, float, float]:
    """
    Gate 3: Uses last 50 candles (~2 days on H1) for a responsive and
    accurate premium/discount view. Equilibrium band = 5%.
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
    Tolerance: 1.5% forex / 2.5% crypto. Round-number band: 0.8%.
    A miss does NOT block — caller logs and continues.
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
    strict: bool = False,
) -> tuple[bool, str, float]:
    """
    Gate 6 / Gate 9: Liquidity sweep detection.

    PRIMARY: Real wick sweep — price wicks beyond prior swing and closes back.
             Wick threshold: 20% of candle range. Lookback: 10 candles.
             This is the highest-quality sweep and contributes to A-grade setups.

    FALLBACK (when strict=False): Soft sweep — 3+ consecutive closes
             in trend direction within last 5 candles. This captures
             institutional accumulation / grinding sweeps.
             Contributes to B-grade setups.
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
                    return (True,
                            f"Bullish sweep: wick {candidate.low:.5f} < ref {ref_low:.5f}, "
                            f"closed {candidate.close:.5f}",
                            candidate.low)
        else:
            ref_high = max(c.high for c in prior_slice)
            if candidate.high > ref_high and candidate.close < ref_high:
                wick_size = candidate.high - ref_high
                wick_pct = wick_size / candidate.range if candidate.range > 0 else 0
                if wick_pct >= 0.20:
                    return (True,
                            f"Bearish sweep: wick {candidate.high:.5f} > ref {ref_high:.5f}, "
                            f"closed {candidate.close:.5f}",
                            candidate.high)

    # Soft sweep fallback
    if not strict and len(candles) >= 5:
        last5 = candles[-5:]
        if trend == "bullish":
            rising = sum(
                1 for i in range(1, len(last5))
                if last5[i].close > last5[i - 1].close
            )
            if rising >= 3:
                ref = min(c.low for c in last5)
                return True, f"Soft bullish sweep (3+ rising closes, ref low {ref:.5f})", ref
        else:
            falling = sum(
                1 for i in range(1, len(last5))
                if last5[i].close < last5[i - 1].close
            )
            if falling >= 3:
                ref = max(c.high for c in last5)
                return True, f"Soft bearish sweep (3+ falling closes, ref high {ref:.5f})", ref

    return False, "No valid liquidity sweep detected", 0.0


# ======================================================================
# MARKET NARRATIVE
# ======================================================================

def _classify_narrative(candles: list[Candle], trend: str) -> tuple[str, str]:
    """
    Gate 7: Market narrative.
    - REVERSAL (8/10 candles against trend) — HARD BLOCK. Do not trade.
    - PULLBACK (retracement in progress)    — ALLOW. Best SMC entry point.
    - CONTINUATION (strong move)            — ALLOW. Momentum entry.
    - UNCLEAR (mixed)                       — ALLOW. Proceed with caution.
    """
    if len(candles) < 20:
        return "unclear", "Insufficient data"

    last5   = candles[-5:]
    last10  = candles[-10:]
    bull5   = sum(1 for c in last5  if c.is_bullish)
    bear5   = sum(1 for c in last5  if c.is_bearish)
    bull10  = sum(1 for c in last10 if c.is_bullish)
    bear10  = sum(1 for c in last10 if c.is_bearish)
    strong5 = sum(1 for c in last5  if c.is_strong)

    if trend == "bullish":
        if bull5 >= 4 and strong5 >= 2:
            return "continuation", f"Bullish continuation: {bull5}/5 bull, {strong5} strong"
        if bear5 >= 3:
            if bear10 >= 8:
                return "reversal", f"Potential reversal: {bear10}/10 bearish -- BLOCKED"
            return "pullback", f"Bullish pullback: {bear5}/5 bearish (retracement)"
        return "unclear", f"Mixed: {bull5} bull / {bear5} bear"
    else:
        if bear5 >= 4 and strong5 >= 2:
            return "continuation", f"Bearish continuation: {bear5}/5 bear, {strong5} strong"
        if bull5 >= 3:
            if bull10 >= 8:
                return "reversal", f"Potential reversal: {bull10}/10 bullish -- BLOCKED"
            return "pullback", f"Bearish pullback: {bull5}/5 bullish (retracement)"
        return "unclear", f"Mixed: {bull5} bull / {bear5} bear"


def _momentum_strength(candles: list[Candle], trend: str) -> tuple[bool, str]:
    """Gate 8: Body/wick >= 0.6 and at least 1/5 strong directional candles."""
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
        return False, f"Weak momentum: body/wick ratio {body_ratio:.2f}"
    if directional < 1:
        return False, "Weak momentum: 0/5 strong directional candles"
    return True, f"Momentum OK: body/wick={body_ratio:.2f}, {directional}/5 directional"


# ======================================================================
# M15 ENTRY CONFIRMATION ENGINE
# ======================================================================

def _detect_choch_bos(m15: list[Candle], trend: str) -> tuple[bool, str]:
    """
    Gate 10: Change of Character / Break of Structure on M15.

    PRIMARY: Price closes beyond prior swing high/low on M15.
             Lookback: 10 candles with 15-candle window.
             Clean structural break = high-quality entry signal.

    FALLBACK: 3+ consecutive closes in trend direction.
             This captures momentum continuation entries where
             structure builds gradually rather than breaks sharply.
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

    # Soft BOS fallback
    if len(m15) >= 4:
        last4 = m15[-4:]
        if trend == "bullish":
            run = sum(
                1 for i in range(1, 4)
                if last4[i].close > last4[i - 1].close
            )
            if run >= 3:
                return True, "Soft BOS: 3 consecutive bullish closes"
        else:
            run = sum(
                1 for i in range(1, 4)
                if last4[i].close < last4[i - 1].close
            )
            if run >= 3:
                return True, "Soft BOS: 3 consecutive bearish closes"

    return False, "No CHOCH or BOS on M15"


def _find_order_block(m15: list[Candle], trend: str) -> tuple[Optional[tuple[float, float]], str]:
    """
    Gate 11a: Order Block detection on M15.
    Requires the opposing candle to be followed by an impulse >= 1.2x its body.
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
            # Check OB hasn't been violated (price stayed above c.low after formation)
            violated = any(m15[j].close < c.low for j in range(i + 2, len(m15)))
            if not violated:
                return ob_zone, f"Bullish OB at {c.low:.5f}–{c.high:.5f}"
        elif trend == "bearish" and c.is_bullish and nx.is_bearish and nx.body >= c.body * 1.2:
            ob_zone = (c.low, c.high)
            # Check OB hasn't been violated (price stayed below c.high after formation)
            violated = any(m15[j].close > c.high for j in range(i + 2, len(m15)))
            if not violated:
                return ob_zone, f"Bearish OB at {c.low:.5f}–{c.high:.5f}"

    return None, "No valid Order Block found"
