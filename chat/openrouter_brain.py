"""
openrouter_brain.py — AI-Powered Chat Using OpenRouter API
============================================================
PRODUCTION VERSION:
- OpenRouter AI is the SOLE responder for all messages
- Static/template fallbacks are COMPLETELY DISABLED
- Static responses only activate if the API key is missing OR the API is down
- Admin recognition: different system prompt for admin users
- User memory: includes returning-user context in prompts
- Vision API: analyzes images via OpenRouter vision-capable models
"""
from __future__ import annotations

import base64
import json
import os
import logging
from typing import Optional

import requests

from core.logger import get_logger

log = get_logger(__name__)

# ── API Configuration ──────────────────────────────────────────────────────────
OPENROUTER_API_KEY   = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL     = os.getenv("OPENROUTER_MODEL", "openrouter/auto")
OPENROUTER_VIS_MODEL = os.getenv("OPENROUTER_VISION_MODEL", "openai/gpt-4o-mini")

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_HEADERS_BASE = {
    "HTTP-Referer": "https://trading-signals.local",
    "X-Title": "Trading Signal Bot",
    "Content-Type": "application/json",
}

if OPENROUTER_API_KEY:
    log.info("✅ OPENROUTER_API_KEY loaded")
else:
    log.warning("❌ OPENROUTER_API_KEY not set — AI responses disabled until key is configured")

# ── System prompts ─────────────────────────────────────────────────────────────
_SYSTEM_USER = """You are a helpful trading signal assistant for a Smart Money Concepts (SMC) trading platform.

Your role:
- Explain how trading signals work
- Answer questions about the subscription/pricing
- Describe the platform's strategy and markets
- Guide users through the payment process
- Be warm, professional, and encouraging

Key platform facts:
- Strategy: Smart Money Concepts (SMC) multi-timeframe analysis
- Markets: Forex (EUR/USD, GBP/USD, USD/JPY, AUD/USD, USD/CHF, USD/CAD, NZD/USD) & Crypto (BTC/USD, ETH/USD, SOL/USD)
- Signals: Entry, Stop Loss, TP1 (1:2 R:R), TP2 (1:3 R:R), Final TP
- Delivery: Private Telegram channel
- Quality: 14-step validation filter — only high-quality setups pass
- Price: ₦10,000 for 30 days | Bank: Moniepoint | Account: 6576999590 | Name: Isreal Bethel Ojotule
- Minimum R:R: 1:2 on every trade, no exceptions

Rules:
- NEVER share actual live signals or real-time market data
- NEVER reveal signal details before a user subscribes
- NEVER predict market direction as a real forecast
- Always encourage subscribing with /pay
- Keep responses under 300 words
- Use friendly emojis occasionally (not excessively)
- Always end with a soft call-to-action when relevant
- If someone has a small budget (₦2,000–₦5,000), be encouraging — explain micro-lots and position sizing

Topic guides:
- "How do signals work?" → Explain the 6-step SMC process (Order Blocks, FVGs, CHOCH, BOS, liquidity sweeps)
- "What's the cost?" → ₦10,000/month, Moniepoint 6576999590
- "Can I see an example?" → Yes, after they subscribe
- "Why should I trust you?" → Explain the 14-step validation, risk management, R:R discipline
- "How do I pay?" → Transfer to Moniepoint 6576999590, then send screenshot via /pay
- "I have small budget" → Be encouraging, explain micro-lots and scaling"""

_SYSTEM_ADMIN = """You are a helpful assistant for the ADMIN of this trading signal platform.

The person you're talking to is the platform owner and administrator. They have full control.

Admin commands available:
- /pending — View all pending payment submissions
- /approve <user_id> — Approve payment and send private channel invite
- /reject <user_id> — Reject a payment submission

Platform facts:
- Strategy: Smart Money Concepts (SMC)
- Price: ₦10,000 / 30 days | Moniepoint 6576999590 | Isreal Bethel Ojotule
- Private channel invite is sent automatically on /approve

Be concise, technical, and helpful. The admin may ask about:
- System status and signals
- How to manage subscribers
- Payment processing
- Bot configuration and features

Always be direct and professional."""

_SYSTEM_VISION = """You are a payment verification assistant for a trading signal platform.

Your ONLY job is to analyze the image provided and determine:
1. Is this a bank payment/transfer receipt/screenshot? (YES or NO)
2. If YES: What is the transfer amount? (look for amounts like ₦10,000, 10000, etc.)
3. If YES: Is the amount at least ₦10,000 (the required subscription fee)?

Respond ONLY in JSON format like this:
{
  "is_payment_screenshot": true,
  "amount_detected": 10000,
  "amount_sufficient": true,
  "currency": "NGN",
  "bank_name": "Moniepoint",
  "description": "Brief description of what you see"
}

If it is NOT a payment screenshot, respond:
{
  "is_payment_screenshot": false,
  "amount_detected": null,
  "amount_sufficient": false,
  "currency": null,
  "bank_name": null,
  "description": "Brief description of what the image actually shows"
}

Required subscription amount: ₦10,000"""


def _make_headers() -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    return {**_HEADERS_BASE, "Authorization": f"Bearer {api_key}"}


def _call_openrouter(user_text: str, user_id: int, system_prompt: str) -> Optional[str]:
    """Call OpenRouter API. Returns None on any failure."""
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        return None

    model = os.getenv("OPENROUTER_MODEL") or OPENROUTER_MODEL

    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_text},
            ],
            "temperature": 0.7,
            "max_tokens": 500,
            "top_p": 0.95,
        }

        resp = requests.post(_API_URL, json=payload, headers=_make_headers(), timeout=20)

        if resp.status_code != 200:
            log.warning("OpenRouter error: HTTP %s — %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "").strip()
            if content:
                log.info("✅ OpenRouter response for user %s (%d chars)", user_id, len(content))
                return content

        log.warning("OpenRouter: empty/malformed response: %s", data)
        return None

    except requests.exceptions.Timeout:
        log.warning("OpenRouter timeout for user %s", user_id)
        return None
    except requests.exceptions.ConnectionError:
        log.warning("OpenRouter connection error for user %s", user_id)
        return None
    except Exception as exc:
        log.error("OpenRouter error for user %s: %s", user_id, exc)
        return None


def analyze_payment_image(img_bytes: bytes) -> dict:
    """
    Use OpenRouter vision to analyze a payment screenshot.
    Returns a dict with keys: is_payment_screenshot, amount_detected, amount_sufficient, description, etc.
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        log.warning("No API key — cannot analyze payment image")
        return {
            "is_payment_screenshot": True,  # Assume valid so admin can review
            "amount_detected": None,
            "amount_sufficient": None,
            "description": "Image received (AI analysis unavailable — no API key)",
        }

    try:
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        vision_model = os.getenv("OPENROUTER_VISION_MODEL") or OPENROUTER_VIS_MODEL

        payload = {
            "model": vision_model,
            "messages": [
                {"role": "system", "content": _SYSTEM_VISION},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                        {
                            "type": "text",
                            "text": "Analyze this image. Is it a bank payment receipt/screenshot?",
                        },
                    ],
                },
            ],
            "max_tokens": 300,
            "temperature": 0.1,
        }

        resp = requests.post(_API_URL, json=payload, headers=_make_headers(), timeout=30)

        if resp.status_code != 200:
            log.warning("Vision API error: HTTP %s", resp.status_code)
            return {"is_payment_screenshot": True, "amount_detected": None, "amount_sufficient": None,
                    "description": "Image received (vision analysis failed — will be reviewed by admin)"}

        data = resp.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not content:
            return {"is_payment_screenshot": True, "amount_detected": None, "amount_sufficient": None,
                    "description": "Image received (empty vision response — admin will review)"}

        # Strip markdown fences if present
        clean = content.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)
        log.info("✅ Payment image analyzed: %s", result)
        return result

    except json.JSONDecodeError:
        log.warning("Vision response was not valid JSON: %s", content[:200] if 'content' in dir() else "N/A")
        return {"is_payment_screenshot": True, "amount_detected": None, "amount_sufficient": None,
                "description": "Image received (could not parse AI response — admin will review)"}
    except Exception as exc:
        log.error("analyze_payment_image error: %s", exc)
        return {"is_payment_screenshot": True, "amount_detected": None, "amount_sufficient": None,
                "description": "Image received (analysis error — admin will review)"}


def get_response(
    user_text: str,
    user_id: int,
    is_admin: bool = False,
    is_new: bool = False,
    last_topic: str = None,
) -> str:
    """
    Get AI response from OpenRouter.
    
    IMPORTANT: Static/template fallbacks are DISABLED.
    - If the API key is set: ALWAYS returns an AI response, or a brief error message on API failure.
    - If the API key is NOT set: Returns a simple "AI unavailable" notice.
    
    Never mixes AI responses with pre-written template blocks.
    """
    if not user_text or not user_text.strip():
        return "👋 I'm here! Ask me about signals, pricing, or how to subscribe."

    user_text = user_text.strip()

    # /start command: ask AI to write a welcome message
    if user_text == "/start":
        user_text = "Write a warm welcome message introducing yourself as the trading signal assistant for this SMC platform. Keep it concise and friendly."

    # Choose system prompt
    system = _SYSTEM_ADMIN if is_admin else _SYSTEM_USER

    # Add returning-user context if applicable
    if not is_admin and not is_new and last_topic:
        context_note = f"\n\n[Context: This is a returning user. Their last conversation was about: {last_topic.replace('_', ' ')}]"
        system = system + context_note

    # API key check
    api_key = os.getenv("OPENROUTER_API_KEY") or OPENROUTER_API_KEY
    if not api_key:
        log.warning("OPENROUTER_API_KEY not set — returning AI-unavailable message")
        return (
            "⚠️ *AI assistant is temporarily offline.*\n\n"
            "You can still:\n"
            "• Type /pay to get payment details and subscribe\n"
            "• Contact our admin directly for help\n\n"
            "We'll be back shortly!"
        )

    # Call OpenRouter
    response = _call_openrouter(user_text, user_id, system)

    if response:
        return response

    # API failed — return a minimal error, NOT template responses
    log.warning("OpenRouter API failed for user %s — returning fallback error message", user_id)
    return (
        "⚠️ I'm having a brief technical issue right now.\n\n"
        "Please try again in a moment, or type /pay to see subscription details.\n"
        "You can also contact our admin directly if you need urgent help."
    )


def get_response_with_image(img_bytes: bytes, caption: str, user_id: int) -> str:
    """Analyze an uploaded image with context (for web chat)."""
    result = analyze_payment_image(img_bytes)

    if result.get("is_payment_screenshot"):
        amount = result.get("amount_detected")
        sufficient = result.get("amount_sufficient")
        from core.config import PAYMENT_AMOUNT
        required = PAYMENT_AMOUNT

        if amount is not None and not sufficient:
            return (
                f"⚠️ *Payment amount too low*\n\n"
                f"The amount I detected in your screenshot is *₦{amount:,}*, "
                f"but our subscription requires *₦{required:,}*.\n\n"
                f"Please make sure you transfer the full amount to:\n"
                f"🏦 Moniepoint | Account: 6576999590 | Name: Isreal Bethel Ojotule\n\n"
                f"Once you've made the correct payment, send the new screenshot."
            )
        return (
            "✅ *Payment screenshot received!*\n\n"
            "Your image has been forwarded to our admin for verification. "
            "You'll be notified once your subscription is approved (usually within a few hours).\n\n"
            "Questions? Just ask!"
        )
    else:
        desc = result.get("description", "")
        return (
            "❌ *This doesn't look like a payment screenshot.*\n\n"
            f"{('I can see: ' + desc + chr(10) + chr(10)) if desc else ''}"
            "Please send a *screenshot of your bank transfer receipt* to complete your subscription.\n\n"
            "Make the payment to:\n"
            "🏦 Moniepoint | Account: 6576999590 | Name: Isreal Bethel Ojotule | Amount: ₦10,000\n\n"
            "Then send the transfer screenshot here."
)
