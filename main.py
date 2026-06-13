# === SECTION 1: Imports & Config ===

import os
import json
import textwrap
import asyncio
import traceback
import uuid
import logging
import re
import shlex
import subprocess
import tempfile
import time
import hmac
import html
import sys
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Set, Callable
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import asyncpg
import httpx
import smtplib
import ssl as ssl_lib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email import encoders
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File, Form, Header, Request, Response
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jaxvora")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEEPSEEK_V4_API_KEY = os.environ.get("DEEPSEEK_V4_API_KEY", "")

# OpenCode Zen — free/unlimited DeepSeek V4 Flash, used as the agents' brain.
OPENCODE_ZEN_API_KEY = os.environ.get("OPENCODE_ZEN_API_KEY", "")
OPENCODE_ZEN_BASE = os.environ.get("OPENCODE_ZEN_BASE", "https://opencode.ai/zen/v1")
OPENCODE_ZEN_MODEL = os.environ.get("OPENCODE_ZEN_MODEL", "deepseek-v4-flash-free")
# When the key is present, route every agent LLM call through OpenCode Zen
# (override with OPENCODE_ZEN_PRIMARY=0 to fall back to per-agent providers).
OPENCODE_ZEN_PRIMARY = bool(OPENCODE_ZEN_API_KEY) and os.environ.get("OPENCODE_ZEN_PRIMARY", "1") not in ("0", "false", "False", "no")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
PORT = int(os.environ.get("PORT", 8080))

GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "")
GMAIL_AUTOMATION_USER = os.environ.get("GMAIL_AUTOMATION_USER", os.environ.get("GMAIL_USER", ""))
GMAIL_CLIENT_ID = os.environ.get("GMAIL_CLIENT_ID", "")
GMAIL_CLIENT_SECRET = os.environ.get("GMAIL_CLIENT_SECRET", "")
GMAIL_REFRESH_TOKEN = os.environ.get("GMAIL_REFRESH_TOKEN", "")
GMAIL_AUTOMATION_API_TOKEN = os.environ.get("GMAIL_AUTOMATION_API_TOKEN", os.environ.get("JAXVORA_ADMIN_TOKEN", ""))
GMAIL_TEMPLATE_FILE = os.environ.get("GMAIL_TEMPLATE_FILE", "")
GMAIL_LOGO_FILE = os.environ.get("GMAIL_LOGO_FILE", "")
GMAIL_LOGO_CID = "jaxvora-logo"
GMAIL_SCOPES = os.environ.get(
    "GMAIL_SCOPES",
    "https://mail.google.com/",
)

SSH_HOST = os.environ.get("SSH_HOST", "")
SSH_PORT = int(os.environ.get("SSH_PORT", "22"))
SSH_USER = os.environ.get("SSH_USER", "")
SSH_PASSWORD = os.environ.get("SSH_PASSWORD", "")
SSH_KEY = os.environ.get("SSH_KEY", "")
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH", "")

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
DEEPSEEK_MODEL = "deepseek/deepseek-chat-v3-0324"
LLAMA_MODEL = "llama-3.3-70b-versatile"
DEEPSEEK_V4_MODEL = "deepseek/deepseek-v4-flash:free"
DEEPSEEK_V4_BASE = "https://openrouter.ai/api/v1"
MAX_TOKENS = 8192
DEEPSEEK_V4_MAX_TOKENS = 64000
# Default output budget per LLM call. 64000 was wasteful (slow + free-tier rejections).
# Keep it modest; callers that truly need more can pass a larger max_tokens.
DEFAULT_MAX_TOKENS = int(os.environ.get("LLM_DEFAULT_MAX_TOKENS", "4096") or 4096)
# Per-provider output caps — free tiers can't serve huge outputs (OpenRouter 402:
# "you requested up to N tokens but can only serve fewer"; Groq/Zen TPM rate limits).
ZEN_MAX_OUT = int(os.environ.get("ZEN_MAX_TOKENS", "8192") or 8192)
GROQ_MAX_OUT = int(os.environ.get("GROQ_MAX_TOKENS", "8192") or 8192)
DEEPSEEK_V4_MAX_OUT = int(os.environ.get("DEEPSEEK_V4_MAX_TOKENS_OUT", "2048") or 2048)
OPENROUTER_MAX_OUT = int(os.environ.get("OPENROUTER_MAX_TOKENS", "1024") or 1024)
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_UPLOAD_TEXT_CHARS = 24000

# ── LLM helpers ────────────────────────────────────────────────────────────────

# ── Multi-provider LLM failover ─────────────────────────────────────────────
# Every agent LLM call flows through call_llm_failover(), which walks an ordered
# chain of providers and uses the first that works. If a provider errors or rate
# -limits, it is tripped into a short cooldown and the call automatically shifts
# to the next provider — so no single provider being down can stall the agents.
# In the spirit of the evidence-driven rule: when ALL providers fail we return an
# explicit error, never a fabricated answer.

PROVIDER_COOLDOWN_SECONDS = float(os.environ.get("LLM_PROVIDER_COOLDOWN", "45") or 45)
# 429 recovers fast — bench briefly so we don't pile onto the other free providers.
RATE_LIMIT_COOLDOWN_SECONDS = float(os.environ.get("LLM_RATE_LIMIT_COOLDOWN", "12") or 12)
# 401/402/403 won't recover without a key/credits fix — bench long so the chain stops
# wasting failover attempts on a permanently-broken provider.
AUTH_COOLDOWN_SECONDS = float(os.environ.get("LLM_AUTH_COOLDOWN", "1800") or 1800)
# name -> unix ts until which the provider is skipped after a failure.
_PROVIDER_COOLDOWN: Dict[str, float] = {}

# Cap simultaneous outbound LLM calls so the multi-agent / parallel-team bursts don't
# trip free-tier rate limits (429). Tune with LLM_MAX_CONCURRENCY.
LLM_MAX_CONCURRENCY = int(os.environ.get("LLM_MAX_CONCURRENCY", "4") or 4)
_llm_semaphore = asyncio.Semaphore(LLM_MAX_CONCURRENCY)


def _cooldown_for(msg: str) -> float:
    """Pick a cooldown based on the failure: auth/payment = long, rate-limit = short."""
    m = msg.lower()
    if any(s in m for s in ("401", "402", "403", "unauthorized", "payment", "forbidden", "invalid api key")):
        return AUTH_COOLDOWN_SECONDS
    if "429" in m or "rate limit" in m or "too many requests" in m:
        return RATE_LIMIT_COOLDOWN_SECONDS
    return PROVIDER_COOLDOWN_SECONDS


class _RetryableLLMError(Exception):
    """Transient provider failure (429 / timeout) — retry same provider briefly."""


async def _post_chat(url: str, headers: Dict[str, str], model: str,
                     system: str, user: str, max_tokens: int, timeout: float) -> str:
    """One OpenAI-compatible chat call. Raises _RetryableLLMError on 429/5xx/timeout,
    a plain Exception (with the HTTP status in the message) on any other failure;
    returns the message content on success. Throttled by the global LLM semaphore so
    bursts don't trip free-tier rate limits."""
    try:
        async with _llm_semaphore:
            async with httpx.AsyncClient(timeout=timeout) as client:
                r = await client.post(
                    url,
                    headers={"Content-Type": "application/json", **headers},
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": max_tokens,
                    },
                )
            if r.status_code == 429 or r.status_code >= 500:
                raise _RetryableLLMError(f"HTTP {r.status_code}")
            if r.status_code >= 400:
                # auth/payment/other client errors — non-retryable; keep the status code
                # in the message so the failover can pick a long (auth) cooldown.
                raise Exception(f"HTTP {r.status_code} {r.text[:120]}")
            content = (r.json()["choices"][0]["message"]["content"] or "").strip()
            if not content:
                raise Exception("empty response")
            return content
    except (httpx.TimeoutException, httpx.ConnectError) as e:
        raise _RetryableLLMError(str(e))


async def _raw_zen(system: str, user: str, max_tokens: int) -> str:
    return await _post_chat(
        f"{OPENCODE_ZEN_BASE}/chat/completions",
        {"Authorization": f"Bearer {OPENCODE_ZEN_API_KEY}"},
        OPENCODE_ZEN_MODEL, system, user, min(max_tokens, ZEN_MAX_OUT), timeout=120)


async def _raw_groq(system: str, user: str, max_tokens: int) -> str:
    return await _post_chat(
        "https://api.groq.com/openai/v1/chat/completions",
        {"Authorization": f"Bearer {GROQ_API_KEY}"},
        LLAMA_MODEL, system, user, min(max_tokens, GROQ_MAX_OUT), timeout=60)


async def _raw_deepseek_v4(system: str, user: str, max_tokens: int) -> str:
    return await _post_chat(
        f"{DEEPSEEK_V4_BASE}/chat/completions",
        {"Authorization": f"Bearer {DEEPSEEK_V4_API_KEY}",
         "HTTP-Referer": "https://jaxvora.ai", "X-Title": "Jaxvora"},
        DEEPSEEK_V4_MODEL, system, user, min(max_tokens, DEEPSEEK_V4_MAX_OUT), timeout=120)


async def _raw_openrouter(system: str, user: str, max_tokens: int) -> str:
    return await _post_chat(
        f"{OPENROUTER_BASE}/chat/completions",
        {"Authorization": f"Bearer {OPENROUTER_API_KEY}",
         "HTTP-Referer": "https://jaxvora.ai", "X-Title": "Jaxvora"},
        DEEPSEEK_MODEL, system, user, min(max_tokens, OPENROUTER_MAX_OUT), timeout=60)


# Registry: ordered identity + key predicate + single-attempt fn. `.strip()` so a
# whitespace-only key (which causes "Missing Authentication header" 401s) counts as
# not configured and the provider is skipped instead of failing every request.
_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "zen": {"enabled": lambda: bool((OPENCODE_ZEN_API_KEY or "").strip()), "fn": _raw_zen},
    "groq": {"enabled": lambda: bool((GROQ_API_KEY or "").strip()), "fn": _raw_groq},
    "deepseek_v4": {"enabled": lambda: bool((DEEPSEEK_V4_API_KEY or "").strip()), "fn": _raw_deepseek_v4},
    "openrouter": {"enabled": lambda: bool((OPENROUTER_API_KEY or "").strip()), "fn": _raw_openrouter},
}


def _build_provider_chain(prefer: Optional[str] = None) -> List[str]:
    """Ordered list of provider names to try. `LLM_PROVIDER_ORDER` (comma-separated)
    pins the chain — set it to e.g. "zen,groq" to use OpenCode Zen instead of
    OpenRouter. Otherwise Zen leads when primary; `prefer` (an agent's own model) is
    moved to the front; configured-but-not-cooled providers come first."""
    env_order = (os.environ.get("LLM_PROVIDER_ORDER", "") or "").strip()
    if env_order:
        order = [p.strip() for p in env_order.split(",") if p.strip() in _PROVIDERS]
    elif OPENCODE_ZEN_PRIMARY:
        order = ["zen", "groq", "deepseek_v4", "openrouter"]
    else:
        order = ["groq", "deepseek_v4", "openrouter", "zen"]
    if not order:
        order = ["zen", "groq"]
    if prefer in _PROVIDERS and prefer in order:
        order = [prefer] + [p for p in order if p != prefer]
    enabled = [n for n in order if _PROVIDERS[n]["enabled"]()]
    now = time.time()
    hot = [n for n in enabled if _PROVIDER_COOLDOWN.get(n, 0) <= now]
    return hot or enabled  # if everything is cooling down, try them all anyway


def llm_provider_status() -> Dict[str, Any]:
    """Snapshot for /settings/status — which providers are configured/cooling."""
    now = time.time()
    return {
        "primary": "zen" if OPENCODE_ZEN_PRIMARY else "per-agent",
        "chain": _build_provider_chain(),
        "providers": {
            n: {
                "configured": _PROVIDERS[n]["enabled"](),
                "cooldown_s": max(0, round(_PROVIDER_COOLDOWN.get(n, 0) - now, 1)),
            } for n in _PROVIDERS
        },
    }


async def _provider_attempt(name: str, system: str, user: str, max_tokens: int) -> str:
    """Try one provider with brief in-place retry for transient (429/timeout)."""
    fn = _PROVIDERS[name]["fn"]
    delays = [0, 1, 3]
    last: Optional[Exception] = None
    for d in delays:
        if d:
            await asyncio.sleep(d)
        try:
            return await fn(system, user, max_tokens)
        except _RetryableLLMError as e:
            last = e
            continue
    raise last or RuntimeError("retryable attempts exhausted")


async def call_llm_failover(system: str, user: str,
                            max_tokens: int = DEFAULT_MAX_TOKENS,
                            prefer: Optional[str] = None) -> str:
    """Single entrypoint every agent/tool uses. Walks the provider chain and
    returns the first working response, shifting providers on any failure."""
    cache_key = f"llm|{max_tokens}|{system}|{user}"
    cached = await redis_cache.get("llm", cache_key)
    if cached:
        return cached
    chain = _build_provider_chain(prefer)
    if not chain:
        return f"[Mock response — no LLM provider configured] Task: {user[:100]}"
    errors: List[str] = []
    for idx, name in enumerate(chain):
        try:
            result = await _provider_attempt(name, system, user, max_tokens)
            _PROVIDER_COOLDOWN.pop(name, None)
            if idx > 0:
                logger.warning(f"LLM shifted to fallback provider '{name}' after: {'; '.join(errors)}")
            await redis_cache.set("llm", cache_key, result)
            return result
        except Exception as e:
            msg = str(e)
            errors.append(f"{name}: {msg}")
            cool = _cooldown_for(msg)
            _PROVIDER_COOLDOWN[name] = time.time() + cool
            logger.warning(f"LLM provider '{name}' failed ({msg}) — benched {int(cool)}s, shifting to next")
            continue
    logger.error(f"All LLM providers failed: {errors}")
    return f"[All LLM providers failed] {'; '.join(errors)[:400]}"


# ── Backwards-compatible wrappers (all route through the failover chain) ─────
async def call_opencode_zen(system: str, user: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    return await call_llm_failover(system, user, max_tokens, prefer="zen")


async def call_openrouter(system: str, user: str, model: str = DEEPSEEK_MODEL) -> str:
    return await call_llm_failover(system, user, prefer=None if OPENCODE_ZEN_PRIMARY else "openrouter")


async def call_groq(system: str, user: str) -> str:
    return await call_llm_failover(system, user, prefer=None if OPENCODE_ZEN_PRIMARY else "groq")


async def call_deepseek_v4(system: str, user: str) -> str:
    return await call_llm_failover(system, user, prefer=None if OPENCODE_ZEN_PRIMARY else "deepseek_v4")


# The Chief Orchestrator runs many think-steps per request; OpenCode Zen
# (deepseek-v4-flash-free) is high-quality but slow. Run the orchestrator on the
# FASTEST free model — Groq (LPU inference, llama-3.3-70b) — FIRST, regardless of
# OPENCODE_ZEN_PRIMARY, with the rest of the chain (incl. Zen) as fallback. Agents
# still default to Zen as their brain.
ORCHESTRATOR_PROVIDER = os.environ.get("ORCHESTRATOR_PROVIDER", "groq")


async def call_orchestrator_llm(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    return await call_llm_failover(system, user, max_tokens, prefer=ORCHESTRATOR_PROVIDER)


def _resolve_app_resource(env_value: str, *relative_parts: str) -> Optional[Path]:
    app_dir = Path(__file__).resolve().parent
    candidates: List[Path] = []
    if env_value:
        configured = Path(env_value)
        candidates.append(configured if configured.is_absolute() else app_dir / configured)
    relative_path = Path(*relative_parts)
    candidates.extend([app_dir / relative_path, app_dir / "server" / relative_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _plain_text_to_email_html(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return '<p style="margin:0 0 14px 0;">Jaxvora update generated successfully.</p>'
    blocks = re.split(r"\n{2,}", text)
    html_blocks = []
    for block in blocks:
        safe = html.escape(block.strip()).replace("\n", "<br>")
        if safe:
            html_blocks.append(f'<p style="margin:0 0 14px 0;">{safe}</p>')
    return "\n".join(html_blocks)


def _sanitize_email_html(value: str) -> str:
    cleaned = str(value or "")
    cleaned = re.sub(r"<\s*(script|style|iframe|object|embed)[^>]*>.*?</\s*\1\s*>", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\son[a-z]+\s*=\s*(['\"]).*?\1", "", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"\s(href|src)\s*=\s*(['\"])\s*javascript:.*?\2", r' \1="#"', cleaned, flags=re.I | re.S)
    return cleaned or '<p style="margin:0 0 14px 0;">Jaxvora update generated successfully.</p>'


def _html_to_plain_text(value: str) -> str:
    text = re.sub(r"(?i)<br\s*/?>", "\n", str(value or ""))
    text = re.sub(r"(?i)</p\s*>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _gmail_template_html(params: Dict[str, Any]) -> str:
    subject = str(params.get("subject") or "Jaxvora notification")
    body = str(params.get("body") or params.get("text") or "")
    body_html = _sanitize_email_html(body) if params.get("html") else _plain_text_to_email_html(body)
    preheader = str(params.get("preheader") or params.get("summary") or f"{subject} from Jaxvora AI")[:160]
    sender = str(params.get("sender_label") or GMAIL_AUTOMATION_USER or GMAIL_SENDER or "Jaxvora AI")
    template_path = _resolve_app_resource(GMAIL_TEMPLATE_FILE, "templates", "gmail_body_template.html")
    if template_path:
        template = template_path.read_text(encoding="utf-8")
    else:
        template = (
            "<html><body>"
            '<img src="cid:{{logo_cid}}" width="52" height="52" alt="Jaxvora">'
            "<h1>{{subject}}</h1>{{body_html}}"
            "<p>Sent by Jaxvora AI from {{sender}}.</p>"
            "</body></html>"
        )
    replacements = {
        "{{subject}}": html.escape(subject),
        "{{preheader}}": html.escape(preheader),
        "{{body_html}}": body_html,
        "{{logo_cid}}": GMAIL_LOGO_CID,
        "{{sender}}": html.escape(sender),
        "{{year}}": str(datetime.now(timezone.utc).year),
    }
    for token, value in replacements.items():
        template = template.replace(token, value)
    return template


def _gmail_logo_bytes() -> bytes:
    logo_path = _resolve_app_resource(GMAIL_LOGO_FILE, "assets", "jaxvora-gmail-logo.png")
    if not logo_path:
        return b""
    try:
        return logo_path.read_bytes()
    except Exception as exc:
        logger.warning(f"Could not read Gmail logo asset: {exc}")
        return b""


def _gmail_mime_message(params: Dict[str, Any], attachments: Optional[List[MIMEBase]] = None) -> MIMEMultipart:
    to = (params.get("to") or "").strip()
    subject = str(params.get("subject") or "Jaxvora notification")
    if not to:
        raise ValueError("Recipient 'to' is required")

    raw_body = str(params.get("body") or params.get("text") or "")
    body_html = _gmail_template_html(params)
    plain_text = _html_to_plain_text(raw_body) if params.get("html") else raw_body.strip()
    if not plain_text:
        plain_text = _html_to_plain_text(body_html) or subject

    body_part = MIMEMultipart("related")
    alternatives = MIMEMultipart("alternative")
    alternatives.attach(MIMEText(plain_text, "plain", "utf-8"))
    alternatives.attach(MIMEText(body_html, "html", "utf-8"))
    body_part.attach(alternatives)

    logo_bytes = _gmail_logo_bytes()
    if logo_bytes:
        logo = MIMEImage(logo_bytes, _subtype="png", name="jaxvora-gmail-logo.png")
        logo.add_header("Content-ID", f"<{GMAIL_LOGO_CID}>")
        logo.add_header("Content-Disposition", "inline", filename="jaxvora-gmail-logo.png")
        body_part.attach(logo)

    root: MIMEMultipart = body_part
    if attachments:
        root = MIMEMultipart("mixed")
        root.attach(body_part)
        for attachment in attachments:
            root.attach(attachment)

    root["To"] = to
    root["From"] = params.get("from") or GMAIL_AUTOMATION_USER or GMAIL_SENDER or "me"
    root["Subject"] = subject
    if params.get("cc"):
        root["Cc"] = str(params.get("cc"))
    if params.get("bcc"):
        root["Bcc"] = str(params.get("bcc"))
    return root


async def send_gmail(to_email: str, subject: str, body: str, attachment_name: str = "", attachment_data: bytes = b"") -> str:
    """Send a branded Jaxvora email through Gmail API first, then SMTP fallback."""
    if not to_email:
        return "[No recipient email — set NOTIFICATION_EMAIL in Settings]"
    api_configured = gmail_automation_status().get("configured")
    if api_configured:
        api_params = {
            "action": "send",
            "to": to_email,
            "subject": subject,
            "body": body,
            "confirm": True,
        }
        if attachment_data and attachment_name:
            api_params["attachment_name"] = attachment_name
            api_params["attachment_data"] = base64.b64encode(attachment_data).decode("utf-8")
        api_result = await run_gmail_automation(api_params)
        if api_result.get("ok"):
            logger.info(f"✉ Email sent via Gmail API to {to_email}: {subject}")
            return f"Email sent to {to_email} via Gmail API"
        logger.warning(f"Gmail API send failed, falling back to SMTP: {api_result.get('error')}")
    if not GMAIL_SENDER or not GMAIL_APP_PASSWORD:
        return "[Gmail not configured - set Gmail OAuth or GMAIL_SENDER and GMAIL_APP_PASSWORD]"
    try:
        attachments = []
        if attachment_data and attachment_name:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(attachment_data)
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
            attachments.append(part)
        msg = _gmail_mime_message(
            {
                "to": to_email,
                "subject": subject,
                "body": body,
                "from": f"Jaxvora <{GMAIL_SENDER}>",
                "sender_label": GMAIL_SENDER,
            },
            attachments=attachments,
        )
        ctx = ssl_lib.create_default_context()
        smtp_password = GMAIL_APP_PASSWORD.replace(" ", "")
        def _send():
            errors = []
            try:
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=8, context=ctx) as srv:
                    srv.login(GMAIL_SENDER, smtp_password)
                    srv.sendmail(GMAIL_SENDER, to_email, msg.as_string())
                    return
            except Exception as exc:
                errors.append(f"SMTP SSL 465: {type(exc).__name__}: {exc}")
            try:
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=8) as srv:
                    srv.ehlo()
                    srv.starttls(context=ctx)
                    srv.ehlo()
                    srv.login(GMAIL_SENDER, smtp_password)
                    srv.sendmail(GMAIL_SENDER, to_email, msg.as_string())
                    return
            except Exception as exc:
                errors.append(f"SMTP STARTTLS 587: {type(exc).__name__}: {exc}")
            raise RuntimeError("; ".join(errors))
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _send)
        logger.info(f"✉ Email sent to {to_email}: {subject}")
        return f"Email sent to {to_email}"
    except Exception as e:
        logger.error(f"Gmail error: {e}")
        return f"[Gmail error: {e}]"


GMAIL_SAFE_ACTIONS = {
    "status",
    "search",
    "read",
    "list_labels",
    "list_drafts",
}
GMAIL_GUARDED_ACTIONS = {
    "draft",
    "send",
    "send_draft",
    "archive",
    "trash",
    "delete",
    "delete_forever",
    "mark_important",
    "unmark_important",
    "star",
    "unstar",
    "mark_read",
    "mark_unread",
    "create_label",
    "apply_label",
    "remove_label",
    "create_filter",
}
GMAIL_ACTIONS = GMAIL_SAFE_ACTIONS | GMAIL_GUARDED_ACTIONS


def gmail_automation_status() -> Dict[str, Any]:
    missing = []
    if not GMAIL_CLIENT_ID:
        missing.append("GMAIL_CLIENT_ID")
    if not GMAIL_CLIENT_SECRET:
        missing.append("GMAIL_CLIENT_SECRET")
    if not GMAIL_REFRESH_TOKEN:
        missing.append("GMAIL_REFRESH_TOKEN")
    return {
        "configured": not missing,
        "user": GMAIL_AUTOMATION_USER or "me",
        "missing": missing,
        "api_guard_configured": bool(GMAIL_AUTOMATION_API_TOKEN),
        "action_api_ready": not missing and bool(GMAIL_AUTOMATION_API_TOKEN),
        "safe_actions": sorted(GMAIL_SAFE_ACTIONS),
        "guarded_actions": sorted(GMAIL_GUARDED_ACTIONS),
        "required_scopes": GMAIL_SCOPES.split(),
        "policy": "Public Gmail actions require X-Jaxvora-Admin-Token when OAuth is configured. Mailbox mutations also require confirm=true; permanent delete requires confirm_text='DELETE FOREVER'.",
    }


def _gmail_action_authorized(provided_token: Optional[str]) -> bool:
    return bool(
        GMAIL_AUTOMATION_API_TOKEN
        and provided_token
        and hmac.compare_digest(str(provided_token), str(GMAIL_AUTOMATION_API_TOKEN))
    )


def _gmail_decode_b64url(value: str) -> str:
    if not value:
        return ""
    padded = value + ("=" * (-len(value) % 4))
    try:
        return base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _gmail_payload_text(payload: Dict[str, Any]) -> str:
    plain_parts: List[str] = []
    html_parts: List[str] = []

    def walk(part: Dict[str, Any]):
        mime_type = part.get("mimeType", "")
        data = (part.get("body") or {}).get("data", "")
        if data and mime_type == "text/plain":
            plain_parts.append(_gmail_decode_b64url(data))
        elif data and mime_type == "text/html":
            html_parts.append(re.sub(r"<[^>]+>", " ", _gmail_decode_b64url(data)))
        for child in part.get("parts") or []:
            walk(child)

    walk(payload or {})
    text = "\n".join(p.strip() for p in plain_parts if p.strip())
    if not text:
        text = "\n".join(re.sub(r"\s+", " ", p).strip() for p in html_parts if p.strip())
    return text[:12000]


def _gmail_headers(payload: Dict[str, Any]) -> Dict[str, str]:
    headers = {}
    for item in (payload or {}).get("headers") or []:
        name = item.get("name", "").lower()
        if name:
            headers[name] = item.get("value", "")
    return headers


def _gmail_message_summary(message: Dict[str, Any], include_body: bool = False) -> Dict[str, Any]:
    payload = message.get("payload") or {}
    headers = _gmail_headers(payload)
    summary = {
        "id": message.get("id", ""),
        "thread_id": message.get("threadId", ""),
        "label_ids": message.get("labelIds", []),
        "snippet": message.get("snippet", ""),
        "subject": headers.get("subject", ""),
        "from": headers.get("from", ""),
        "to": headers.get("to", ""),
        "date": headers.get("date", ""),
    }
    if include_body:
        summary["body"] = _gmail_payload_text(payload)
    return summary


def _gmail_raw_message(params: Dict[str, Any]) -> str:
    attachments = None
    attachment_data_b64 = params.get("attachment_data")
    attachment_name = params.get("attachment_name")
    if attachment_data_b64 and attachment_name:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(base64.b64decode(attachment_data_b64))
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment_name}"')
        attachments = [part]
    msg = _gmail_mime_message(params, attachments=attachments)
    return base64.urlsafe_b64encode(msg.as_bytes()).decode("utf-8")


def _as_list(value: Any) -> List[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v)]
    return [str(value)]


async def _gmail_access_token() -> str:
    status = gmail_automation_status()
    if not status["configured"]:
        raise RuntimeError("Gmail automation is not configured. Missing: " + ", ".join(status["missing"]))
    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": GMAIL_CLIENT_ID,
                "client_secret": GMAIL_CLIENT_SECRET,
                "refresh_token": GMAIL_REFRESH_TOKEN,
                "grant_type": "refresh_token",
            },
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Google token refresh failed ({response.status_code}): {response.text[:400]}")
    data = response.json()
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Google token refresh did not return an access token")
    return token


async def _gmail_request(method: str, path: str, token: str, **kwargs) -> Dict[str, Any]:
    user = GMAIL_AUTOMATION_USER or "me"
    url = f"https://gmail.googleapis.com/gmail/v1/users/{user}{path}"
    headers = kwargs.pop("headers", {})
    headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.request(method, url, headers=headers, **kwargs)
    if response.status_code == 204:
        return {"ok": True}
    try:
        data = response.json()
    except Exception:
        data = {"raw": response.text[:1000]}
    if response.status_code >= 400:
        raise RuntimeError(f"Gmail API error ({response.status_code}): {data}")
    return data


async def run_gmail_automation(params: Dict[str, Any]) -> Dict[str, Any]:
    action = str(params.get("action", "status")).strip().lower()
    if action not in GMAIL_ACTIONS:
        return {"ok": False, "error": f"Unknown Gmail action '{action}'", "available_actions": sorted(GMAIL_ACTIONS)}
    status = gmail_automation_status()
    if action == "status":
        return {"ok": status["configured"], **status}
    if not status["configured"]:
        return {
            "ok": False,
            "error": "Gmail automation is not configured",
            "missing": status["missing"],
            "user": status["user"],
            "setup_hint": "Set GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET, and GMAIL_REFRESH_TOKEN for the Jaxvora Gmail OAuth client.",
        }
    if action in GMAIL_GUARDED_ACTIONS and not bool(params.get("confirm")):
        return {
            "ok": False,
            "error": f"Gmail action '{action}' requires confirm=true",
            "policy": status["policy"],
        }
    if action == "delete_forever" and params.get("confirm_text") != "DELETE FOREVER":
        return {"ok": False, "error": "Permanent delete requires confirm_text='DELETE FOREVER'"}

    try:
        token = await _gmail_access_token()
        if action == "search":
            max_results = max(1, min(int(params.get("max_results") or params.get("limit") or 10), 25))
            list_params: Dict[str, Any] = {"maxResults": max_results}
            if params.get("query"):
                list_params["q"] = params.get("query")
            labels = _as_list(params.get("label_ids") or params.get("labelIds"))
            if labels:
                list_params["labelIds"] = labels
            data = await _gmail_request("GET", "/messages", token, params=list_params)
            messages = []
            for item in data.get("messages", [])[:max_results]:
                msg = await _gmail_request(
                    "GET",
                    f"/messages/{item['id']}",
                    token,
                    params=[
                        ("format", "metadata"),
                        ("metadataHeaders", "Subject"),
                        ("metadataHeaders", "From"),
                        ("metadataHeaders", "To"),
                        ("metadataHeaders", "Date"),
                    ],
                )
                messages.append(_gmail_message_summary(msg))
            return {"ok": True, "action": action, "query": params.get("query", ""), "messages": messages}

        if action == "read":
            message_id = params.get("message_id") or params.get("id")
            if not message_id:
                return {"ok": False, "error": "message_id is required"}
            msg = await _gmail_request("GET", f"/messages/{message_id}", token, params={"format": "full"})
            return {"ok": True, "action": action, "message": _gmail_message_summary(msg, include_body=True)}

        if action == "list_labels":
            labels = await _gmail_request("GET", "/labels", token)
            return {"ok": True, "action": action, "labels": labels.get("labels", [])}

        if action == "list_drafts":
            drafts = await _gmail_request("GET", "/drafts", token, params={"maxResults": min(int(params.get("max_results") or 10), 25)})
            return {"ok": True, "action": action, "drafts": drafts.get("drafts", [])}

        if action == "draft":
            data = await _gmail_request("POST", "/drafts", token, json={"message": {"raw": _gmail_raw_message(params)}})
            return {"ok": True, "action": action, "draft_id": data.get("id"), "message_id": (data.get("message") or {}).get("id")}

        if action == "send":
            data = await _gmail_request("POST", "/messages/send", token, json={"raw": _gmail_raw_message(params)})
            return {"ok": True, "action": action, "message_id": data.get("id"), "thread_id": data.get("threadId")}

        if action == "send_draft":
            draft_id = params.get("draft_id") or params.get("id")
            if not draft_id:
                return {"ok": False, "error": "draft_id is required"}
            data = await _gmail_request("POST", "/drafts/send", token, json={"id": draft_id})
            return {"ok": True, "action": action, "message_id": data.get("id"), "thread_id": data.get("threadId")}

        message_id = params.get("message_id") or params.get("id")
        if action in {"archive", "trash", "delete", "delete_forever", "mark_important", "unmark_important", "star", "unstar", "mark_read", "mark_unread", "apply_label", "remove_label"} and not message_id:
            return {"ok": False, "error": "message_id is required"}

        if action == "archive":
            data = await _gmail_request("POST", f"/messages/{message_id}/modify", token, json={"removeLabelIds": ["INBOX"]})
            return {"ok": True, "action": action, "message": _gmail_message_summary(data)}

        if action in {"trash", "delete"}:
            data = await _gmail_request("POST", f"/messages/{message_id}/trash", token)
            return {"ok": True, "action": "trash", "message": _gmail_message_summary(data)}

        if action == "delete_forever":
            await _gmail_request("DELETE", f"/messages/{message_id}", token)
            return {"ok": True, "action": action, "message_id": message_id}

        label_ops = {
            "mark_important": (["IMPORTANT"], []),
            "unmark_important": ([], ["IMPORTANT"]),
            "star": (["STARRED"], []),
            "unstar": ([], ["STARRED"]),
            "mark_read": ([], ["UNREAD"]),
            "mark_unread": (["UNREAD"], []),
            "apply_label": (_as_list(params.get("label_ids") or params.get("label_id")), []),
            "remove_label": ([], _as_list(params.get("label_ids") or params.get("label_id"))),
        }
        if action in label_ops:
            add_labels, remove_labels = label_ops[action]
            data = await _gmail_request(
                "POST",
                f"/messages/{message_id}/modify",
                token,
                json={"addLabelIds": add_labels, "removeLabelIds": remove_labels},
            )
            return {"ok": True, "action": action, "message": _gmail_message_summary(data)}

        if action == "create_label":
            name = (params.get("label_name") or params.get("name") or "").strip()
            if not name:
                return {"ok": False, "error": "label_name is required"}
            data = await _gmail_request(
                "POST",
                "/labels",
                token,
                json={
                    "name": name,
                    "labelListVisibility": params.get("label_list_visibility", "labelShow"),
                    "messageListVisibility": params.get("message_list_visibility", "show"),
                },
            )
            return {"ok": True, "action": action, "label": data}

        if action == "create_filter":
            criteria = dict(params.get("criteria") or {})
            for src, dest in (("from_email", "from"), ("to_email", "to"), ("subject", "subject"), ("query", "query")):
                if params.get(src):
                    criteria[dest] = params.get(src)
            filter_action = dict(params.get("filter_action") or params.get("mail_action") or {})
            if params.get("add_label_ids"):
                filter_action["addLabelIds"] = _as_list(params.get("add_label_ids"))
            if params.get("remove_label_ids"):
                filter_action["removeLabelIds"] = _as_list(params.get("remove_label_ids"))
            if not criteria or not filter_action:
                return {"ok": False, "error": "create_filter requires criteria and filter_action/add_label_ids/remove_label_ids"}
            data = await _gmail_request("POST", "/settings/filters", token, json={"criteria": criteria, "action": filter_action})
            return {"ok": True, "action": action, "filter": data}

        return {"ok": False, "error": f"Action '{action}' is declared but not implemented"}
    except Exception as e:
        logger.error(f"Gmail automation error: {e}")
        return {"ok": False, "action": action, "error": str(e)}


# === SECTION 2: Cache Layer (Redis) ==========================================

REDIS_URL: str = os.environ.get("REDIS_URL", "")
CACHE_TTL: int = int(os.environ.get("CACHE_TTL", "3600"))

class RedisCache:
    """Async Redis cache for LLM responses, RAG results, and agent outputs."""

    def __init__(self):
        self._client: Optional[Any] = None
        self._enabled = bool(REDIS_URL)

    async def connect(self):
        if not self._enabled:
            return
        try:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                REDIS_URL, decode_responses=True, socket_timeout=2
            )
            await self._client.ping()
        except Exception:
            self._client = None
            self._enabled = False

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    def _key(self, prefix: str, data: str) -> str:
        import hashlib
        return f"{prefix}:{hashlib.sha256(data.encode()).hexdigest()[:32]}"

    async def get(self, prefix: str, data: str) -> Optional[str]:
        if not self._enabled or not self._client:
            return None
        try:
            return await self._client.get(self._key(prefix, data))
        except Exception:
            return None

    async def set(self, prefix: str, data: str, value: str, ttl: int = CACHE_TTL):
        if not self._enabled or not self._client:
            return
        try:
            await self._client.setex(self._key(prefix, data), ttl, value)
        except Exception:
            pass

    async def delete_prefix(self, prefix: str):
        if not self._enabled or not self._client:
            return
        try:
            cursor = 0
            while True:
                cursor, keys = await self._client.scan(cursor, match=f"{prefix}:*", count=100)
                if keys:
                    await self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:
            pass


redis_cache = RedisCache()


# === SECTION 3: Database Layer ================================================

db_pool: Optional[asyncpg.Pool] = None

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;

CREATE TABLE IF NOT EXISTS projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    repo_url TEXT DEFAULT '',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES projects(id) ON DELETE SET NULL,
    agent_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    input TEXT NOT NULL DEFAULT '',
    output TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID REFERENCES tasks(id) ON DELETE CASCADE,
    level TEXT NOT NULL DEFAULT 'INFO',
    message TEXT NOT NULL,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    action TEXT NOT NULL,
    payload TEXT DEFAULT '',
    approved BOOLEAN DEFAULT NULL,
    decision_reason TEXT DEFAULT '',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
ALTER TABLE audit ADD COLUMN IF NOT EXISTS decision_reason TEXT DEFAULT '';

CREATE TABLE IF NOT EXISTS agent_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT NOT NULL,
    task_summary TEXT NOT NULL,
    outcome TEXT NOT NULL DEFAULT 'success',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    search_vector TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS knowledge_base_fts ON knowledge_base USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS knowledge_base_trgm ON knowledge_base USING GIN(content gin_trgm_ops);

CREATE TABLE IF NOT EXISTS app_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL DEFAULT '',
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_text TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT '',
    metadata JSONB DEFAULT '{}',
    embedding DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS rag_documents_fts ON rag_documents USING GIN(to_tsvector('english', chunk_text));

-- Phase 4: jaxvora.* tables for v1.0 operations
CREATE TABLE IF NOT EXISTS jaxvora_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL DEFAULT 'default',
    task_id TEXT NOT NULL DEFAULT '',
    state JSONB DEFAULT '{}',
    context TEXT DEFAULT '',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jaxvora_subtask_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES jaxvora_sessions(id) ON DELETE CASCADE,
    task_id TEXT NOT NULL,
    agent_name TEXT NOT NULL,
    subtask TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    output TEXT DEFAULT '',
    error TEXT DEFAULT '',
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS jaxvora_operation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    tool_name TEXT DEFAULT '',
    input TEXT DEFAULT '',
    output TEXT DEFAULT '',
    success BOOLEAN DEFAULT TRUE,
    duration_ms INTEGER DEFAULT 0,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS jaxvora_ssh_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES jaxvora_sessions(id) ON DELETE SET NULL,
    command TEXT NOT NULL,
    host TEXT DEFAULT '',
    user TEXT DEFAULT '',
    exit_code INTEGER DEFAULT 0,
    output TEXT DEFAULT '',
    approved BOOLEAN DEFAULT NULL,
    risk_level TEXT DEFAULT 'low',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
"""

async def get_db() -> asyncpg.Pool:
    global db_pool
    if db_pool is None:
        raise RuntimeError("Database pool not initialised")
    return db_pool


async def db_execute(query: str, *args):
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.execute(query, *args)


async def db_fetch(query: str, *args) -> List[asyncpg.Record]:
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.fetch(query, *args)


async def db_fetchrow(query: str, *args) -> Optional[asyncpg.Record]:
    pool = await get_db()
    async with pool.acquire() as conn:
        return await conn.fetchrow(query, *args)


async def log_to_db(level: str, message: str, task_id: Optional[str] = None):
    try:
        await db_execute(
            "INSERT INTO logs (task_id, level, message) VALUES ($1, $2, $3)",
            task_id, level, message
        )
    except Exception:
        pass
    await ws_manager.broadcast_log(level, message)


# === SECTION 3: MCP Tool Registry =============================================

class MCPTool:
    RISK_LEVELS = {"low": 0, "medium": 1, "high": 2, "critical": 3}

    def __init__(self, name: str, description: str, risk_level: str = "low",
                 requires_confirmation: bool = False):
        self.name = name
        self.description = description
        self.risk_level = risk_level if risk_level in self.RISK_LEVELS else "low"
        self.requires_confirmation = requires_confirmation

    async def run(self, params: Dict[str, Any]) -> str:
        raise NotImplementedError

    def permission_check(self, params: Dict[str, Any]) -> Optional[str]:
        """Return None if allowed, or a string reason if denied."""
        return None


class FileSystemTool(MCPTool):
    """Full file capabilities scoped to the agent workspace — like a coding agent."""
    SANDBOX = Path("/root/jaxvora-ai/workspace")

    def __init__(self):
        super().__init__(
            "file_system",
            ("Read/write/edit/append/list/mkdir/delete files in the workspace folder. "
             "params: action (read|write|edit|append|list|mkdir|delete|exists), path, "
             "content (for write/append), old_string + new_string (for edit). "
             "All paths are sandboxed to /root/jaxvora-ai/workspace."),
            risk_level="medium", requires_confirmation=True)
        self.SANDBOX.mkdir(parents=True, exist_ok=True)

    def _resolve(self, raw_path: str) -> Path:
        resolved = (self.SANDBOX / (raw_path or "")).resolve()
        if not str(resolved).startswith(str(self.SANDBOX.resolve())):
            raise ValueError("path escapes workspace sandbox")
        return resolved

    def permission_check(self, params: Dict[str, Any]) -> Optional[str]:
        try:
            self._resolve(params.get("path", ""))
        except ValueError:
            return f"Path traversal blocked: {params.get('path', '')}"
        return None

    async def run(self, params: Dict[str, Any]) -> str:
        try:
            action = params.get("action", "read")
            resolved = self._resolve(params.get("path", ""))
            rel = resolved.relative_to(self.SANDBOX.resolve())
            if action == "read":
                if not resolved.exists():
                    return f"file_system: not found: {rel}"
                return resolved.read_text(encoding="utf-8", errors="ignore")[:20000]
            if action == "write":
                resolved.parent.mkdir(parents=True, exist_ok=True)
                content = params.get("content", "")
                resolved.write_text(content, encoding="utf-8")
                return f"Wrote {len(content)} chars to {rel}"
            if action == "append":
                resolved.parent.mkdir(parents=True, exist_ok=True)
                with open(resolved, "a", encoding="utf-8") as f:
                    f.write(params.get("content", ""))
                return f"Appended to {rel}"
            if action == "edit":
                if not resolved.exists():
                    return f"file_system: not found: {rel}"
                old = params.get("old_string", "")
                new = params.get("new_string", "")
                txt = resolved.read_text(encoding="utf-8", errors="ignore")
                if old and old not in txt:
                    return "file_system edit: old_string not found in file"
                resolved.write_text(txt.replace(old, new) if old else txt + new, encoding="utf-8")
                return f"Edited {rel}"
            if action in ("list", "ls"):
                base = resolved if resolved.is_dir() else resolved.parent
                if not base.exists():
                    return "(empty)"
                items = [("[dir] " if p.is_dir() else "      ") + p.name for p in sorted(base.iterdir())]
                return "\n".join(items) or "(empty)"
            if action == "mkdir":
                resolved.mkdir(parents=True, exist_ok=True)
                return f"Created directory {rel}"
            if action == "exists":
                return "true" if resolved.exists() else "false"
            if action == "delete":
                if resolved.is_dir():
                    import shutil
                    shutil.rmtree(resolved)
                elif resolved.exists():
                    resolved.unlink()
                else:
                    return "file_system: nothing to delete"
                return f"Deleted {rel}"
            return f"file_system: unknown action '{action}'"
        except ValueError as e:
            return f"file_system error: {e}"
        except Exception as e:
            return f"file_system error: {e}"


class TerminalTool(MCPTool):
    """Run shell commands in the agent workspace (git, npm, docker, scp, ...)."""
    WORKSPACE = Path("/root/jaxvora-ai/workspace")
    ALLOWED = {
        # inspect / navigate
        "ls", "cat", "pwd", "find", "grep", "wc", "head", "tail", "tree", "which",
        "file", "stat", "du", "df", "echo", "env", "date", "whoami", "sort", "uniq", "diff",
        # file ops
        "mkdir", "touch", "cp", "mv", "rm", "sed", "awk", "tar", "unzip", "zip", "chmod", "ln",
        # dev toolchains
        "git", "npm", "npx", "node", "yarn", "pnpm", "python3", "python", "pip", "pip3",
        "go", "cargo", "rustc", "make", "gcc", "g++", "java", "mvn", "gradle",
        # ops
        "docker", "docker-compose", "curl", "wget", "scp", "ssh", "kubectl",
    }
    DENY = ["rm -rf /", "rm -rf /*", "rm -rf ~", "mkfs", "dd if=", " :/", "shutdown",
            "reboot", "> /etc", "/etc/passwd", "/etc/shadow", "/root/.ssh", "sudo ",
            "chmod -R 777 /", ":(){", "> /dev/sda"]

    def __init__(self):
        super().__init__(
            "terminal",
            ("Run a shell command inside the workspace folder (/root/jaxvora-ai/workspace). "
             "Supports git, npm/npx/node, python3/pip, go, cargo, make, docker, curl, scp, and "
             "file ops. params: command (string), timeout (seconds, default 120). One command "
             "per call — no pipes/redirects/&&."),
            risk_level="high", requires_confirmation=True)
        self.WORKSPACE.mkdir(parents=True, exist_ok=True)

    async def run(self, params: Dict[str, Any]) -> str:
        cmd = (params.get("command") or "").strip()
        if not cmd:
            return "terminal error: empty command"
        low = cmd.lower()
        for bad in self.DENY:
            if bad in low:
                return f"Command blocked for safety (matched '{bad.strip()}')."
        try:
            parts = shlex.split(cmd)
        except ValueError:
            return "terminal error: could not parse command"
        if not parts:
            return "terminal error: empty command"
        if parts[0] not in self.ALLOWED:
            return f"Command '{parts[0]}' is not allowed. Allowed: {', '.join(sorted(self.ALLOWED))}"
        for p in parts[1:]:
            if ".." in p.split("/"):
                return f"Parent-directory access blocked: {p}"
            if p.startswith("/") and not p.startswith("-") and "://" not in p and not p.startswith(str(self.WORKSPACE)):
                return f"Absolute path outside workspace blocked: {p}"
        try:
            timeout_s = float(params.get("timeout", 120) or 120)
        except (TypeError, ValueError):
            timeout_s = 120.0
        timeout_s = max(1.0, min(timeout_s, 300.0))
        try:
            proc = await asyncio.create_subprocess_exec(
                *parts, cwd=str(self.WORKSPACE),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except Exception:
                    pass
                return f"Command timed out after {int(timeout_s)}s"
            text = (out or b"").decode("utf-8", "ignore")
            text = text[:8000] if text.strip() else f"(exit {proc.returncode}, no output)"
            return f"$ {cmd}\n{text}"
        except FileNotFoundError:
            return f"terminal error: '{parts[0]}' is not installed on the server"
        except Exception as e:
            return f"terminal error: {e}"


class PostgreSQLTool(MCPTool):
    def __init__(self):
        super().__init__("postgresql", "Execute SQL queries against the database",
                         risk_level="high", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        sql = params.get("query", "")
        if not sql.strip().lower().startswith("select"):
            return "Only SELECT queries allowed via MCP tool"
        try:
            rows = await db_fetch(sql)
            return json.dumps([dict(r) for r in rows], default=str)
        except Exception as e:
            return f"postgresql error: {e}"


class BrowserTool(MCPTool):
    def __init__(self):
        super().__init__("browser", "Fetch and parse web pages",
                         risk_level="low")

    async def run(self, params: Dict[str, Any]) -> str:
        url = params.get("url", "")
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.get(url, headers={"User-Agent": "Jaxvora/1.0"})
                text = r.text[:3000]
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
        except Exception as e:
            return f"browser error: {e}"


class SecurityScannerTool(MCPTool):
    PATTERNS = [
        (r'(?i)(password|passwd|pwd)\s*=\s*["\'][^"\']{3,}', "Hardcoded password"),
        (r'(?i)(api_?key|apikey|secret)\s*=\s*["\'][^"\']{8,}', "Hardcoded API key"),
        (r'(?i)(token)\s*=\s*["\'][^"\']{8,}', "Hardcoded token"),
        (r'-----BEGIN (RSA |EC )?PRIVATE KEY-----', "Private key in source"),
        (r'(?i)eval\s*\(', "eval() usage"),
        (r'(?i)exec\s*\(', "exec() usage"),
        (r'(?i)subprocess\.call\([^)]+shell\s*=\s*True', "shell=True in subprocess"),
    ]

    def __init__(self):
        super().__init__("security_scanner", "Static analysis for secrets and vulnerabilities",
                         risk_level="low")

    async def run(self, params: Dict[str, Any]) -> str:
        content = params.get("content", "")
        findings = []
        for pattern, desc in self.PATTERNS:
            if re.search(pattern, content):
                findings.append(f"⚠ {desc}")
        return "\n".join(findings) if findings else "✓ No obvious issues found"


class CodeFormatterTool(MCPTool):
    def __init__(self):
        super().__init__("code_formatter", "Lint and format code suggestions",
                         risk_level="low")

    async def run(self, params: Dict[str, Any]) -> str:
        lang = params.get("language", "python")
        code = params.get("code", "")
        lines = code.split("\n")
        issues = []
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i}: exceeds 120 chars ({len(line)})")
            if "\t" in line and lang == "python":
                issues.append(f"Line {i}: tab indentation (use spaces)")
        return "\n".join(issues) if issues else "✓ Code looks clean"


class EmailNotificationTool(MCPTool):
    def __init__(self):
        super().__init__("email_notify", "Send email notifications for bugs, issues, and alerts to the configured recipient",
                         risk_level="medium", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        to = params.get("to", NOTIFICATION_EMAIL)
        subject = params.get("subject", "Jaxvora Alert")
        body = params.get("body", "")
        return await send_gmail(to, subject, body)


class GmailAutomationTool(MCPTool):
    def __init__(self):
        super().__init__(
            "gmail_automation",
            "Governed Gmail API automation for Jaxvora Gmail: search, read, draft, send, archive, delete, labels, and filters",
            risk_level="critical", requires_confirmation=True,
        )

    async def run(self, params: Dict[str, Any]) -> str:
        result = await run_gmail_automation(params)
        return json.dumps(result, indent=2, default=str)


SSH_BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\brm\s+-rf\s+/\*",
    r"\brm\s+-rf\s+\.",
    r"\bmkfs\b",
    r"\bdd\s+if=",
    r":\s*\(\s*\)\s*\{",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
    r"\bhalt\b",
    r"\binit\s+[06]\b",
    r"\bpasswd\b",
    r"\buserdel\b",
    r"\bdeluser\b",
    r"\bchmod\s+-R\s+777\s+",
    r"\bchown\s+-R\b",
    r"\bufw\s+(disable|reset)\b",
    r"\biptables\b",
    r"\bnft\s+flush\b",
    r"\bcurl\b.*\|\s*(sh|bash)",
    r"\bwget\b.*\|\s*(sh|bash)",
]


def validate_ssh_command(command: str) -> Dict[str, Any]:
    cleaned = str(command or "").strip()
    if not cleaned:
        return {"allowed": False, "reason": "Command is empty."}
    if len(cleaned) > 500:
        return {"allowed": False, "reason": "Command is too long for direct chat execution."}
    lowered = cleaned.lower()
    for pattern in SSH_BLOCKED_COMMAND_PATTERNS:
        if re.search(pattern, lowered):
            return {"allowed": False, "reason": "Blocked by SSH safety policy because it can damage the server or security posture."}
    return {"allowed": True, "reason": "Allowed by SSH safety policy."}


class SSHTool(MCPTool):
    def __init__(self):
        super().__init__("ssh_exec", "Execute commands on a remote server via SSH for 24/7 monitoring and management",
                         risk_level="critical", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        host = params.get("host", SSH_HOST)
        port = int(params.get("port", SSH_PORT))
        user = params.get("user", SSH_USER)
        password = params.get("password", SSH_PASSWORD)
        command = params.get("command", "")
        if not host or not user or not command:
            return "[SSH not configured — provide host, user, and command]"
        policy = validate_ssh_command(command)
        if not policy["allowed"]:
            return f"[SSH command blocked: {policy['reason']}]"
        try:
            import asyncssh
            conn_kwargs: Dict[str, Any] = {
                "host": host, "port": port, "username": user,
            }
            if password:
                conn_kwargs["password"] = password
            elif SSH_KEY_PATH:
                conn_kwargs["client_keys"] = [SSH_KEY_PATH]
            elif SSH_KEY:
                import io
                conn_kwargs["client_keys"] = [asyncssh.import_private_key(SSH_KEY)]
            async with asyncssh.connect(**conn_kwargs) as conn:
                result = await conn.run(command, timeout=30)
                output = result.stdout or result.stderr or "(no output)"
                return output[:3000]
        except ImportError:
            return "[asyncssh not installed]"
        except Exception as e:
            return f"[SSH error: {e}]"


class WebSearchTool(MCPTool):
    def __init__(self):
        super().__init__("web_search", "Search the internet via DuckDuckGo for current information, news, documentation",
                         risk_level="low")

    async def run(self, params: Dict[str, Any]) -> str:
        query = params.get("query", "")
        max_results = min(int(params.get("max_results", 5)), 10)
        if not query:
            return "[Web search tool: no query provided]"
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                r = await client.post(
                    "https://lite.duckduckgo.com/lite/",
                    data={"q": query},
                    headers={"User-Agent": "Mozilla/5.0 (compatible; Jaxvora/1.0)"},
                )
                results = []
                lines = r.text.split("\n")
                title, snippet, url = "", "", ""
                in_res = False
                capturing = False
                for line in lines:
                    if '"result-link"' in line or "'result-link'" in line or '"result__a"' in line or "'result__a'" in line:
                        if title:
                            results.append(f"\u2022 {title}\n  {html.unescape(re.sub(r'<[^>]+>', '', snippet)).strip()}\n  {url}")
                        title = snippet = url = ""
                        capturing = False
                        h = re.search(r'href="([^"]+)"', line)
                        if not h:
                            h = re.search(r"href='([^']+)'", line)
                        if h:
                            url = html.unescape(h.group(1))
                        t = re.search(r'>([^<]+)<', line)
                        if t:
                            title = html.unescape(t.group(1))
                        in_res = True
                    elif in_res and ('class="result-snippet"' in line or "class='result-snippet'" in line or 'class="result__snippet"' in line or "class='result__snippet'" in line):
                        capturing = True
                        after_class = line.split(">", 1)[1] if ">" in line else ""
                        snippet += after_class + " "
                    elif in_res and capturing:
                        if "</td>" in line or "</TD>" in line:
                            capturing = False
                        elif "<tr" not in line and line.strip() and not line.strip().startswith("<"):
                            snippet += line.strip() + " "
                    elif in_res and line.strip() in ("</div>", "</div"):
                        if title:
                            results.append(f"\u2022 {title}\n  {html.unescape(re.sub(r'<[^>]+>', '', snippet)).strip()}\n  {url}")
                        title = snippet = url = ""
                        in_res = False
                        capturing = False
                    if len(results) >= max_results:
                        break
                if title:
                    results.append(f"\u2022 {title}\n  {snippet}\n  {url}")
                if not results:
                    return f"No web results found for '{query}'."
                return f"Web search results for '{query}':\n\n" + "\n\n".join(results[:max_results])
        except Exception as e:
            return f"[Web search error: {e}]"


class AgentInvokeTool(MCPTool):
    def __init__(self):
        super().__init__("agent_invoke", "Invoke a specialist agent by name to perform a sub-task",
                         risk_level="medium", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        name = params.get("name", "")
        task = params.get("task", "")
        if not name or name not in AGENT_REGISTRY:
            return f"[AgentInvokeTool] Agent '{name}' not found. Available: {', '.join(list(AGENT_REGISTRY.keys())[:10])}..."
        agent = AGENT_REGISTRY[name]
        result = await agent.run(task)
        output = result.output[:3000] if result.output else "(no output)"
        return f"[Agent: {name}]\n{output}\n[Success: {result.success}]"


def _role_base_prompt(role_name: str) -> str:
    """Expertise prompt for any agent role. Uses the registered agent's own system
    prompt so parallel workers carry that role's real specialization."""
    agent = AGENT_REGISTRY.get(role_name)
    if agent is not None:
        base = getattr(agent, "_system", None) or getattr(agent, "description", "")
        if base:
            return base
    return f"You are a senior {role_name}. Apply expert, production-quality judgment in your domain."


async def _run_parallel_team(role_name: str, task: str, parts: Any, project: str) -> Dict[str, Any]:
    """Generic map-reduce over a role: split the task into independent slices, run
    N workers of that role IN PARALLEL (fast direct LLM calls with provider
    failover — no slow nested agent loops), then a Head/Lead of that role reviews
    and MERGES every submission into one finalized deliverable. Works for ANY of
    the 37 agent roles (Software Engineer, Data Analyst, QA, Research, ...)."""
    role_name = (role_name or "Software Engineer").strip()
    base = _role_base_prompt(role_name)
    report: Dict[str, Any] = {"tool": "parallel_team", "role": role_name, "task": task[:200]}

    worker_sys = (
        f"{base}\n\nYou are ONE worker in a parallel team of {role_name}s. You own a single, focused "
        f"slice of a larger task. Complete ONLY your slice with concrete, production-ready output — do "
        f"not wait for or assume other workers. For code, put each file in a fenced block with its path "
        f"on the opening fence. State briefly what your slice covers and any interface others depend on. "
        f"Never claim something works that you did not actually produce.")
    head_sys = (
        f"{base}\n\nYou are the Head/Lead {role_name}. Several {role_name}s worked IN PARALLEL on separate "
        f"slices of one task. Review every submission for correctness, conflicts, duplication, and gaps, "
        f"then MERGE them into a single finalized, coherent deliverable with one consistent style and clean "
        f"interfaces between slices. End with a '## Review notes' section: what each worker contributed, the "
        f"conflicts you resolved, and any remaining risks. Be evidence-driven — never claim the merged result "
        f"works unless the pieces actually fit; call out anything unverified.")

    # 1. Decompose into independent, parallelizable slices.
    subtasks: List[str] = []
    if isinstance(parts, list) and parts:
        subtasks = [str(p).strip() for p in parts if str(p).strip()][:6]
    else:
        try:
            n = max(0, min(int(parts), 6)) if parts not in (None, "") else 0
        except (TypeError, ValueError):
            n = 0
        want = f"exactly {n}" if n else "2 to 4"
        ctx = f"\nProject/context: {project}" if project else ""
        raw = await call_llm_failover(
            f"You are a lead {role_name}. Split the task into independent, parallelizable slices — each a "
            f"self-contained piece one {role_name} can own without blocking others. Return ONLY a JSON array "
            f"of short slice descriptions.",
            f"Task: {task}{ctx}\n\nReturn {want} slices as a JSON array of strings.")
        m = re.search(r"\[.*\]", raw or "", re.DOTALL)
        if m:
            try:
                subtasks = [str(x).strip() for x in json.loads(m.group(0)) if str(x).strip()][:6]
            except Exception:
                subtasks = []
        if not subtasks:
            subtasks = [task]  # fall back to a single slice
    report["slices"] = subtasks

    # 2. Workers run concurrently — they do NOT wait on each other.
    sem = asyncio.Semaphore(max(1, MAX_PARALLEL_AGENTS))

    async def _work(i: int, sub: str) -> Dict[str, Any]:
        async with sem:
            try:
                out = await asyncio.wait_for(
                    call_llm_failover(
                        worker_sys,
                        f"Overall task: {task}\n\nYour slice ({i + 1}/{len(subtasks)}): {sub}"),
                    timeout=150)
                ok = not str(out).startswith("[All LLM providers failed")
            except asyncio.TimeoutError:
                out, ok = f"[Worker {i + 1} TIMED OUT after 150s — Verification Incomplete]", False
            except Exception as e:
                out, ok = f"[Worker {i + 1} failed: {e}]", False
        return {"worker": f"{role_name} Worker {i + 1}", "subtask": sub, "ok": ok, "output": out}

    results = await asyncio.gather(*[_work(i, s) for i, s in enumerate(subtasks)])
    report["workers"] = [
        {"worker": r["worker"], "subtask": r["subtask"], "ok": r["ok"], "chars": len(str(r["output"]))}
        for r in results]
    ok_workers = [r for r in results if r["ok"]]
    if not ok_workers:
        report["verdict"] = "FAIL"
        report["reason"] = f"all {role_name} workers failed (see workers[])"
        return report

    # 3. Head/Lead of the role reviews + merges every submission.
    bundle = "\n\n".join(
        f"### {r['worker']} — slice: {r['subtask']}\n{str(r['output'])[:6000]}" for r in results)
    merged = await call_llm_failover(
        head_sys,
        f"Original task: {task}\n\nWorker submissions (parallel):\n\n{bundle}\n\n"
        "Produce the final merged, reviewed deliverable.",
        max_tokens=MAX_TOKENS)
    merge_ok = not str(merged).startswith("[All LLM providers failed")
    report["workers_succeeded"] = len(ok_workers)
    report["workers_total"] = len(results)
    report["verdict"] = "PASS" if merge_ok else "FAIL"
    report["final"] = merged
    return report


class ParallelTeamTool(MCPTool):
    """Run a big task with multiple workers of ANY agent role IN PARALLEL, then a
    Head/Lead of that role reviews and merges their work into one finalized result."""

    def __init__(self):
        super().__init__("parallel_team",
            "Run a BIG / multi-part task with multiple workers of ANY role IN PARALLEL (fast, no slow nested "
            "agent loops), then a Head/Lead of that role reviews and MERGES all their work into one finalized "
            "deliverable. Works for any agent role (Software Engineer, Data Analyst, QA/Test Agent, Research, "
            "Cybersecurity, Documentation, ...). Use instead of dispatching one agent for large/multi-part work. "
            "params: role (agent/role name, default 'Software Engineer'); task (required); parts (optional: "
            "integer worker count 2-6, OR a list of subtask strings); project (optional context).",
            risk_level="medium")

    async def run(self, params: Dict[str, Any]) -> str:
        task = (params.get("task") or "").strip()
        if not task:
            return json.dumps({"ok": False, "error": "task is required", "verdict": "FAIL"})
        role = (params.get("role") or params.get("agent") or "Software Engineer").strip()
        project = (params.get("project") or "").strip()
        report = await _run_parallel_team(role, task, params.get("parts"), project)
        return json.dumps(report, indent=2)


class ParallelEngineeringTool(MCPTool):
    """Back-compat alias: parallel_engineering == parallel_team with role=Software Engineer."""

    def __init__(self):
        super().__init__("parallel_engineering",
            "Run a BIG / multi-part coding task with multiple Software Engineer workers IN PARALLEL, then a "
            "Head of Software Engineering reviews and merges all their work into one finalized result. "
            "(Alias of parallel_team with role='Software Engineer'.) "
            "params: task (required); parts (optional integer 2-6 or list of subtasks); project (optional).",
            risk_level="medium")

    async def run(self, params: Dict[str, Any]) -> str:
        task = (params.get("task") or "").strip()
        if not task:
            return json.dumps({"ok": False, "error": "task is required", "verdict": "FAIL"})
        report = await _run_parallel_team("Software Engineer", task, params.get("parts"),
                                          (params.get("project") or "").strip())
        return json.dumps(report, indent=2)


WORKSPACE_DIR = Path("/root/jaxvora-ai/workspace")


# === Social media connectors & autonomous posting ===========================
SOCIAL_PLATFORMS = ["x", "facebook", "instagram", "whatsapp", "reddit", "linkedin"]
SOCIAL_LABELS = {"x": "X (Twitter)", "facebook": "Facebook", "instagram": "Instagram",
                 "whatsapp": "WhatsApp", "reddit": "Reddit", "linkedin": "LinkedIn"}
_SOCIAL_CACHE: Dict[str, Dict[str, Any]] = {}


async def social_load() -> Dict[str, Dict[str, Any]]:
    global _SOCIAL_CACHE
    if db_pool is not None:
        try:
            row = await db_fetchrow("SELECT value FROM app_settings WHERE key='social_connectors'")
            if row and row["value"]:
                _SOCIAL_CACHE = json.loads(row["value"])
        except Exception as e:
            logger.warning(f"social_load failed: {e}")
    return _SOCIAL_CACHE


async def social_save(data: Dict[str, Dict[str, Any]]):
    global _SOCIAL_CACHE
    _SOCIAL_CACHE = data
    if db_pool is not None:
        try:
            await db_execute(
                "INSERT INTO app_settings (key, value, updated_at) VALUES ('social_connectors', $1, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET value=$1, updated_at=NOW()",
                json.dumps(data),
            )
        except Exception as e:
            logger.warning(f"social_save failed: {e}")


def social_public_view(data: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in SOCIAL_PLATFORMS:
        c = data.get(p, {}) or {}
        tok = c.get("token") or ""
        out.append({
            "platform": p, "label": SOCIAL_LABELS[p],
            "connected": bool(tok),
            "auto_post": bool(c.get("auto_post")),
            "token_hint": (tok[:4] + "…" + tok[-4:]) if len(tok) > 8 else ("set" if tok else ""),
            "meta": dict(c.get("meta") or {}),
        })
    return out


async def social_publish(platform: str, conn: Dict[str, Any], text: str, link: str = "", image_url: str = "") -> Dict[str, Any]:
    """Best-effort token-based publish. Each platform needs its API token (and a
    few platform-specific meta fields) obtained from that platform's dev portal."""
    token = (conn.get("token") or "").strip()
    meta = conn.get("meta") or {}
    if not token:
        return {"ok": False, "error": f"{SOCIAL_LABELS.get(platform, platform)} is not connected."}
    body = text + ((" " + link) if link else "")
    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            if platform == "x":
                r = await client.post("https://api.twitter.com/2/tweets",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"text": body[:280]})
            elif platform == "linkedin":
                author = meta.get("author") or meta.get("urn")
                if not author:
                    return {"ok": False, "error": "LinkedIn needs meta.author URN (e.g. urn:li:person:XXXX)."}
                r = await client.post("https://api.linkedin.com/v2/ugcPosts",
                    headers={"Authorization": f"Bearer {token}", "X-Restli-Protocol-Version": "2.0.0", "Content-Type": "application/json"},
                    json={"author": author, "lifecycleState": "PUBLISHED",
                          "specificContent": {"com.linkedin.ugc.ShareContent": {"shareCommentary": {"text": body}, "shareMediaCategory": "NONE"}},
                          "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"}})
            elif platform == "facebook":
                page = meta.get("page_id")
                if not page:
                    return {"ok": False, "error": "Facebook needs meta.page_id."}
                r = await client.post(f"https://graph.facebook.com/{page}/feed",
                    params={"message": body, "access_token": token})
            elif platform == "instagram":
                return {"ok": False, "error": "Instagram requires an image + 2-step container publish; provide meta.ig_user_id and image_url to enable."}
            elif platform == "reddit":
                sub = meta.get("subreddit")
                if not sub:
                    return {"ok": False, "error": "Reddit needs meta.subreddit."}
                r = await client.post("https://oauth.reddit.com/api/submit",
                    headers={"Authorization": f"Bearer {token}", "User-Agent": "Jaxvora/1.0"},
                    data={"sr": sub, "kind": "self", "title": (text[:280] or "Update"), "text": body})
            elif platform == "whatsapp":
                phone_id, to = meta.get("phone_id"), meta.get("to")
                if not (phone_id and to):
                    return {"ok": False, "error": "WhatsApp needs meta.phone_id and meta.to."}
                r = await client.post(f"https://graph.facebook.com/v19.0/{phone_id}/messages",
                    headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                    json={"messaging_product": "whatsapp", "to": to, "type": "text", "text": {"body": body}})
            else:
                return {"ok": False, "error": f"Unknown platform {platform}"}
        return {"ok": r.status_code < 300, "status": r.status_code, "response": r.text[:300]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


class SocialMediaTool(MCPTool):
    def __init__(self):
        super().__init__(
            "social_post",
            ("Post to a connected social platform. params: platform (x|facebook|instagram|"
             "whatsapp|reddit|linkedin), text, link (optional), force (optional). If that "
             "platform's auto_post is OFF, returns a DRAFT for approval instead of publishing."),
            risk_level="critical", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        platform = (params.get("platform") or "").lower().strip()
        text = params.get("text") or ""
        if platform not in SOCIAL_PLATFORMS:
            return f"social_post error: unknown platform '{platform}'. Options: {', '.join(SOCIAL_PLATFORMS)}"
        if not text:
            return "social_post error: empty text"
        data = await social_load()
        conn = data.get(platform, {}) or {}
        if not conn.get("token"):
            return f"[{SOCIAL_LABELS[platform]}] not connected — connect it on the Connectors page first."
        force = str(params.get("force", "")).lower() in ("1", "true", "yes")
        if not conn.get("auto_post") and not force:
            return f"[DRAFT · {SOCIAL_LABELS[platform]}] auto-post is OFF. Draft for your approval:\n\n{text}"
        result = await social_publish(platform, conn, text, params.get("link", ""), params.get("image_url", ""))
        if result.get("ok"):
            return f"[Posted to {SOCIAL_LABELS[platform]}] (HTTP {result.get('status')})"
        return f"[{SOCIAL_LABELS[platform]} post failed] {result.get('error') or result.get('response')}"


class CodeRunnerTool(MCPTool):
    SANDBOX = Path("/root/jaxvora-ai/workspace")

    def __init__(self):
        super().__init__("code_runner",
            "Run Python, Node.js, or shell code in a sandboxed workspace. "
            "params: language (python|node|shell), code (string of code to run), timeout_seconds (optional, max 120). "
            "Returns stdout, stderr, and exit code. Use this to test snippets, evaluate logic, or execute scripts.",
            risk_level="medium", requires_confirmation=True)

    def permission_check(self, params: Dict[str, Any]) -> Optional[str]:
        code = params.get("code", "")
        blocked = ["rm -rf", "rm -rf /", "mkfs", "dd if=", "> /dev/", "chmod 777 /",
                   "import os; os.remove", "import shutil; shutil.rmtree",
                   "__import__('os').system", "subprocess.call"]
        for pattern in blocked:
            if pattern in code:
                return f"Blocked dangerous pattern: {pattern}"
        return None

    async def run(self, params: Dict[str, Any]) -> str:
        lang = (params.get("language") or "python").lower().strip()
        code = params.get("code", "")
        timeout_s = min(int(params.get("timeout_seconds", 30)), 120)

        self.SANDBOX.mkdir(parents=True, exist_ok=True)

        if lang == "python":
            cmd = ["python3", "-c", code]
        elif lang == "node":
            cmd = ["node", "-e", code]
        elif lang == "shell":
            cmd = ["bash", "-c", code]
        else:
            return f"code_runner error: unsupported language '{lang}'. Use python, node, or shell."

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(self.SANDBOX),
                ),
                timeout=timeout_s,
            )
            stdout, stderr = await proc.communicate()
            out = stdout.decode("utf-8", errors="replace").strip()
            err = stderr.decode("utf-8", errors="replace").strip()
            parts = []
            if out:
                parts.append(f"[stdout]\n{out}")
            if err:
                parts.append(f"[stderr]\n{err}")
            parts.append(f"[exit code] {proc.returncode}")
            return "\n\n".join(parts)
        except asyncio.TimeoutError:
            return f"code_runner error: timed out after {timeout_s}s"
        except Exception as e:
            return f"code_runner error: {e}"


class PlaywrightTool(MCPTool):
    def __init__(self):
        super().__init__("playwright",
            "Browser automation via Playwright. "
            "params: action (navigate|screenshot|text|click|fill|evaluate), "
            "url (for navigate), selector (for click/fill), value (for fill), "
            "script (for evaluate), timeout_seconds (optional, max 60). "
            "Use this to interact with web pages, take screenshots, extract content, or fill forms.",
            risk_level="medium", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        action = (params.get("action") or "").lower().strip()
        url = params.get("url", "")
        selector = params.get("selector", "")
        value = params.get("value", "")
        script = params.get("script", "")
        timeout_s = min(int(params.get("timeout_seconds", 30)), 60)

        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=["--no-sandbox", "--disable-gpu"])
                page = await browser.new_page()
                await page.set_viewport_size({"width": 1280, "height": 720})

                if action == "navigate":
                    if not url:
                        return "playwright error: url required for navigate"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
                    title = await page.title()
                    text = await page.inner_text("body")
                    text = re.sub(r'\s+', ' ', text).strip()[:5000]
                    await browser.close()
                    return f"[{title}]\n{text}"

                elif action == "screenshot":
                    if not url:
                        return "playwright error: url required for screenshot"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
                    await page.wait_for_timeout(1000)
                    screenshot_path = str(self.SANDBOX / f"screenshot_{int(time.time())}.png")
                    await page.screenshot(path=screenshot_path, full_page=True)
                    await browser.close()
                    return f"Screenshot saved to {screenshot_path}"

                elif action == "text":
                    if not url:
                        return "playwright error: url required for text"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded")
                    text = await page.inner_text("body")
                    text = re.sub(r'\s+', ' ', text).strip()[:5000]
                    await browser.close()
                    return text

                elif action == "click":
                    if not selector:
                        return "playwright error: selector required for click"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded") if url else None
                    await page.click(selector, timeout=timeout_s * 1000)
                    await page.wait_for_timeout(500)
                    text = await page.inner_text("body")
                    text = re.sub(r'\s+', ' ', text).strip()[:3000]
                    await browser.close()
                    return f"Clicked {selector}\n{text}"

                elif action == "fill":
                    if not selector or not value:
                        return "playwright error: selector and value required for fill"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded") if url else None
                    await page.fill(selector, value, timeout=timeout_s * 1000)
                    await page.wait_for_timeout(500)
                    await browser.close()
                    return f"Filled {selector} with '{value}'"

                elif action == "evaluate":
                    if not script:
                        return "playwright error: script required for evaluate"
                    await page.goto(url, timeout=timeout_s * 1000, wait_until="domcontentloaded") if url else None
                    result = await page.evaluate(script)
                    await browser.close()
                    return str(result)[:5000]

                else:
                    await browser.close()
                    return f"playwright error: unknown action '{action}'. Use navigate, screenshot, text, click, fill, or evaluate."

        except Exception as e:
            return f"playwright error: {e}"

    SANDBOX = Path("/root/jaxvora-ai/workspace")


def _http_health_check(port: int, timeout_s: float = 3) -> Optional[Dict]:
    """Return dict with http_code, body (truncated) if port responds, else None."""
    try:
        r = subprocess.run(
            ["curl", "-sS", "-o", "-", "-w", "\n%{http_code}", f"http://127.0.0.1:{port}/"],
            capture_output=True, text=True, timeout=timeout_s,
        )
        parts = r.stdout.strip().rsplit("\n", 1)
        if len(parts) == 2:
            body, code = parts
            return {"http_code": code, "body": body[:500]}
        return None
    except Exception:
        return None


def _verify_proxy_url(url_path: str, timeout_s: float = 5) -> Dict:
    """Check if the jaxvora proxy URL responds with 200 (with retries)."""
    full = f"http://127.0.0.1:8090{url_path}"
    for attempt in range(3):
        try:
            r = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", full],
                capture_output=True, text=True, timeout=timeout_s,
            )
            code = r.stdout.strip()
            if code in ("200", "301", "302"):
                return {"status": "PASS", "proxy_url": full, "http_code": code}
            if attempt < 2:
                import time; time.sleep(1.5)
        except Exception:
            if attempt < 2:
                import time; time.sleep(1.5)
    return {"status": "FAIL", "proxy_url": full, "http_code": code if 'code' in dir() else '?'}


class FrontendPreviewTool(MCPTool):
    def __init__(self):
        super().__init__("frontend_preview",
            "Serve a directory of static files as a live web preview. "
            "params: directory (path under workspace, e.g. 'my-app/build'), name (preview name), port (optional, auto). "
            "Starts an HTTP server in the background and returns a structured verification report.",
            risk_level="low")

    async def run(self, params: Dict[str, Any]) -> str:
        directory = (params.get("directory") or "").strip()
        name = (params.get("name") or directory.replace("/", "-")).strip()
        base = Path("/root/jaxvora-ai/workspace")
        serve_dir = base / directory
        report: Dict[str, Any] = {"tool": "frontend_preview", "params": params, "checks": []}

        def add_check(label: str, status: str, detail: str):
            report["checks"].append({"check": label, "status": status, "detail": detail})

        # 1. Directory existence
        if not serve_dir.is_dir():
            add_check("Directory exists", "FAIL", f"'{serve_dir}' does not exist")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)
        files = list(serve_dir.iterdir())
        add_check("Directory exists", "PASS", f"{serve_dir} ({len(files)} entries)")

        # 2. Find free port
        port = int(params.get("port", 0))
        if port < 8080 or port > 8099:
            for i in range(8080, 8100):
                if i not in [info["port"] for info in APP_REGISTRY.values()]:
                    port = i
                    break
        add_check("Port selected", "PASS" if port else "FAIL", str(port))

        # 3. Start screen session and capture startup logs
        screen_name = f"preview_{name}"
        cmd = f"cd {serve_dir} && python3 -m http.server {port} --directory {serve_dir}"
        start_logs = ""
        try:
            r = subprocess.run(["screen", "-dmS", screen_name, "bash", "-c", cmd],
                               capture_output=True, timeout=10)
            start_logs += f"screen exit code: {r.returncode}\nstdout: {r.stdout.decode(errors='replace')[:300]}\n"
            start_logs += f"stderr: {r.stderr.decode(errors='replace')[:300]}"
        except Exception as e:
            add_check("Screen session start", "FAIL", f"exception: {e}")
            report["startup_logs"] = start_logs
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        add_check("Screen session start", "PASS" if r.returncode == 0 else "FAIL",
                  f"exit_code={r.returncode}")
        report["startup_logs"] = start_logs
        await asyncio.sleep(2)

        # 4. Verify process alive (screen session exists)
        screen_check = subprocess.run(["screen", "-ls", screen_name],
                                      capture_output=True, text=True, timeout=5)
        pid = None
        if screen_check.returncode == 0 and screen_name in screen_check.stdout:
            pid_match = re.search(r'(\d+)\.' + re.escape(screen_name), screen_check.stdout)
            pid = pid_match.group(1) if pid_match else "unknown"
            add_check("Process alive", "PASS", f"screen PID {pid}")
        else:
            # Try pgrep as fallback
            pg = subprocess.run(["pgrep", "-f", screen_name], capture_output=True, text=True, timeout=5)
            if pg.stdout.strip():
                pid = pg.stdout.strip().split("\n")[0]
                add_check("Process alive (pgrep)", "PASS", f"PID {pid}")
            else:
                add_check("Process alive", "FAIL",
                          f"screen session '{screen_name}' not found")
                report["verdict"] = "FAIL"
                return json.dumps(report, indent=2)
        report["pid"] = pid

        # 5. Verify port is listening
        ss_check = subprocess.run(
            ["ss", "-tlnp"], capture_output=True, text=True, timeout=5
        )
        port_listening = f":{port}" in ss_check.stdout
        add_check("Port listening", "PASS" if port_listening else "FAIL",
                  f"port {port} {'found' if port_listening else 'not found in ss output'}")

        if not port_listening:
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        # 6. HTTP health check
        health = _http_health_check(port, timeout_s=5)
        if health:
            http_ok = health["http_code"] in ("200", "301", "302", "308")
            add_check("HTTP health check", "PASS" if http_ok else "FAIL",
                      f"HTTP {health['http_code']}, body: {health['body'][:200]}")
        else:
            add_check("HTTP health check", "FAIL", "no response from port")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        # 7. Register in app registry
        info = {
            "name": name, "port": port, "directory": str(serve_dir),
            "status": "running", "url": f"/apps/{name}/",
            "type": "static_preview", "pid": pid,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        APP_REGISTRY[name] = info
        add_check("App registry", "PASS", f"registered as '/apps/{name}/'")

        # 8. Proxy URL verification — real check. Run in a worker thread so the
        # event loop stays free to serve this same server's /apps/ route (a
        # synchronous self-curl on the loop would deadlock).
        proxy = await asyncio.to_thread(_verify_proxy_url, f"/apps/{name}/")
        add_check("Proxy URL verification", proxy["status"],
                  f"GET {proxy['proxy_url']} -> HTTP {proxy.get('http_code', '?')}")
        if proxy["status"] != "PASS":
            report["verdict"] = "FAIL"
            report["reason"] = (
                f"proxy route /apps/{name}/ did not respond "
                f"(HTTP {proxy.get('http_code', '?')})"
            )
            return json.dumps(report, indent=2)

        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        report["verdict"] = "PASS" if not fails else "FAIL"
        report["preview_url"] = f"https://jaxvora.vercel.app/apps/{name}/"
        return json.dumps(report, indent=2)


class ServerRunnerTool(MCPTool):
    def __init__(self):
        super().__init__("server_runner",
            "Run any command as a background server and register it in the app proxy. "
            "params: name (required), command (required), directory (optional), port (optional). "
            "Returns a structured verification report with PASS/FAIL per step.",
            risk_level="medium", requires_confirmation=True)

    async def run(self, params: Dict[str, Any]) -> str:
        name = (params.get("name") or "").strip()
        command = (params.get("command") or "").strip()
        report: Dict[str, Any] = {"tool": "server_runner", "params": params, "checks": []}

        def add_check(label: str, status: str, detail: str):
            report["checks"].append({"check": label, "status": status, "detail": detail})

        if not name or not command:
            add_check("Parameters", "FAIL", "name and command are required")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        directory = (params.get("directory") or "").strip()
        base = Path("/root/jaxvora-ai/workspace")
        cwd = str(base / directory) if directory else str(base)

        if not Path(cwd).is_dir():
            add_check("Directory exists", "FAIL", f"'{cwd}' does not exist")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)
        add_check("Directory exists", "PASS", cwd)

        # 1. Find free port
        port_hint = int(params.get("port", 0))
        port = port_hint if 8080 <= port_hint <= 8099 else 8080
        for i in range(8080, 8100):
            if i not in [info["port"] for info in APP_REGISTRY.values()]:
                port = i
                break
        add_check("Port selected", "PASS", str(port))

        # 2. Start screen session
        screen_name = f"srv_{name}"
        full_cmd = f"cd {cwd} && {command}"
        start_logs = ""
        try:
            r = subprocess.run(["screen", "-dmS", screen_name, "bash", "-c", full_cmd],
                               capture_output=True, timeout=10)
            start_logs += f"screen exit code: {r.returncode}\n"
            start_logs += f"stdout: {r.stdout.decode(errors='replace')[:300]}\n"
            start_logs += f"stderr: {r.stderr.decode(errors='replace')[:300]}"
        except Exception as e:
            add_check("Screen session start", "FAIL", f"exception: {e}")
            report["startup_logs"] = start_logs
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        add_check("Screen session start", "PASS" if r.returncode == 0 else "FAIL",
                  f"exit_code={r.returncode}")
        report["startup_logs"] = start_logs
        await asyncio.sleep(2)

        # 3. Verify process alive
        screen_check = subprocess.run(["screen", "-ls", screen_name],
                                      capture_output=True, text=True, timeout=5)
        pid = None
        if screen_check.returncode == 0 and screen_name in screen_check.stdout:
            pid_match = re.search(r'(\d+)\.' + re.escape(screen_name), screen_check.stdout)
            pid = pid_match.group(1) if pid_match else "unknown"
            add_check("Process alive", "PASS", f"screen PID {pid}")
        else:
            pg = subprocess.run(["pgrep", "-f", screen_name], capture_output=True, text=True, timeout=5)
            if pg.stdout.strip():
                pid = pg.stdout.strip().split("\n")[0]
                add_check("Process alive (pgrep)", "PASS", f"PID {pid}")
            else:
                add_check("Process alive", "FAIL",
                          f"screen session '{screen_name}' not found")
                report["verdict"] = "FAIL"
                return json.dumps(report, indent=2)
        report["pid"] = pid

        # 4. Detect actual listening port
        actual_port = port
        ports_found = []
        for try_port in range(8080, 8100):
            chk = subprocess.run(
                ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{try_port}/"],
                capture_output=True, text=True, timeout=3,
            )
            code = chk.stdout.strip()
            if code in ("200", "301", "302", "308"):
                ports_found.append(try_port)
                if actual_port == port:
                    actual_port = try_port
                    break

        if not ports_found:
            add_check("Port detection", "FAIL",
                      "no listening port found in range 8080-8099")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        # Also verify via ss
        ss_check = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
        port_confirmed = f":{actual_port}" in ss_check.stdout
        add_check("Port listening", "PASS" if port_confirmed else "WARN",
                  f"port {actual_port} detected (HTTP{' ' if port_confirmed else ' but ss disagrees'})")
        report["detected_ports"] = ports_found

        # 5. HTTP health check on detected port
        health = _http_health_check(actual_port, timeout_s=5)
        if health:
            http_ok = health["http_code"] in ("200", "301", "302", "308")
            add_check("HTTP health check", "PASS" if http_ok else "FAIL",
                      f"HTTP {health['http_code']}, body: {health['body'][:200]}")
        else:
            add_check("HTTP health check", "FAIL", "no response from port")
            report["verdict"] = "FAIL"
            return json.dumps(report, indent=2)

        # 6. Register in app registry
        info = {
            "name": name, "port": actual_port, "directory": cwd,
            "status": "running", "url": f"/apps/{name}/",
            "type": "server", "pid": pid,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        APP_REGISTRY[name] = info
        add_check("App registry", "PASS", f"registered as '/apps/{name}/'")

        # 7. Proxy URL verification — real check via a worker thread so the event
        # loop stays free to serve this server's own /apps/ route (no deadlock).
        proxy = await asyncio.to_thread(_verify_proxy_url, f"/apps/{name}/")
        add_check("Proxy URL verification", proxy["status"],
                  f"GET {proxy['proxy_url']} -> HTTP {proxy.get('http_code', '?')}")
        if proxy["status"] != "PASS":
            report["verdict"] = "FAIL"
            report["reason"] = (
                f"proxy route /apps/{name}/ did not respond "
                f"(HTTP {proxy.get('http_code', '?')})"
            )
            return json.dumps(report, indent=2)

        fails = [c for c in report["checks"] if c["status"] == "FAIL"]
        report["verdict"] = "PASS" if not fails else "FAIL"
        report["preview_url"] = f"https://jaxvora.vercel.app/apps/{name}/"

        return json.dumps(report, indent=2)


class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    async def run(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Tool '{name}' not found"
        tool = self._tools[name]
        denial = tool.permission_check(params)
        if denial:
            return f"[Permission denied] {denial}"
        return await tool.run(params)

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description, "risk_level": t.risk_level,
                  "requires_confirmation": t.requires_confirmation} for t in self._tools.values()]

    def get_risk_level(self, name: str) -> str:
        tool = self._tools.get(name)
        return tool.risk_level if tool else "unknown"

    def requires_confirmation(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.requires_confirmation if tool else False


tool_registry = MCPToolRegistry()


class ErrorEscalation:
    """Error escalation chain: agent → division lead → Chief Orchestrator → human."""

    LEVELS = ["agent", "division_lead", "orchestrator", "human"]

    def __init__(self):
        self._escalations: List[Dict] = []

    async def escalate(self, agent_name: str, error: str, context: str = "",
                       level: str = "agent") -> Dict:
        entry = {
            "agent": agent_name,
            "error": error[:500],
            "context": context[:1000],
            "level": level,
            "resolved": False,
            "resolution": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._escalations.append(entry)
        # Log to DB
        await log_to_db("ERROR",
            f"[Escalation:{level}] {agent_name}: {error[:200]}")
        # Auto-escalate through chain if not resolved
        if level == "agent":
            return await self._try_division_lead(entry)
        return entry

    async def _try_division_lead(self, entry: Dict) -> Dict:
        lead_name = DIVISION_LEADS.get(
            next((a.division for a in AGENT_REGISTRY.values()
                  if a.name == entry["agent"]), ""), "")
        if lead_name and lead_name in AGENT_REGISTRY:
            lead = AGENT_REGISTRY[lead_name]
            result = await lead.run(
                f"Resolve error from {entry['agent']}: {entry['error']}\nContext: {entry['context']}")
            entry["lead_response"] = result.output if hasattr(result, 'output') else str(result)[:500]
            entry["level"] = "division_lead"
            if "resolved" in (result.output if hasattr(result, 'output') else "").lower():
                entry["resolved"] = True
                entry["resolution"] = "Resolved by division lead"
                return entry
        entry["level"] = "orchestrator"
        entry["resolved"] = False
        entry["resolution"] = "Needs human intervention"
        return entry

    def recent_escalations(self, limit: int = 10) -> List[Dict]:
        return self._escalations[-limit:]

    def unresolved_count(self) -> int:
        return sum(1 for e in self._escalations if not e["resolved"])


error_escalation = ErrorEscalation()


def _doctor_check(name: str, ok: bool, detail: str, output: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "detail": detail,
        "output": str(output or "")[:2000],
    }


async def _doctor_subprocess_check(name: str, command: List[str], timeout_seconds: int = 25) -> Dict[str, Any]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(Path(__file__).resolve().parent),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
        output = (stdout or b"").decode("utf-8", errors="ignore")
        err = (stderr or b"").decode("utf-8", errors="ignore")
        combined = (output + ("\n" + err if err else "")).strip()
        return _doctor_check(name, proc.returncode == 0, f"exit={proc.returncode}", combined)
    except asyncio.TimeoutError:
        return _doctor_check(name, False, f"Timed out after {timeout_seconds}s")
    except Exception as exc:
        return _doctor_check(name, False, f"{type(exc).__name__}: {exc}")


async def _doctor_http_check(name: str, path: str, method: str = "GET", expected_status: int = 200) -> Dict[str, Any]:
    url = f"http://127.0.0.1:{PORT}{path}"
    try:
        async with httpx.AsyncClient(timeout=12) as client:
            response = await client.request(method, url)
        ok = response.status_code == expected_status
        content_type = response.headers.get("content-type", "")
        content_length = response.headers.get("content-length", str(len(response.content)))
        detail = f"{method} {path} -> {response.status_code}"
        if content_type:
            detail += f" ({content_type}, {content_length} bytes)"
        output = "" if not content_type.startswith(("application/json", "text/")) else response.text[:1200]
        return _doctor_check(name, ok, detail, output)
    except Exception as exc:
        return _doctor_check(name, False, f"{method} {path} failed: {type(exc).__name__}: {exc}")


async def _run_jaxvora_doctor_iteration(iteration: int) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    app_dir = Path(__file__).resolve().parent

    py_targets = [str(app_dir / "main.py")]
    server_main = app_dir / "server" / "main.py"
    if server_main.exists():
        py_targets.append(str(server_main))
    checks.append(await _doctor_subprocess_check("Python compile", [sys.executable, "-m", "py_compile", *py_targets]))

    checks.append(_doctor_check("Database pool", db_pool is not None, "PostgreSQL pool is available" if db_pool else "PostgreSQL pool is not available"))
    checks.append(_doctor_check("Agent registry", len(AGENT_REGISTRY) >= 20, f"{len(AGENT_REGISTRY)} agents registered"))
    checks.append(_doctor_check("Tool registry", len(tool_registry.list_tools()) >= 8, f"{len(tool_registry.list_tools())} MCP tools registered"))

    checks.append(await _doctor_http_check("Settings API", "/settings/status"))
    checks.append(await _doctor_http_check("Agents API", "/agents"))
    checks.append(await _doctor_http_check("Analytics API", "/analytics"))
    checks.append(await _doctor_http_check("Gmail status API", "/gmail/status"))
    checks.append(await _doctor_http_check("Favicon asset", "/favicon.png"))

    gmail_status = gmail_automation_status()
    checks.append(_doctor_check(
        "Gmail automation",
        gmail_status.get("configured") and gmail_status.get("action_api_ready"),
        f"user={gmail_status.get('user')} missing={','.join(gmail_status.get('missing') or []) or 'none'}",
    ))

    index_path = app_dir / "index.html"
    if not index_path.exists():
        index_path = app_dir / "server" / "index.html"
    index_text = index_path.read_text(encoding="utf-8", errors="ignore") if index_path.exists() else ""
    checks.append(_doctor_check(
        "Chat renderer",
        "renderRichText" in index_text and "md-content" in index_text,
        "Structured chat renderer is installed" if "renderRichText" in index_text else "Structured chat renderer is missing",
    ))

    if SSH_HOST and SSH_USER:
        ssh_output = await SSHTool().run({"command": "printf 'SSH_OK\\n'; hostname; pwd; uptime"})
        checks.append(_doctor_check("SSH tool", "SSH_OK" in ssh_output, f"{SSH_USER}@{SSH_HOST}:{SSH_PORT}", ssh_output))
    else:
        checks.append(_doctor_check("SSH tool", False, "SSH_HOST or SSH_USER is missing"))

    failed = [check for check in checks if not check["ok"]]
    return {"iteration": iteration, "ok": not failed, "failed": failed, "checks": checks}


def _format_doctor_report(iterations: List[Dict[str, Any]]) -> str:
    final = iterations[-1] if iterations else {"ok": False, "checks": [], "failed": []}
    status = "stable" if final.get("ok") else "issues found"
    lines = [
        f"## Jaxvora Doctor: {status}",
        "",
        f"**Iterations:** {len(iterations)}",
        f"**Final result:** {'All checks passed' if final.get('ok') else str(len(final.get('failed') or [])) + ' check(s) failed'}",
        "",
        "### Checks",
    ]
    for check in final.get("checks", []):
        marker = "PASS" if check.get("ok") else "FAIL"
        lines.append(f"- **{marker}** `{check.get('name')}` - {check.get('detail')}")

    if final.get("failed"):
        lines.extend(["", "### Required fixes"])
        for check in final["failed"]:
            lines.append(f"- `{check['name']}`: {check['detail']}")
    else:
        lines.extend([
            "",
            "### Result",
            "- Runtime APIs are responding.",
            "- Gmail automation is connected.",
            "- SSH tool is connected and executing safe diagnostics.",
            "- Chat markdown renderer is installed.",
        ])

    ssh_check = next((check for check in final.get("checks", []) if check.get("name") == "SSH tool"), None)
    if ssh_check and ssh_check.get("output"):
        lines.extend(["", "### SSH diagnostic output", "```text", ssh_check["output"].strip(), "```"])
    return "\n".join(lines)


async def run_jaxvora_doctor(max_iterations: int = 2) -> Dict[str, Any]:
    iterations: List[Dict[str, Any]] = []
    max_iterations = max(1, min(int(max_iterations or 1), 3))
    for iteration in range(1, max_iterations + 1):
        result = await _run_jaxvora_doctor_iteration(iteration)
        iterations.append(result)
        if result["ok"]:
            break
        await asyncio.sleep(0.6)

    final = iterations[-1]
    report = _format_doctor_report(iterations)
    try:
        await log_to_db("INFO" if final["ok"] else "WARN", f"Jaxvora Doctor run: {'stable' if final['ok'] else 'issues found'}")
    except Exception:
        pass
    return {"ok": final["ok"], "iterations": iterations, "report": report, "failed": final.get("failed", [])}


# === SECTION 4: Memory Manager ================================================

class MemoryManager:
    COLLECTIONS = ["architecture_knowledge", "security_findings", "code_fixes", "org_knowledge"]

    async def store(self, collection: str, content: str, metadata: Dict = None):
        try:
            await db_execute(
                "INSERT INTO knowledge_base (collection, content, metadata) VALUES ($1, $2, $3)",
                collection, content, json.dumps(metadata) if metadata else None
            )
        except Exception as e:
            logger.warning(f"Memory store error: {e}")

    async def search(self, query: str, collection: Optional[str] = None, limit: int = 5) -> List[Dict]:
        try:
            if collection:
                rows = await db_fetch(
                    """SELECT id, collection, content, metadata, created_at,
                              ts_rank(search_vector, plainto_tsquery('english', $1)) AS rank
                       FROM knowledge_base
                       WHERE collection = $2
                         AND search_vector @@ plainto_tsquery('english', $1)
                       ORDER BY rank DESC LIMIT $3""",
                    query, collection, limit
                )
            else:
                rows = await db_fetch(
                    """SELECT id, collection, content, metadata, created_at,
                              ts_rank(search_vector, plainto_tsquery('english', $1)) AS rank
                       FROM knowledge_base
                       WHERE search_vector @@ plainto_tsquery('english', $1)
                       ORDER BY rank DESC LIMIT $2""",
                    query, limit
                )
            return [
                {"id": str(r["id"]), "collection": r["collection"],
                 "content": r["content"], "score": float(r["rank"])}
                for r in rows
            ]
        except Exception as e:
            logger.warning(f"Memory search error: {e}")
            return []


memory = MemoryManager()


# === SECTION 4b: RAG Engine ===================================================

class RAGEngine:
    """Retrieval-Augmented Generation engine with hybrid search (FTS + vector)."""

    def __init__(self):
        self._embed_model = None
        self._index: Dict[str, List[float]] = {}
        self._index_loaded = False
        self.dim = 384

    def _lazy_model(self):
        if self._embed_model is None:
            from fastembed import TextEmbedding
            self._embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

    def _cosine_sim(self, a: List[float], b: List[float]) -> float:
        import math
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-10)

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                last_newline = text.rfind("\n", start, end)
                if last_newline > start + chunk_size // 2:
                    end = last_newline
                else:
                    last_space = text.rfind(" ", start, end)
                    if last_space > start + chunk_size // 2:
                        end = last_space
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap if end < len(text) else len(text)
        return chunks or [text.strip()]

    async def rebuild_index(self):
        """Rebuild in-memory vector index from all DB documents."""
        self._index = {}
        try:
            rows = await db_fetch("SELECT id, embedding FROM rag_documents WHERE embedding IS NOT NULL")
            for r in rows:
                self._index[str(r["id"])] = r["embedding"]
            self._index_loaded = True
            logger.info(f"RAG index rebuilt: {len(self._index)} vectors")
        except Exception as e:
            logger.warning(f"RAG index rebuild failed: {e}")

    async def ingest(self, text: str, source: str = "", metadata: Optional[Dict] = None) -> int:
        """Chunk text, embed chunks, store in DB and index."""
        self._lazy_model()
        chunks = self._chunk_text(text)
        if not chunks:
            return 0
        embeddings = list(self._embed_model.embed(chunks))
        count = 0
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            emb_list = [float(v) for v in emb]
            row = await db_fetchrow(
                """INSERT INTO rag_documents (chunk_text, source, metadata, embedding)
                   VALUES ($1, $2, $3, $4::double precision[]) RETURNING id""",
                chunk, source, json.dumps(metadata or {}), emb_list
            )
            if row:
                self._index[str(row["id"])] = emb_list
                count += 1
        logger.info(f"RAG ingested {count} chunks from '{source}'")
        return count

    async def search(self, query: str, top_k: int = 5, fts_only: bool = False) -> List[Dict]:
        """Hybrid search: FTS candidates → vector rerank. Falls back to FTS-only."""
        cached = await redis_cache.get("rag", f"search|{query}|{top_k}|{fts_only}")
        if cached:
            return json.loads(cached)
        if not self._index_loaded:
            await self.rebuild_index()
        if not self._index_loaded:
            return []

        self._lazy_model()
        query_emb = list(self._embed_model.embed([query]))[0]
        query_emb_list = [float(v) for v in query_emb]

        try:
            fts_rows = await db_fetch(
                """SELECT id, chunk_text, source, metadata, created_at,
                          ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', $1)) AS rank
                   FROM rag_documents
                   WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', $1)
                   ORDER BY rank DESC LIMIT $2""",
                query, top_k * 3
            )
            candidates = [(str(r["id"]), r) for r in fts_rows]
        except Exception:
            candidates = []

        if not candidates and self._index:
            doc_ids = list(self._index.keys())
            rows = await db_fetch(
                "SELECT id, chunk_text, source, metadata, created_at FROM rag_documents WHERE id = ANY($1::uuid[]) LIMIT $2",
                [uuid.UUID(did) for did in doc_ids[:top_k * 3]], top_k * 3
            )
            candidates = []
            for r in rows:
                did = str(r["id"])
                emb = self._index.get(did)
                if emb:
                    sim = self._cosine_sim(query_emb_list, emb)
                    candidates.append((did, r, sim))
                else:
                    candidates.append((did, r, 0.0))

        scored = []
        seen_ids = set()
        for item in candidates:
            if len(item) == 2:
                did, row = item
                emb = self._index.get(did)
                sim = self._cosine_sim(query_emb_list, emb) if emb else 0.0
                fts_rank = float(row.get("rank", 0) or 0)
            else:
                did, row, sim = item
                fts_rank = float(row.get("rank", 0) or 0)
            if did in seen_ids:
                continue
            seen_ids.add(did)
            hybrid_score = sim * 0.7 + min(fts_rank, 1.0) * 0.3
            scored.append((hybrid_score, {
                "id": did,
                "content": row["chunk_text"][:2000],
                "source": row.get("source", ""),
                "score": round(hybrid_score, 4),
                "created_at": str(row.get("created_at", "")),
            }))

        scored.sort(key=lambda x: -x[0])
        results = [s[1] for s in scored[:top_k]]
        await redis_cache.set("rag", f"search|{query}|{top_k}|{fts_only}", json.dumps(results, default=str), ttl=600)
        return results

    async def augment(self, query: str, top_k: int = 3) -> str:
        """Search RAG and return formatted context string for LLM injection."""
        results = await self.search(query, top_k=top_k)
        if not results:
            return ""
        parts = ["## Retrieved Context (RAG)"]
        for i, r in enumerate(results, 1):
            source = f" [{r['source']}]" if r.get("source") else ""
            parts.append(f"{i}.{source} {r['content']}")
        return "\n\n".join(parts)


rag_engine = RAGEngine()


# === SECTION 5: Agent Registry & Base Agent Class =============================

class AgentResult:
    def __init__(self, agent: str, task: str, output: str, success: bool = True, task_id: str = None):
        self.agent = agent
        self.task = task
        self.output = output
        self.success = success
        self.task_id = task_id or str(uuid.uuid4())

    def to_dict(self):
        return {
            "agent": self.agent, "task": self.task,
            "output": self.output, "success": self.success,
            "task_id": self.task_id
        }


class BaseAgent:
    name: str = "base"
    model: str = "deepseek"
    division: str = "general"
    description: str = ""
    _status: str = "idle"
    _current_task: str = ""
    force_prefer: Optional[str] = None  # pin this agent to a provider (e.g. Chief -> groq)

    async def call_llm(self, system: str, user: str) -> str:
        # Every agent runs through the multi-provider failover chain. An agent can
        # pin a provider via force_prefer (used to keep the Chief on fast Groq);
        # otherwise Zen leads when primary, else the agent's own model. Any failure
        # shifts to the next provider automatically.
        if self.force_prefer:
            return await call_llm_failover(system, user, prefer=self.force_prefer)
        prefer = {"groq": "groq", "deepseek_v4": "deepseek_v4"}.get(self.model, "openrouter")
        return await call_llm_failover(system, user, prefer=None if OPENCODE_ZEN_PRIMARY else prefer)

    async def run(self, task: str, project_id: Optional[str] = None) -> AgentResult:
        self._status = "running"
        self._current_task = task
        task_id = str(uuid.uuid4())
        try:
            await db_execute(
                "INSERT INTO tasks (id, project_id, agent_name, status, input) VALUES ($1, $2, $3, 'running', $4)",
                task_id, project_id, self.name, task
            )
            await log_to_db("INFO", f"[{self.name}] Starting: {task[:80]}", task_id)
            await ws_manager.broadcast_agent_status(self.name, "running", task[:60])
            output = await self._execute(task)
        except asyncio.CancelledError:
            # Dispatch timed out / was cancelled. CancelledError is NOT an Exception
            # subclass, so it would otherwise escape and leave _status stuck at
            # "running" forever — keeping a ghost node in the Agent Flow graph. Settle
            # the status, record it, and re-raise to honor the cancellation.
            self._status = "idle"
            self._current_task = ""
            try:
                await db_execute(
                    "UPDATE tasks SET status='cancelled', completed_at=NOW() WHERE id=$1", task_id)
            except Exception:
                pass
            raise
        except Exception as e:
            tb = traceback.format_exc()
            try:
                await db_execute(
                    "UPDATE tasks SET status='failed', output=$1, completed_at=NOW() WHERE id=$2",
                    str(e), task_id
                )
                await db_execute(
                    "INSERT INTO agent_history (agent_name, task_summary, outcome) VALUES ($1, $2, 'failed')",
                    self.name, task[:120]
                )
                await log_to_db("ERROR", f"[{self.name}] Error: {e}", task_id)
            except Exception:
                pass
            # Settle to idle (not a sticky "error") so the agent does not linger in the
            # flow graph; the failure is still carried in the returned AgentResult.
            self._status = "idle"
            self._current_task = ""
            try:
                await ws_manager.broadcast_agent_status(self.name, "error", str(e)[:60])
            except Exception:
                pass
            return AgentResult(self.name, task, f"Error: {e}\n{tb}", False, task_id)

        try:
            await db_execute(
                "UPDATE tasks SET status='completed', output=$1, completed_at=NOW() WHERE id=$2",
                output, task_id
            )
            await db_execute(
                "INSERT INTO agent_history (agent_name, task_summary, outcome) VALUES ($1, $2, 'success')",
                self.name, task[:120]
            )
            await log_to_db("INFO", f"[{self.name}] Completed ✓", task_id)
        except Exception as e:
            logger.warning(f"[{self.name}] Post-execution bookkeeping failed: {e}")
        self._status = "idle"
        self._current_task = ""
        await ws_manager.broadcast_agent_status(self.name, "idle", "")
        return AgentResult(self.name, task, output, True, task_id)

    async def _execute(self, task: str) -> str:
        raise NotImplementedError

    def status_dict(self):
        return {
            "name": self.name, "model": self.model,
            "division": self.division, "description": self.description,
            "status": self._status, "current_task": self._current_task,
            "division_lead": DIVISION_LEADS.get(self.division) == self.name,
            "collaborators": AGENT_NETWORK.get(self.name, []),
        }


# === SECTION 5.5: Agent Graph Framework (TAOR v1.0) ============================

class AgentGraphState:
    """State for the THINK→ACT→OBSERVE→REFLECT loop (v1.0 protocol)."""
    def __init__(self, task: str, system_prompt: str, max_iterations: int = 8,
                 cancel_flag: Optional[Callable[[], bool]] = None):
        self.task = task
        self._cancel_flag = cancel_flag
        self.system_prompt = system_prompt
        self.messages: List[Dict[str, str]] = []
        self.tool_results: List[Dict[str, Any]] = []
        self.iteration = 0
        self.max_iterations = max_iterations
        self.final_output: Optional[str] = None
        self.conversation_log: List[str] = []

        self.task_id = str(uuid.uuid4())
        self.raw_intent = ""
        self.subtasks: Dict[str, Dict] = {}
        self.parallel_groups: List[List[str]] = []
        self.rag_required = False
        self.rag_query = ""
        self.estimated_hops = 0
        self.risk_flags: List[str] = []
        self.observe_results: Dict[str, Dict] = {}
        self.confidence_score = 0.0
        self.goal_fulfilled = False
        self.handoff_queue: List[Dict] = []
        self.pending_confirmation: Optional[Dict] = None
        self.agent_latencies: Dict[str, float] = {}
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.last_reflect: Optional[Dict] = None
        self.steps: List[Dict] = []

    def add_step(self, type_: str, agent: str, description: str, detail: str = "", status: str = "done"):
        self.steps.append({
            "type": type_,
            "agent": agent,
            "description": description,
            "detail": str(detail)[:800],
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self.conversation_log.append(f"[{role.upper()}] {content[:200]}")

    def to_prompt(self) -> str:
        lines = [f"## Task\n{self.task}\n"]
        lines.append(f"**Task ID:** {self.task_id}")
        if self.raw_intent:
            lines.append(f"**Intent:** {self.raw_intent}")
        if self.tool_results:
            lines.append("\n## Previous Tool & Agent Results")
            for tr in self.tool_results[-5:]:
                lines.append(f"\n- **{tr.get('tool', 'unknown')}**")
                lines.append(f"  Result: {str(tr.get('result', ''))[:400]}")
                if tr.get("error"):
                    lines.append(f"  Error: {tr['error']}")
        if self.observe_results:
            lines.append("\n## Observation Summary")
            for sid, res in list(self.observe_results.items())[-3:]:
                lines.append(f"- Subtask {sid}: {res.get('status', '?')} (conf: {res.get('confidence', 'N/A')})")
        lines.append(f"\n## Iteration {self.iteration + 1} of {self.max_iterations}")
        for m in self.messages[-4:]:
            lines.append(f"\n[{m['role'].upper()}] {m['content'][:200]}")
        return "\n".join(lines)


class AgentWorkflow:
    """THINK → ACT → OBSERVE → REFLECT loop — v1.0 structured protocol."""

    SYSTEM_TEMPLATE = """You are Jaxvora's Chief Orchestrator following the TAOR v1.0 protocol.

## THINK Phase
Before any action, output a structured <think> block:

<think>
  <task_id>{task_id}</task_id>
  <raw_intent>what the user actually wants</raw_intent>
  <decomposed_subtasks>
    <subtask id="1" depends_on="none">description</subtask>
    <subtask id="2" depends_on="1">description</subtask>
  </decomposed_subtasks>
  <division_routing>
    <route subtask_id="1" division="Engineering" agent="Software Engineer" reason="why this agent"/>
  </division_routing>
  <parallel_groups>[[1], [2]]</parallel_groups>
  <rag_required>true|false</rag_required>
  <rag_query>query if rag_required</rag_query>
  <estimated_hops>n</estimated_hops>
  <risk_flags>list irreversible actions or NONE</risk_flags>
</think>

## ACT Phase
Dispatch to agents with <dispatch> or call MCP tools with <mcp_call>:

<dispatch agent="Agent Name" subtask_id="1" priority="high">
  <context>relevant context from RAG or prior output</context>
  <instruction>precise scoped instruction</instruction>
  <output_schema>code|json|text|file_path</output_schema>
  <timeout_seconds>120</timeout_seconds>
</dispatch>

<mcp_call tool="tool_name" subtask_id="2">
  <parameter name="param1">value1</parameter>
</mcp_call>

## REFLECT Phase
After results, assess completion:

<reflect>
  <goal_fulfilled>true|false</goal_fulfilled>
  <confidence_score>0.0-1.0</confidence_score>
  <quality_issues>
    <issue subtask_id="1">description of gap</issue>
  </quality_issues>
  <next_action>
    <!-- continue_loop, request_human_input, or finalize -->
    <continue_loop reason="why another iteration is needed"/>
  </next_action>
  <loop_iteration>{{n}}</loop_iteration>
</reflect>

When done, output:
<final_answer>
Your complete synthesized answer
</final_answer>

Available tools:
{tools}

Available specialist agents (via agent_invoke):
{agents}

Rules:
- Never skip THINK
- Flag risk_flags for irreversible actions
- Continue loop if confidence_score < 0.75
- Force-finalize at iteration 8"""

    @staticmethod
    def _build_tool_descriptions() -> str:
        lines = []
        for t in tool_registry.list_tools():
            lines.append(f"- **{t['name']}**: {t['description']}")
        return "\n".join(lines) if lines else "(no tools configured)"

    @staticmethod
    def _build_agent_list() -> str:
        by_div: Dict[str, List[str]] = defaultdict(list)
        for name, ag in AGENT_REGISTRY.items():
            by_div[ag.division].append(name)
        lines = []
        for div, agents in by_div.items():
            lines.append(f"- **{div}**: {', '.join(agents)}")
        return "\n".join(lines)

    @staticmethod
    def _parse_xml_block(text: str, tag: str) -> Optional[str]:
        m = re.search(f'<{tag}[^>]*>(.*?)</{tag}>', text, re.DOTALL)
        return m.group(1).strip() if m else None

    @staticmethod
    def _parse_think_block(text: str) -> Dict[str, Any]:
        think = AgentWorkflow._parse_xml_block(text, "think")
        if not think:
            return {}
        result: Dict[str, Any] = {}
        for field in ["task_id", "raw_intent", "parallel_groups", "rag_required",
                       "rag_query", "estimated_hops", "risk_flags"]:
            m = re.search(f'<{field}>(.*?)</{field}>', think, re.DOTALL)
            if m:
                val = m.group(1).strip()
                if field == "parallel_groups":
                    try:
                        val = json.loads(val)
                    except Exception:
                        val = []
                elif field == "rag_required":
                    val = val.lower() == "true"
                elif field == "estimated_hops":
                    try:
                        val = int(val)
                    except Exception:
                        val = 0
                result[field] = val
        result["subtasks"] = [
            {"id": m.group(1), "depends_on": m.group(2) or "none", "description": m.group(3).strip()}
            for m in re.finditer(r'<subtask\s+id="([^"]*)"(?:\s+depends_on="([^"]*)")?\s*>(.*?)</subtask>', think, re.DOTALL)
        ]
        result["routing"] = [
            {"subtask_id": m.group(1), "division": m.group(2), "agent": m.group(3), "reason": m.group(4).strip()}
            for m in re.finditer(r'<route\s+subtask_id="([^"]*)"\s+division="([^"]*)"\s+agent="([^"]*)"[^>]*>(.*?)</route>', think, re.DOTALL)
        ]
        return result

    @staticmethod
    def _parse_dispatch_blocks(text: str) -> List[Dict]:
        dispatches = []
        for d in re.finditer(r'<dispatch\s+agent="([^"]*)"\s+subtask_id="([^"]*)"(?:\s+priority="([^"]*)")?\s*>(.*?)</dispatch>', text, re.DOTALL):
            block = d.group(4)
            dp: Dict[str, Any] = {"type": "dispatch", "agent": d.group(1), "subtask_id": d.group(2), "priority": d.group(3) or "normal"}
            for field in ["context", "instruction", "output_schema", "timeout_seconds"]:
                m = re.search(f'<{field}>(.*?)</{field}>', block, re.DOTALL)
                if m:
                    val: Any = m.group(1).strip()
                    if field == "timeout_seconds":
                        try:
                            val = int(val)
                        except Exception:
                            val = 120
                    dp[field] = val
            dispatches.append(dp)
        return dispatches

    @staticmethod
    def _parse_mcp_call_blocks(text: str) -> List[Dict]:
        calls = []
        for m in re.finditer(r'<mcp_call\s+tool="([^"]*)"(?:\s+subtask_id="([^"]*)")?\s*>(.*?)</mcp_call>', text, re.DOTALL):
            block = m.group(3)
            params = {}
            for p in re.finditer(r'<parameter\s+name="([^"]+)">(.*?)</parameter>', block, re.DOTALL):
                params[p.group(1)] = p.group(2).strip()
            calls.append({"type": "mcp_call", "tool": m.group(1), "subtask_id": m.group(2) or "", "params": params})
        return calls

    @staticmethod
    def _parse_reflect_block(text: str) -> Optional[Dict]:
        block = AgentWorkflow._parse_xml_block(text, "reflect")
        if not block:
            return None
        result: Dict[str, Any] = {}
        for field in ["goal_fulfilled", "confidence_score", "loop_iteration"]:
            m = re.search(f'<{field}>(.*?)</{field}>', block, re.DOTALL)
            if m:
                val: Any = m.group(1).strip()
                if field == "goal_fulfilled":
                    val = val.lower() == "true"
                elif field == "confidence_score":
                    try:
                        val = float(val)
                    except Exception:
                        val = 0.0
                elif field == "loop_iteration":
                    try:
                        val = int(val)
                    except Exception:
                        val = 0
                result[field] = val
        na = re.search(r'<next_action>\s*(.*?)\s*</next_action>', block, re.DOTALL)
        if na:
            inner = na.group(1)
            for action in ["continue_loop", "request_human_input", "finalize"]:
                m = re.search(f'<{action}[^>]*>(.*?)</{action}>', inner, re.DOTALL)
                if m:
                    result["next_action"] = action
                    if action == "continue_loop":
                        r = re.search(r'reason="([^"]*)"', m.group(0))
                        result["reason"] = r.group(1) if r else ""
                    elif action == "request_human_input":
                        q = re.search(r'question="([^"]*)"', m.group(0))
                        result["question"] = q.group(1) if q else ""
                    elif action == "finalize":
                        s = re.search(r'synthesis="([^"]*)"', m.group(0))
                        result["synthesis"] = s.group(1) if s else ""
        result["quality_issues"] = [
            {"subtask_id": qi.group(1), "description": qi.group(2).strip()}
            for qi in re.finditer(r'<issue\s+subtask_id="([^"]*)">(.*?)</issue>', block, re.DOTALL)
        ]
        return result

    @staticmethod
    async def run(agent: BaseAgent, task: str, max_iterations: int = 8,
                  state: Optional[AgentGraphState] = None,
                  pending_states: Optional[Dict[str, 'AgentGraphState']] = None) -> str:
        system = agent._system_prompt() if hasattr(agent, '_system_prompt') else agent.description
        tools_desc = AgentWorkflow._build_tool_descriptions()
        agents_desc = AgentWorkflow._build_agent_list()
        if state is None:
            state = AgentGraphState(task, system, max_iterations)
            state.add_message("user", task)

        for iteration in range(state.iteration, max_iterations):
            state.iteration = iteration
            if getattr(state, '_cancel_flag', None) and state._cancel_flag():
                state.final_output = "The task was cancelled by the user."
                state.add_step("final", agent.name, "Cancelled by user", state.final_output[:500])
                return state.final_output

            # ── THINK ──
            think_prompt = (
                f"{system}\n\n"
                f"{AgentWorkflow.SYSTEM_TEMPLATE.format(task_id=state.task_id, tools=tools_desc, agents=agents_desc)}\n\n"
                f"History so far:\n{state.to_prompt()}\n\n"
                "Begin with your <think> block, then proceed to ACT (<dispatch> or <mcp_call>) or <reflect> with <finalize>."
            )
            response = await agent.call_llm(system, think_prompt)
            state.add_message("assistant", response)

            # ── Fail fast: if every LLM provider is down/rate-limited, abort the
            # loop immediately instead of grinding through all iterations (which
            # turned a provider outage into a ~130s hang). Surface a clean message.
            if response.startswith("[All LLM providers failed"):
                state.add_step("error", agent.name, "LLM providers unavailable", response[:300], "error")
                state.final_output = (
                    "⚠️ All LLM providers are currently unavailable (rate-limited or out of "
                    "credits), so I couldn't complete this request. Details: " + response[:300]
                )
                return state.final_output

            # ── Check for final answer ──
            final_match = re.search(r'<final_answer>(.*?)</final_answer>', response, re.DOTALL)
            if final_match:
                state.final_output = final_match.group(1).strip()
                state.add_step("final", agent.name, "Final answer ready", state.final_output[:500])
                return state.final_output

            # ── Parse THINK for risk_flags ──
            if AgentWorkflow._parse_xml_block(response, "think"):
                think_data = AgentWorkflow._parse_think_block(response)
                state.raw_intent = think_data.get("raw_intent", state.raw_intent)
                state.rag_required = think_data.get("rag_required", state.rag_required)
                state.rag_query = think_data.get("rag_query", state.rag_query)
                state.estimated_hops = think_data.get("estimated_hops", state.estimated_hops)
                risk_flags_raw = think_data.get("risk_flags", "")
                state.add_step("think", agent.name, f"Thinking about subtask", think_data.get("raw_intent", ""))
                if risk_flags_raw and risk_flags_raw.strip().upper() != "NONE" and agent.name != "Chief Orchestrator":
                    state.pending_confirmation = {"risk_flags": risk_flags_raw, "task_id": state.task_id}
                    if pending_states is not None:
                        pending_states[state.task_id] = state
                    return f"__CONFIRMATION__:{state.task_id}:{risk_flags_raw}"

            # ── Parse REFLECT ──
            reflect = AgentWorkflow._parse_reflect_block(response)
            if reflect:
                state.last_reflect = reflect
                state.goal_fulfilled = reflect.get("goal_fulfilled", False)
                state.confidence_score = reflect.get("confidence_score", 0.0)
                state.add_step("reflect", agent.name, f"Reflected (conf:{reflect.get('confidence_score', 0.0):.2f})", str(reflect))
                next_action = reflect.get("next_action")
                if next_action == "finalize":
                    if reflect.get("synthesis"):
                        final = await agent.call_llm(system,
                            f"Task: {task}\n\nSynthesis: {reflect['synthesis']}\n\n"
                            f"Conversation:\n{state.to_prompt()}\n\nProvide the final answer.")
                        state.final_output = final
                        state.add_step("final", agent.name, "Final answer from synthesis", final[:500])
                        return final
                    if state.messages:
                        content = state.messages[-1].get("content", "")
                        state.final_output = content[:4000]
                        state.add_step("final", agent.name, "Final answer from last message", state.final_output[:500])
                        return state.final_output
                elif next_action == "request_human_input":
                    return f"__HUMAN_INPUT__:{state.task_id}:{reflect.get('question', 'Need your input to continue.')}"
                # continue_loop → fall through to ACT

            # ── ACT: parse dispatches and MCP calls ──
            dispatches = AgentWorkflow._parse_dispatch_blocks(response)
            mcp_calls = AgentWorkflow._parse_mcp_call_blocks(response)

            if not dispatches and not mcp_calls:
                if not reflect or reflect.get("next_action") != "continue_loop":
                    if iteration >= max_iterations - 1:
                        if state.messages:
                            content = state.messages[-1].get("content", "")
                            state.final_output = content[:4000]
                            return state.final_output
                        return "I was unable to complete this task."
                    state.add_message("system", "Continue. Output a <think> block followed by <dispatch> or <mcp_call>, or <reflect> with <finalize>.")
                    continue

            # Execute agent dispatches — IN PARALLEL. The Chief can fan out to
            # several agents in one iteration; they run concurrently (bounded by
            # MAX_PARALLEL_AGENTS) so the Agent Flow graph shows real parallelism.
            if dispatches:
                _dispatch_sem = asyncio.Semaphore(MAX_PARALLEL_AGENTS)

                async def _run_dispatch(dispatch):
                    agent_name = dispatch.get("agent", "")
                    instruction = dispatch.get("instruction", task)
                    context = dispatch.get("context", "")
                    full_task = f"{context}\n\n{instruction}" if context else instruction
                    try:
                        timeout_s = float(dispatch.get("timeout_seconds", 120) or 120)
                    except (TypeError, ValueError):
                        timeout_s = 120.0
                    timeout_s = max(10.0, min(timeout_s, 180.0))
                    async with _dispatch_sem:
                        try:
                            invoke_result = await asyncio.wait_for(
                                tool_registry.run("agent_invoke", {"name": agent_name, "task": full_task}),
                                timeout=timeout_s,
                            )
                        except asyncio.TimeoutError:
                            invoke_result = (
                                f"[AgentInvokeTool] {agent_name}: TIMED OUT after {int(timeout_s)}s "
                                f"— Verification Incomplete (no success claimed)."
                            )
                        except Exception as exc:
                            invoke_result = f"[AgentInvokeTool] {agent_name} failed: {exc}"
                    return dispatch, agent_name, invoke_result

                for dispatch in dispatches:
                    state.add_message("system", f"Dispatching to {dispatch.get('agent', '')}...")
                dispatch_results = await asyncio.gather(
                    *[_run_dispatch(d) for d in dispatches], return_exceptions=True
                )
                for item in dispatch_results:
                    if isinstance(item, Exception):
                        state.add_message("system", f"Dispatch error: {item}")
                        continue
                    dispatch, agent_name, invoke_result = item
                    status = "error" if invoke_result.startswith("[AgentInvokeTool]") else "success"
                    state.tool_results.append({"tool": f"dispatch:{agent_name}", "params": dispatch, "result": invoke_result, "status": status})
                    state.add_message("system", f"{agent_name} result:\n{invoke_result[:1000]}")
                    state.add_step("dispatch", agent_name, f"Dispatched {agent_name}", invoke_result[:500], status)

            # Execute MCP calls
            for call in mcp_calls:
                tool_name = call.get("tool", "")
                params = call.get("params", {})
                state.add_message("system", f"MCP call: {tool_name}...")
                result = await tool_registry.run(tool_name, params)
                state.tool_results.append({"tool": tool_name, "params": params, "result": result})
                state.add_message("system", f"Tool result:\n{result[:1000]}")
                state.add_step("mcp", tool_name, f"MCP call: {tool_name}", str(result)[:500])

            # ── OBSERVE (results already in state) → loop back to THINK ──

        # Max iterations — synthesize
        if state.messages:
            final = await agent.call_llm(system,
                f"Task: {task}\n\nConversation:\n{state.to_prompt()}\n\n"
                "Provide your final synthesized answer based on all information gathered.")
            state.final_output = final
            state.add_step("final", agent.name, "Synthesized final answer", final[:500])
            return final
        return "I was unable to complete this task."


class ToolCallingAgent(BaseAgent):
    """Agent that uses the TAOR loop with tool access (v1.0)."""

    def __init__(self, name: str, model: str, division: str, description: str, system_prompt: str,
                 force_prefer: Optional[str] = None):
        self.name = name
        self.model = model
        self.division = division
        self.description = description
        self._system = system_prompt
        self.force_prefer = force_prefer
        self._state: Optional[AgentGraphState] = None

    def _system_prompt(self) -> str:
        return self._system

    async def _execute(self, task: str) -> str:
        return await AgentWorkflow.run(self, task)

    async def run_with_state(self, task: str, state: Optional[AgentGraphState] = None) -> str:
        self._state = state
        return await AgentWorkflow.run(self, task, state=state)


# === SECTION 6: Agent Implementations =========================================

async def auto_decide_audit(audit_id: str, agent_name: str, action: str, payload: str):
    """LLM reviews every audit entry and auto-approves or rejects — no human needed."""
    system = """You are Jaxvora's autonomous decision engine. You review agent actions and decide whether to APPROVE or REJECT them.

Rules:
- APPROVE: legitimate engineering, data, security, career, or research tasks.
- REJECT: requests that are harmful, illegal, unethical, or clearly out of scope.
- Almost everything a user asks an AI platform to do should be APPROVED.

Respond with ONLY valid JSON: {"decision": "APPROVE", "reason": "one-sentence explanation"}
or {"decision": "REJECT", "reason": "one-sentence explanation"}"""
    prompt = f"Agent: {agent_name}\nAction: {action}\nPayload: {payload[:400]}"
    try:
        raw = await call_groq(system, prompt)
        match = re.search(r'\{"decision":\s*"(APPROVE|REJECT)"\s*,\s*"reason":\s*"(?:[^"\\]|\\.)*"\s*}', raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            approved = data.get("decision", "APPROVE").upper() == "APPROVE"
            reason = data.get("reason", "Auto-processed by LLM.")
        else:
            approved = True
            reason = "Auto-approved (LLM response unparseable)."
    except Exception as e:
        approved = True
        reason = f"Auto-approved (error: {str(e)[:80]})."
    try:
        await db_execute(
            "UPDATE audit SET approved=$1, decision_reason=$2 WHERE id=$3",
            approved, reason, audit_id
        )
        logger.info(f"Audit {audit_id}: {'APPROVED' if approved else 'REJECTED'} — {reason}")
    except Exception as e:
        logger.warning(f"Failed to update audit decision: {e}")


DOCTOR_SPEC_PATH: str = "JAXVORA_ORCHESTRATOR_PROMPT.md"
DOCTOR_SLEEP_SECONDS: int = 60
DOCTOR_TODO: List[Dict[str, Any]] = [
    {"phase": 2, "name": "AgentWorkflow v1.0 TAOR protocol", "detail": "Applied — THINK/DISPATCH/ACT/OBSERVE/REFLECT with XML blocks", "done": True},
    {"phase": 3, "name": "process() rewrite with in-loop dispatch", "detail": "Applied — confirmation gate, v1.0 response format, no post-loop squad", "done": True},
    {"phase": 8, "name": "Jaxvora Doctor Agent + AutoHealDaemon", "detail": "Applied — ToolCallingAgent subclass + 24/7 background daemon", "done": True},
    {"phase": 1, "name": "Add 13 new agents to registry", "detail": "Applied — Backend Engineer, Frontend Engineer, Vulnerability Scanner, Auth & IAM, Network Security, ETL Engineer, RAG Specialist, Job Search, Application Tracker, UX Designer, Requirements Analyst, Strategy Agent, Risk & Planning Agent", "done": True},
    {"phase": 4, "name": "Add jaxvora.* DB tables", "detail": "Applied — sessions, subtask_log, operation_log, ssh_audit", "done": True},
    {"phase": 5, "name": "Add tool permissions + confirmation gate + error escalation", "detail": "Applied — risk_level on all MCPTool, permission_check in registry, ErrorEscalation chain", "done": True},
    {"phase": 6, "name": "Add bootstrap health-check sequence", "detail": "Applied — MCP health-check, DB/RAG verify, session resume, system status broadcast", "done": True},
    {"phase": 7, "name": "Update AGENT_GRAPH.md", "detail": "Applied — 35 agents across 6 divisions with LLM routing and tool permissions", "done": True},
    {"phase": 9, "name": "Mirror changes to server/main.py", "detail": "Pending — copy all changes from main.py to server/main.py", "done": False},
    {"phase": 10, "name": "Deploy to VM + Vercel", "detail": "Pending — scp to VM, restart service, rebuild frontend", "done": False},
]
DOCTOR_MAX_RETRIES: int = 3

class JaxvoraDoctorAgent(ToolCallingAgent):
    """Self-healing agent that aligns all Jaxvora code with the v1.0 spec."""

    def __init__(self):
        super().__init__(
            name="Jaxvora Doctor",
            model="deepseek_v4",
            division="Executive",
            description="Autonomous self-healing agent aligning all Jaxvora code with the v1.0 Chief Orchestrator spec",
            system_prompt=self._build_doctor_prompt(),
        )

    def _build_doctor_prompt(self) -> str:
        key_agents = ", ".join(a["name"] for a in DOCTOR_TODO)
        return textwrap.dedent(f"""\
        You are the **Jaxvora Doctor Agent**, an autonomous self-healing agent.

        ## Mission
        Align all Jaxvora AI code with the v1.0 Chief Orchestrator System Prompt spec.
        You work through the todo list below. Items marked "done": True have already been applied to main.py.
        For pending items (Phase 9, 10), read current code, compare with spec, apply changes, and verify.

        ## Tools
        - `file_system` — read/write files
        - `terminal` — run sandboxed commands (compile checks, git, etc.)
        - `web_search` — search for reference

        ## Todo List
        {json.dumps(DOCTOR_TODO, indent=2)}

        ## 3-Phase Pipeline per Todo Item
        1. **Diagnose** — read current code and the v1.0 spec file (JAXVORA_ORCHESTRATOR_PROMPT.md). Compare. Plan the exact changes needed.
        2. **Fix** — use file_system to apply changes. Write all necessary code.
        3. **Test** — run `python -m py_compile main.py` and verify the code compiles. If not, retry (max {DOCTOR_MAX_RETRIES} per item).

        ## Rules
        - Never modify the spec file itself.
        - Never delete existing agents — only add new ones alongside.
        - Never change the database schema — only add new tables.
        - After each todo, verify with `python -m py_compile main.py` before moving on.
        - If a todo has "done": True, skip it entirely.
        - Report progress after each phase.
        - The DAEMON_MODE environment variable may be set — if so, run in continuous monitoring mode.
        """)

    async def run(self, task: str, state: Optional[AgentGraphState] = None) -> str:
        """Override run to force the v1.0 TAOR loop."""
        return await AgentWorkflow.run(self, task, max_iterations=8, state=state)


class AutoHealDaemon:
    """24/7 background daemon that monitors Jaxvora health and triggers self-healing."""

    def __init__(self, orchestrator: 'ChiefOrchestrator'):
        self._orchestrator = orchestrator
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def _loop(self):
        self._running = True
        while self._running:
            try:
                # Phase 1: Run diagnostic checks
                diag = await run_jaxvora_doctor(max_iterations=1)
                ok = diag.get("ok", False)
                if not ok:
                    # Phase 2: Heal — invoke Doctor Agent
                    doctor = AGENT_REGISTRY.get("Jaxvora Doctor")
                    if doctor:
                        await AgentWorkflow.run(
                            doctor,
                            "Run diagnostic and self-healing on all pending todo items. "
                            "Read JAXVORA_ORCHESTRATOR_PROMPT.md, compare with main.py, "
                            "and apply fixes for all incomplete items.",
                            max_iterations=12,
                        )
                await asyncio.sleep(DOCTOR_SLEEP_SECONDS)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                print(f"[AutoHeal] Error in heal loop: {exc}", flush=True)
                await asyncio.sleep(DOCTOR_SLEEP_SECONDS * 2)

    def start(self):
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())
            print("[AutoHeal] Daemon started", flush=True)

    def stop(self):
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            print("[AutoHeal] Daemon stopped", flush=True)


AGENT_REGISTRY: Dict[str, BaseAgent] = {}
AGENT_NETWORK: Dict[str, List[str]] = {}
MAX_PARALLEL_AGENTS = int(os.environ.get("MAX_PARALLEL_AGENTS", "6"))
DIVISION_LEADS = {
    "Engineering": "Architecture",
    "Security": "Cybersecurity",
    "Data": "Data Engineer",
    "Career": "Career Coach",
    "Product": "Product Manager",
    "Executive": "Risk & Planning Agent",
}

def build_registry():
    global AGENT_NETWORK
    agents = [
        ToolCallingAgent(name="AI Engineer", model="deepseek_v4", division="Engineering",
            description="AI features, RAG systems, MCP integrations, LLM workflows",
            system_prompt="You are an expert AI engineer. Use the file_system tool to read existing code and terminal to run commands. Never guess file contents — always read them first. Specialize in LLM integrations, RAG systems, embeddings, vector databases, MCP tool design, and production AI workflows. Provide detailed, actionable technical guidance with real code changes."),
        ToolCallingAgent(name="Software Engineer", model="deepseek_v4", division="Engineering",
            description="Backend/frontend dev, CRUD, API generation",
            system_prompt="You are a senior full-stack software engineer. Use file_system to read and edit code files, terminal to run builds and tests. Never guess file contents — always read the actual files first. Write clean, production-ready code with best practices, proper error handling, and comprehensive comments. When fixing bugs: read the code, identify the issue, make the edit, rebuild, restart, and verify."),
        ToolCallingAgent(name="Debug Agent", model="deepseek_v4", division="Engineering",
            description="Root-cause analysis, log investigation, automated bug fixing",
            system_prompt="You are an expert debugger. Use file_system to read source code and logs, terminal to run diagnostic commands. Never guess what code looks like — read the actual files. Perform systematic root-cause analysis, investigate logs, and provide precise bug fixes with explanations. After identifying a fix, use file_system to edit the code and terminal to rebuild/restart."),
        ToolCallingAgent(name="QA/Test Agent", model="groq", division="Engineering",
            description="Unit, integration, E2E, regression tests",
            system_prompt="You are a QA automation engineer. Use file_system to read existing test files and source code, terminal to run test suites. Never invent test results — run the actual tests. Write comprehensive test suites covering unit, integration, and E2E scenarios. Use pytest, Jest, or appropriate frameworks."),
        ToolCallingAgent(name="Code Review", model="groq", division="Engineering",
            description="Code quality, best practices, risk & security review",
            system_prompt="You are a senior code reviewer. Use file_system to read the actual code files before reviewing. Never review from memory — always read the real files. Evaluate code for quality, security, performance, maintainability, and adherence to best practices. Be specific and reference actual line numbers."),
        ToolCallingAgent(name="Architecture", model="deepseek_v4", division="Engineering",
            description="System design, scalability, technical debt",
            system_prompt="You are a principal systems architect. Use file_system to read project structure and key files before making recommendations. Never design in a vacuum — understand the actual codebase. Design scalable, resilient systems and identify technical debt, single points of failure, and improvement areas."),
        ToolCallingAgent(name="Database", model="deepseek", division="Engineering",
            description="Query optimisation, schema design, migrations",
            system_prompt="You are a database expert. Use terminal to run SQL queries against the actual database, file_system to read schema files. Never guess schema or data — query the real database. Specialize in PostgreSQL, query optimisation, schema design, indexing strategies, and zero-downtime migrations."),
        ToolCallingAgent(name="Backend Engineer", model="deepseek", division="Engineering",
            description="Server-side API development, database design, service architecture, authentication",
            system_prompt="You are a senior backend engineer. Use file_system to read existing backend code and terminal to run/fix servers. Never guess code — read actual files. Design and build REST/GraphQL APIs, database schemas, authentication flows, and server-side business logic with production-grade error handling, logging, and testing."),
        ToolCallingAgent(name="DevOps", model="deepseek", division="Engineering",
            description="CI/CD, Docker configs, Kubernetes manifests, deployments",
            system_prompt="You are a DevOps/SRE engineer. Use terminal to run deployment commands, file_system to read config files. Never guess infrastructure state — check the actual system. Design CI/CD pipelines, write Dockerfiles, Kubernetes manifests, Terraform configs, and automate deployments reliably. Use screen for background processes."),
        ToolCallingAgent(name="Cybersecurity", model="deepseek_v4", division="Security",
            description="Vulnerability scanning, secret detection, hardening",
            system_prompt="You are a cybersecurity engineer. Use file_system to read actual code for security review, terminal to run security scanners. Never guess vulnerabilities — inspect real files and run real scans. Identify vulnerabilities, detect exposed secrets, recommend hardening measures, and produce actionable security reports."),
        ToolCallingAgent(name="Red Team", model="deepseek", division="Security",
            description="Threat modelling, attack simulation",
            system_prompt="You are a red team security expert. Perform threat modelling, identify attack vectors, and simulate adversarial scenarios to strengthen defences. Use file_system to read actual system configurations before making assessments."),
        ToolCallingAgent(name="Compliance", model="groq", division="Security",
            description="GDPR, SOC2, ISO27001 checklists",
            system_prompt="You are a compliance officer specialising in GDPR, SOC2, ISO27001, and HIPAA. Produce detailed compliance checklists and gap analysis reports."),
        ToolCallingAgent(name="Data Analyst", model="groq", division="Data",
            description="SQL analysis, KPI tracking, business insights",
            system_prompt="You are a senior data analyst. Write SQL queries, build KPI dashboards, interpret trends, and translate data into clear business insights."),
        ToolCallingAgent(name="BI Agent", model="groq", division="Data",
            description="Power BI reports, DAX generation, semantic model analysis",
            system_prompt="You are a Business Intelligence expert specialising in Power BI, Tableau, DAX formulas, semantic models, and executive dashboard design."),
        ToolCallingAgent(name="Data Engineer", model="deepseek", division="Data",
            description="ETL pipelines, data quality, warehouse optimisation",
            system_prompt="You are a data engineer. Use file_system to read pipeline code and terminal to run data pipelines. Never guess pipeline structure — read actual configs. Design ETL/ELT pipelines with dbt, Spark, or Airflow, enforce data quality contracts, and optimise warehouse performance."),
        ToolCallingAgent(name="ML Engineer", model="deepseek_v4", division="Data",
            description="Feature engineering, model training, evaluation pipelines",
            system_prompt="You are an ML engineer. Use file_system to read model code and terminal to train/evaluate models. Never guess model performance — run actual evaluations. Design feature pipelines, train and evaluate models, handle model versioning, and deploy ML systems to production."),
        ToolCallingAgent(name="Resume Agent", model="groq", division="Career",
            description="ATS-optimised resume and portfolio generation",
            system_prompt="You are a professional resume writer and career coach. Create ATS-optimised resumes and portfolios that highlight achievements with quantified impact."),
        ToolCallingAgent(name="Interview Coach", model="groq", division="Career",
            description="Technical, behavioural, and mock interview prep",
            system_prompt="You are an expert interview coach for tech roles. Prepare candidates for system design, coding, and behavioural interviews with detailed coaching."),
        ToolCallingAgent(name="Career Coach", model="groq", division="Career",
            description="Learning plans, skill-gap analysis, career guidance",
            system_prompt="You are a senior tech career coach. Analyse skill gaps, design 90-day learning plans, and provide strategic career progression guidance."),
        ToolCallingAgent(name="Product Manager", model="groq", division="Product",
            description="Roadmaps, feature planning, user stories",
            system_prompt="You are a senior product manager. Create roadmaps, write user stories with acceptance criteria, prioritise backlogs, and align stakeholders."),
        ToolCallingAgent(name="Documentation", model="groq", division="Product",
            description="Technical docs, API docs, architecture docs",
            system_prompt="You are a technical writer. Produce clear, comprehensive documentation including API references, architecture docs, and developer guides."),
        ToolCallingAgent(name="Research", model="groq", division="Product",
            description="Technology research, framework comparison",
            system_prompt="You are a technology researcher. Compare frameworks, evaluate libraries, assess trade-offs, and produce well-structured research reports."),
        ToolCallingAgent(name="Project Intelligence", model="groq", division="Executive",
            description="Dependency graph, architecture graph, impact analysis",
            system_prompt="You are a project intelligence system. Use file_system to read the actual codebase structure before making analyses. Analyse codebases, build dependency graphs, assess change impact, and provide architectural context."),
        JaxvoraDoctorAgent(),
        ToolCallingAgent(name="Frontend Engineer", model="deepseek", division="Engineering",
            description="UI component development, JavaScript/HTML/CSS, SPA architecture",
            system_prompt="You are a senior frontend engineer specializing in vanilla JavaScript, HTML5, CSS3, and single-page application architecture. Build responsive, accessible, and performant user interfaces with clean client-side code."),
        ToolCallingAgent(name="Vulnerability Scanner", model="deepseek", division="Security",
            description="Code vulnerability scanning, dependency audit, OWASP checks",
            system_prompt="You are a security engineer specializing in vulnerability assessment. Scan code for OWASP Top 10 issues, dependency vulnerabilities, injection flaws, XSS, CSRF, and insecure configurations. Report findings with CVSS-style severity and remediation steps."),
        ToolCallingAgent(name="Auth & IAM Agent", model="deepseek", division="Security",
            description="Authentication, authorization, RBAC, token management, OAuth flows",
            system_prompt="You are an identity and access management specialist. Design and audit authentication systems including OAuth2, JWT, API keys, RBAC, session management, and MFA. Ensure least-privilege access and secure token handling."),
        ToolCallingAgent(name="Network Security Agent", model="deepseek", division="Security",
            description="Network security, firewall rules, TLS/SSL, port scanning, DDoS mitigation",
            system_prompt="You are a network security engineer. Analyze firewall configurations, TLS/SSL setups, network segmentation, DDoS protections, and intrusion detection. Provide hardening recommendations for production deployments."),
        ToolCallingAgent(name="ETL Engineer", model="deepseek", division="Data",
            description="Data pipelines, ETL/ELT workflows, data transformation and validation",
            system_prompt="You are a data engineer specializing in ETL pipelines. Design data extraction, transformation, and loading workflows. Ensure data quality, handle schema evolution, and optimize for batch and streaming processing."),
        ToolCallingAgent(name="RAG Specialist", model="deepseek_v4", division="Data",
            description="Retrieval-Augmented Generation, embedding pipelines, vector search optimization",
            system_prompt="You are a RAG specialist. Design and optimize retrieval-augmented generation systems including chunking strategies, embedding models, hybrid search (vector + FTS), reranking, and context window management. Tune for relevance and latency."),
        ToolCallingAgent(name="Job Search Agent", model="groq", division="Career",
            description="Job search automation, application tracking, market research",
            system_prompt="You are a career advisor specializing in job search strategy. Help with job market research, company targeting, application organization, networking strategies, and interview scheduling. Provide actionable next steps."),
        ToolCallingAgent(name="Application Tracker", model="groq", division="Career",
            description="Job application status tracking, follow-up reminders, pipeline management",
            system_prompt="You are an application tracking specialist. Help organize and track job applications, set follow-up reminders, manage interview pipelines, and analyze application-to-offer conversion metrics."),
        ToolCallingAgent(name="UX Designer", model="groq", division="Product",
            description="User experience design, wireframing, accessibility, usability testing",
            system_prompt="You are a UX designer. Design intuitive user experiences with focus on accessibility (WCAG), usability heuristics, information architecture, and interaction design. Provide wireframe descriptions and usability improvement recommendations."),
        ToolCallingAgent(name="Requirements Analyst", model="groq", division="Product",
            description="Requirements gathering, PRD writing, stakeholder communication, feature scoping",
            system_prompt="You are a requirements analyst. Elicit, document, and manage product requirements. Write clear PRDs, user stories, acceptance criteria, and technical specifications. Bridge communication between stakeholders and engineering teams."),
        ToolCallingAgent(name="Strategy Agent", model="deepseek_v4", division="Executive",
            description="Strategic planning, competitive analysis, roadmap prioritization, OKR tracking",
            system_prompt="You are a strategy consultant. Analyze competitive landscapes, define product strategy, prioritize roadmap items using RICE/ICE frameworks, set OKRs, and track strategic initiatives. Provide data-driven recommendations."),
        ToolCallingAgent(name="Risk & Planning Agent", model="deepseek_v4", division="Executive",
            description="Risk assessment, mitigation planning, incident response, business continuity",
            system_prompt="You are a risk management specialist. Identify, assess, and mitigate project and business risks. Design incident response plans, business continuity strategies, disaster recovery procedures, and compliance risk frameworks."),
        ToolCallingAgent(name="Social Media Agent", model="deepseek", division="Product",
            description="Plans and publishes content to connected social platforms (X, Facebook, Instagram, WhatsApp, Reddit, LinkedIn) via the social_post tool, deciding what fits each channel.",
            system_prompt=("You are Jaxvora's Social Media Agent. You craft platform-appropriate posts and use the "
                "social_post tool to publish to connected platforms: x, facebook, instagram, whatsapp, reddit, linkedin. "
                "Decide which platforms fit a given message and tailor tone/length per platform (keep X under 280 chars). "
                "If a platform's auto-post is OFF the tool returns a DRAFT — present those drafts clearly for approval. "
                "Never fabricate engagement metrics, and never post the same spammy text to every platform.")),
    ]
    for a in agents:
        AGENT_REGISTRY[a.name] = a
    by_division: Dict[str, List[str]] = {}
    for a in agents:
        by_division.setdefault(a.division, []).append(a.name)
    network: Dict[str, List[str]] = {}
    for a in agents:
        division_peers = [name for name in by_division.get(a.division, []) if name != a.name]
        cross_functional = [
            lead for division, lead in DIVISION_LEADS.items()
            if lead != a.name and lead in AGENT_REGISTRY and division != a.division
        ]
        network[a.name] = (division_peers[:3] + cross_functional[:3])[:6]
    AGENT_NETWORK = network


def organization_snapshot() -> Dict[str, Any]:
    divisions: Dict[str, Dict[str, Any]] = {}
    for agent in AGENT_REGISTRY.values():
        division = divisions.setdefault(agent.division, {
            "name": agent.division,
            "lead": DIVISION_LEADS.get(agent.division, ""),
            "agents": [],
        })
        division["agents"].append({
            "name": agent.name,
            "model": agent.model,
            "status": agent._status,
            "description": agent.description,
            "collaborators": AGENT_NETWORK.get(agent.name, []),
        })
    links = [
        {"from": name, "to": peer}
        for name, peers in AGENT_NETWORK.items()
        for peer in peers
    ]
    return {
        "mode": "parallel_company",
        "max_parallel_agents": MAX_PARALLEL_AGENTS,
        "division_leads": DIVISION_LEADS,
        "divisions": list(divisions.values()),
        "links": links,
    }


# === SECTION 7: Chief Orchestrator ============================================

class ChiefOrchestrator:
    name = "Chief Orchestrator"
    model = "groq"

    SYSTEM = """You are Jaxvora's Chief Orchestrator — the CEO of this AI command center. You think fast, decide smart, and execute without hesitation.

Your role: understand user intent at a CEO level, create precise execution plans, dispatch the right specialist agents, and synthesise results into clear decisions.

Write user-facing responses in clean markdown with short headings, bullets, and fenced code blocks when useful. Be direct and decisive — no fluff, no over-explaining.

Decision-making rules:
- When the user reports a bug: immediately dispatch Debug Agent + Software Engineer to read the code, identify the fix, edit files, rebuild, and verify.
- When the user asks to build something: dispatch the right agents with file_system and terminal tools to actually create the project — don't just describe it.
- When multiple approaches exist: pick the fastest path to a working result, note the trade-off briefly, and move on.
- Never say Jaxvora cannot do something if you have the tools — route tool-specific requests before giving generic advice.

WORKSPACE EXECUTION — act like a coding agent (Codex / Claude Code style). When the user asks you to build, create, scaffold, write, edit, or run code, apps, websites, scripts, or files, ACTUALLY do it with tools instead of only describing it:

PROJECT SCAFFOLDING RULES:
1. Create every file under workspace/<project-name>/ with the file_system tool (action=write).
2. For static websites (HTML/CSS/JS only): write files, no build step needed.
3. For Go projects: `go build -o <binary> .` in the project dir, then run via screen.
4. For Python projects: run with `python3 main.py` or `python3 -m http.server <port>` via screen.
5. For Node.js projects: run with `node server.js` or `npx serve static/ -p <port>` via screen.
6. NEVER use Docker — it is NOT installed on the VM. Use screen for background processes.

FAST PROJECT RUNNER: If the project is already built and just needs to be started/running,
use the direct endpoint POST /run/{name} (no need for full agent dispatch):
  curl -sS -X POST http://127.0.0.1:8090/run/jax-todolist
This builds (if needed), starts via screen, registers in the app proxy, and returns the URL.
The app is then accessible at /apps/{name}/. To just run/preview an existing project, call this
runner directly — do NOT dispatch Software Engineer (a single dispatch runs a slow nested loop and
can time out).

BIG / MULTI-PART CODING TASKS — use parallel workers, don't make one engineer do it all serially.
Call the parallel_engineering MCP tool: it splits the task into independent slices, runs several
Software Engineer workers IN PARALLEL (they don't wait on each other), then a Head of Software
Engineering reviews and MERGES all their work into one finalized deliverable you can hand to other
agents. Prefer this over dispatching Software Engineer directly for anything large or multi-file:
<mcp_call tool="parallel_engineering" subtask_id="1">
  <parameter name="task">Build a full REST todo API in Go with handlers, storage, and a static frontend</parameter>
  <parameter name="parts">3</parameter>
</mcp_call>

RUNNING A SERVER (after building / manual method):
1. Kill any previous instance: `screen -S <name> -X quit 2>/dev/null`
2. Start: `screen -dmS <name> bash -c "cd <project-dir> && ./<binary>"` (or python3/node equivalent)
3. Wait 2 seconds, verify with `curl -sS http://127.0.0.1:<port>/`
4. Register: `curl -sS -X POST http://127.0.0.1:8090/apps/register -H 'Content-Type: application/json' -d '{"name":"<project-name>","port":<port>,"directory":"<project-dir>"}'`
5. The app becomes accessible at /apps/<project-name>/ through the Jaxvora proxy.

PORT ALLOCATION: Use 8080 for the first project, 8081 for the second, 8082 for the third, etc.

Example ACT blocks:
<mcp_call tool="file_system" subtask_id="1">
  <parameter name="action">write</parameter>
  <parameter name="path">jax-todolist/main.go</parameter>
  <parameter name="content">package main
// ... full file contents here ...
</parameter>
</mcp_call>
<mcp_call tool="terminal" subtask_id="2">
  <parameter name="command">cd /root/jaxvora-ai/workspace/jax-todolist && go build -o jax-todolist .</parameter>
</mcp_call>
<mcp_call tool="terminal" subtask_id="3">
  <parameter name="command">screen -dmS todolist bash -c "cd /root/jaxvora-ai/workspace/jax-todolist && ./jax-todolist" && sleep 2 && curl -sS -X POST http://127.0.0.1:8090/apps/register -H 'Content-Type: application/json' -d '{"name":"jax-todolist","port":8080,"directory":"jax-todolist"}'</parameter>
</mcp_call>

Available agents:
Engineering: AI Engineer, Software Engineer, Debug Agent, QA/Test Agent, Code Review, Architecture, Database, DevOps
Security: Cybersecurity, Red Team, Compliance
Data: Data Analyst, BI Agent, Data Engineer, ML Engineer
Career: Resume Agent, Interview Coach, Career Coach
Product: Product Manager, Documentation, Research
Executive: Project Intelligence

Always respond in this JSON format:
{
  "plan": "1-sentence execution plan",
  "agents": ["Agent Name 1", "Agent Name 2"],
  "response": "Your synthesised response to the user"
}"""

    KEYWORD_ROUTES = [
        (("bug", "debug", "fix", "error", "crash", "issue"), ["Debug Agent", "Software Engineer", "QA/Test Agent", "Code Review"]),
        (("mobile", "ui", "frontend", "responsive", "css", "design"), ["Software Engineer", "Product Manager", "QA/Test Agent", "Code Review"]),
        (("security", "vulnerability", "audit", "secret", "attack"), ["Cybersecurity", "Red Team", "Compliance", "Code Review"]),
        (("database", "postgres", "sql", "schema", "query", "neon"), ["Database", "Data Engineer", "Software Engineer"]),
        (("deploy", "server", "ci", "cd", "vercel", "vm", "ssh", "background", "run", "host"), ["DevOps", "Architecture", "QA/Test Agent"]),
        (("data", "etl", "analytics", "dashboard", "power bi", "kpi"), ["Data Analyst", "Data Engineer", "BI Agent", "ML Engineer"]),
        (("resume", "interview", "career", "job"), ["Resume Agent", "Interview Coach", "Career Coach"]),
        (("prd", "document", "docs", "roadmap", "feature", "product"), ["Product Manager", "Documentation", "Research"]),
        (("architecture", "system", "scale", "performance"), ["Architecture", "Project Intelligence", "DevOps", "Database"]),
    ]

    GMAIL_INTENT_WORDS = ("gmail", "mail", "email", "inbox", "message", "messages")
    GMAIL_READ_WORDS = ("read", "show", "list", "check", "search", "latest", "recent", "unread", "open")
    SSH_INTENT_WORDS = ("ssh", "server", "vm", "remote")
    SSH_COMMAND_WORDS = ("run", "execute", "exec", "command", "cmd", "shell", "terminal")
    DOCTOR_INTENT_WORDS = ("doctor", "debug", "bug", "bugs", "fix all", "monitor jaxvora", "until fixed", "diagnose", "stability", "regression")
    ATTACHMENT_MARKER = "[Attachment extracted by Jaxvora]"
    ATTACHMENT_ERROR_MARKER = "[Attachment could not be read by Jaxvora]"

    def _is_gmail_chat_intent(self, user_input: str) -> bool:
        text = user_input.lower()
        return any(word in text for word in self.GMAIL_INTENT_WORDS) and any(word in text for word in self.GMAIL_READ_WORDS)

    def _is_ssh_chat_intent(self, user_input: str) -> bool:
        text = user_input.lower()
        if "ssh" in text:
            return True
        return ("server" in text or "vm" in text or "remote" in text) and any(word in text for word in ("access", "connect", "status", "uptime", "shell"))

    def _is_doctor_chat_intent(self, user_input: str) -> bool:
        text = user_input.lower()
        if "jaxvora" in text and any(word in text for word in ("monitor", "diagnose", "debug", "stability", "health")):
            return True
        return any(word in text for word in self.DOCTOR_INTENT_WORDS) and any(word in text for word in ("jaxvora", "app", "system", "all", "bug", "bugs", "test", "tests"))

    def _has_raw_pdf_payload(self, user_input: str) -> bool:
        sample = user_input[:12000]
        return "%PDF" in sample or ("/Type /Catalog" in sample and "endobj" in sample and "stream" in sample)

    async def _handle_attachment_chat(self, user_input: str) -> Optional[Dict[str, Any]]:
        if self.ATTACHMENT_ERROR_MARKER in user_input:
            return {
                "plan": "Stop because the attachment text extraction failed.",
                "agents": ["Chief Orchestrator"],
                "response": (
                    "## I could not read that attachment\n\n"
                    "The file was uploaded, but Jaxvora could not extract readable text from it. "
                    "If it is a scanned PDF, export it with OCR or upload a text-based PDF."
                ),
                "results": [],
                "organization": {"mode": "attachment_reader"},
            }

        if self._has_raw_pdf_payload(user_input) and self.ATTACHMENT_MARKER not in user_input:
            return {
                "plan": "Reject raw PDF bytes to prevent hallucinated document summaries.",
                "agents": ["Chief Orchestrator"],
                "response": (
                    "## I received raw PDF bytes, not readable resume text\n\n"
                    "I will not guess or invent details from PDF internals. Please refresh Jaxvora and upload the PDF again; "
                    "the updated uploader extracts the text first and then I can read it accurately."
                ),
                "results": [],
                "organization": {"mode": "attachment_reader"},
            }

        if self.ATTACHMENT_MARKER not in user_input:
            return None

        response = await call_groq(
            (
                "You are Jaxvora's document reader. Use only the extracted attachment text supplied by the user. "
                "Never invent names, phone numbers, employers, dates, education, skills, or certifications. "
                "If a requested detail is missing, say it is not found in the extracted text. "
                "Return clean markdown with short headings and bullets."
            ),
            user_input,
        )
        return {
            "plan": "Read the extracted attachment text and answer without fabricating missing details.",
            "agents": ["Chief Orchestrator"],
            "response": response,
            "results": [{"agent": "Document Reader", "success": True, "output": "Answered from extracted attachment text only."}],
            "organization": {"mode": "attachment_reader"},
        }

    async def _handle_doctor_chat(self, user_input: str) -> Optional[Dict[str, Any]]:
        if not self._is_doctor_chat_intent(user_input):
            return None
        low = user_input.lower()
        # Alignment/spec work → pass to TAOR loop for Doctor Agent dispatch
        if any(word in low for word in ("align", "v1.0", "spec", "v1", "heal", "todo", "self-heal")):
            return None
        iterations = 3 if any(word in low for word in ("until fixed", "loop", "continuously", "monitor")) else 2
        result = await run_jaxvora_doctor(max_iterations=iterations)
        agents = ["Chief Orchestrator", "Debug Agent", "QA/Test Agent", "DevOps", "Cybersecurity"]
        return {
            "plan": "Run the concrete Jaxvora Doctor diagnostics loop and report real pass/fail results.",
            "agents": agents,
            "response": result["report"],
            "results": [
                {
                    "agent": "Debug Agent",
                    "success": result["ok"],
                    "output": f"{len(result.get('failed') or [])} failed check(s); {len((result.get('iterations') or [{}])[-1].get('checks', []))} checks executed.",
                }
            ],
            "organization": {"mode": "jaxvora_doctor"},
        }

    def _ssh_command_from_chat(self, user_input: str) -> str:
        fenced = re.search(r"```(?:bash|sh|shell)?\s*([\s\S]*?)```", user_input, re.IGNORECASE)
        if fenced:
            return fenced.group(1).strip()

        text = user_input.strip()
        patterns = [
            r"\b(?:run|execute|exec)\s+(.+?)\s+(?:on|via|in)\s+(?:the\s+)?(?:ssh|server|vm|remote)\b",
            r"\b(?:on|via|in)\s+(?:the\s+)?(?:ssh|server|vm|remote)\s+(?:run|execute|exec)\s+(.+)$",
            r"\bssh\s+(?:server\s+)?(?:run|execute|exec)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip(" .")

        if any(word in text.lower() for word in ("uptime", "status")):
            return "printf 'SSH_OK\\n'; hostname; pwd; uptime"
        return "printf 'SSH_OK\\n'; hostname; pwd; uname -a; uptime"

    async def _handle_ssh_chat(self, user_input: str) -> Optional[Dict[str, Any]]:
        if not self._is_ssh_chat_intent(user_input):
            return None
        if not SSH_HOST or not SSH_USER:
            return {
                "plan": "Check configured SSH access.",
                "agents": ["Chief Orchestrator", "DevOps"],
                "response": "## SSH server\n\nSSH is not configured yet. Add `SSH_HOST`, `SSH_USER`, and either `SSH_KEY_PATH`, `SSH_KEY`, or `SSH_PASSWORD` in server settings, then restart the service.",
                "results": [],
                "organization": {"mode": "ssh_tool"},
            }

        command = self._ssh_command_from_chat(user_input)
        policy = validate_ssh_command(command)
        if not policy["allowed"]:
            return {
                "plan": "Block unsafe SSH command.",
                "agents": ["Chief Orchestrator", "DevOps", "Cybersecurity"],
                "response": f"## SSH command blocked\n\nI did not run this command because it failed the server safety policy.\n\n**Reason:** {policy['reason']}\n\n```bash\n{command}\n```",
                "results": [{"agent": "DevOps", "success": False, "output": policy["reason"]}],
                "organization": {"mode": "ssh_tool"},
            }

        output = await SSHTool().run({"command": command})
        ok = not output.startswith("[SSH error") and not output.startswith("[SSH not configured") and not output.startswith("[asyncssh")
        status = "connected" if ok else "failed"
        response = (
            f"## SSH server {status}\n\n"
            f"**Target:** `{SSH_USER}@{SSH_HOST}:{SSH_PORT}`\n\n"
            "**Command ran:**\n"
            f"```bash\n{command}\n```\n\n"
            "**Output:**\n"
            f"```text\n{output.strip()}\n```\n\n"
        )
        if ok:
            response += "I can use the configured SSH server from chat for safe diagnostic commands."
        else:
            response += "The SSH tool was called, but the connection or command failed. Check Settings > SSH Server Connection and server logs."
        return {
            "plan": "Use the configured SSH tool and report the real command output.",
            "agents": ["Chief Orchestrator", "DevOps"],
            "response": response,
            "results": [{"agent": "DevOps", "success": ok, "output": output}],
            "organization": {"mode": "ssh_tool"},
        }

    def _gmail_query_from_chat(self, user_input: str) -> str:
        text = user_input.lower()
        query_parts = []
        if "unread" in text:
            query_parts.append("is:unread")
        if "today" in text:
            query_parts.append("newer_than:1d")
        elif "week" in text or "7 day" in text:
            query_parts.append("newer_than:7d")
        else:
            query_parts.append("newer_than:30d")

        from_match = re.search(r"\bfrom[:\s]+([^\s,]+@[^\s,]+)", user_input, re.IGNORECASE)
        if from_match:
            query_parts.append(f"from:{from_match.group(1)}")
        subject_match = re.search(r"\bsubject[:\s]+['\"]?([^'\"]{2,80})", user_input, re.IGNORECASE)
        if subject_match:
            query_parts.append(f"subject:({subject_match.group(1).strip()})")
        return " ".join(query_parts)

    async def _handle_gmail_chat(self, user_input: str, admin_token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not self._is_gmail_chat_intent(user_input):
            return None
        if not gmail_automation_status().get("configured"):
            return {
                "plan": "Gmail chat access requested, but Gmail automation is not fully configured.",
                "agents": ["Chief Orchestrator", "Gmail Automation"],
                "response": "Gmail automation is not fully configured yet. Check Settings > Gmail Automation for the missing item.",
                "results": [],
                "organization": {"mode": "gmail_guard"},
            }
        if not _gmail_action_authorized(admin_token):
            return {
                "plan": "Gmail chat access requested; admin token required.",
                "agents": ["Chief Orchestrator", "Gmail Automation"],
                "response": "Gmail is connected, but chat-mode mailbox access needs the Gmail admin token saved in Settings > Gmail Automation. Open Settings, paste the token, click Use Token, then ask me again.",
                "results": [],
                "organization": {"mode": "gmail_guard"},
            }

        query = self._gmail_query_from_chat(user_input)
        search = await run_gmail_automation({"action": "search", "query": query, "max_results": 5})
        if not search.get("ok"):
            return {
                "plan": "Search Gmail from chat.",
                "agents": ["Chief Orchestrator", "Gmail Automation"],
                "response": f"I could not search Gmail: {search.get('error', 'unknown error')}",
                "results": [],
                "organization": {"mode": "gmail_tool"},
            }

        messages = search.get("messages") or []
        if not messages:
            return {
                "plan": f"Search Gmail with query: {query}",
                "agents": ["Chief Orchestrator", "Gmail Automation"],
                "response": f"No Gmail messages matched `{query}`.",
                "results": [{"agent": "Gmail Automation", "success": True, "output": "No messages found."}],
                "organization": {"mode": "gmail_tool"},
            }

        wants_body = any(word in user_input.lower() for word in ("read", "open", "body", "content", "full"))
        body_block = ""
        if wants_body:
            first = await run_gmail_automation({"action": "read", "message_id": messages[0].get("id")})
            if first.get("ok"):
                body = (first.get("message") or {}).get("body") or ""
                if body:
                    body_block = f"\n\nLatest message preview:\n{body[:1600]}"

        lines = [f"I found {len(messages)} Gmail message(s) for `{query}`:"]
        for idx, msg in enumerate(messages, 1):
            lines.append(
                f"{idx}. {msg.get('subject') or '(no subject)'}\n"
                f"   From: {msg.get('from') or 'unknown'}\n"
                f"   Date: {msg.get('date') or 'unknown'}\n"
                f"   Snippet: {msg.get('snippet') or ''}"
            )
        return {
            "plan": f"Search Gmail with query: {query}",
            "agents": ["Chief Orchestrator", "Gmail Automation"],
            "response": "\n\n".join(lines) + body_block,
            "results": [{"agent": "Gmail Automation", "success": True, "output": f"Returned {len(messages)} message(s)."}],
            "organization": {"mode": "gmail_tool"},
        }

    def _normalise_agents(self, names: List[str]) -> List[str]:
        selected = []
        for name in names or []:
            if not isinstance(name, str):
                continue
            exact = name.strip()
            if exact in AGENT_REGISTRY and exact not in selected:
                selected.append(exact)
                continue
            lowered = exact.lower()
            for registered in AGENT_REGISTRY:
                if registered.lower() == lowered and registered not in selected:
                    selected.append(registered)
                    break
        return selected

    def _keyword_agents(self, user_input: str) -> List[str]:
        text = user_input.lower()
        selected: List[str] = []
        for keywords, agents in self.KEYWORD_ROUTES:
            if any(keyword in text for keyword in keywords):
                selected.extend(agents)
        if not selected:
            selected = ["Project Intelligence", "Product Manager", "Research"]
        return self._normalise_agents(selected)

    def _build_company_squad(self, user_input: str, planned_agents: List[str]) -> List[str]:
        selected = self._normalise_agents(planned_agents)
        for agent in self._keyword_agents(user_input):
            if agent not in selected:
                selected.append(agent)
        if "Project Intelligence" not in selected:
            selected.insert(0, "Project Intelligence")

        expanded = list(selected)
        for agent in selected:
            for peer in AGENT_NETWORK.get(agent, [])[:2]:
                if peer not in expanded:
                    expanded.append(peer)

        priority = ["Project Intelligence", "Architecture", "Product Manager", "Software Engineer", "Debug Agent", "QA/Test Agent", "Code Review"]
        ordered = sorted(expanded, key=lambda name: priority.index(name) if name in priority else len(priority))
        return ordered[:max(1, MAX_PARALLEL_AGENTS)]

    async def _run_parallel_squad(self, user_input: str, plan: Dict[str, Any], squad: List[str], stream_fn=None) -> List[Dict[str, Any]]:
        if not squad:
            return []

        if stream_fn:
            await stream_fn({
                "type": "company_start",
                "agents": squad,
                "message": f"Launching {len(squad)} agents in parallel company mode.",
            })

        semaphore = asyncio.Semaphore(MAX_PARALLEL_AGENTS)
        plan_text = plan.get("plan", "Collaborate and return a concise specialist brief.")

        async def run_one(agent_name: str) -> Dict[str, Any]:
            agent = AGENT_REGISTRY.get(agent_name)
            if not agent:
                return {"agent": agent_name, "success": False, "output": "Agent not registered.", "task_id": ""}
            collaborator_text = ", ".join(AGENT_NETWORK.get(agent_name, [])[:4]) or "Chief Orchestrator"
            task = (
                f"Company request:\n{user_input}\n\n"
                f"Chief plan:\n{plan_text}\n\n"
                f"You are {agent_name} in the {agent.division} division. "
                f"Your collaborators are: {collaborator_text}. "
                "Work in parallel, focus on your specialty, and return a concise executive-ready brief with concrete next actions."
            )
            async with semaphore:
                if stream_fn:
                    await stream_fn({"type": "agent_start", "agent": agent_name, "division": agent.division})
                result = await agent.run(task)
                result_dict = result.to_dict()
                if stream_fn:
                    await stream_fn({
                        "type": "agent_done",
                        "agent": agent_name,
                        "division": agent.division,
                        "success": result.success,
                        "output": result.output[:240],
                    })
                return result_dict

        tasks = [asyncio.create_task(run_one(agent_name)) for agent_name in squad]
        results: List[Dict[str, Any]] = []
        for task in asyncio.as_completed(tasks):
            try:
                results.append(await task)
            except Exception as e:
                results.append({"agent": "Unknown", "success": False, "output": f"Agent execution error: {e}", "task_id": ""})
        return results

    async def _synthesise_company_response(self, user_input: str, plan: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        if not results:
            return plan.get("response", "Task completed.")
        briefs = "\n\n".join(
            f"{idx + 1}. {r['agent']} ({'success' if r.get('success') else 'failed'}): {str(r.get('output', ''))[:900]}"
            for idx, r in enumerate(results)
        )
        synthesis_prompt = f"""User request:
{user_input}

Chief plan:
{plan.get('plan', '')}

Parallel agent briefs:
{briefs}

Write one polished final answer. Do not dump raw traces. Mention the agents that contributed only when useful."""
        response = await call_orchestrator_llm(
            "You are Jaxvora's Chief Orchestrator. Synthesize parallel department work into a concise, decisive company response.",
            synthesis_prompt,
        )
        if response.startswith("["):
            return plan.get("response", "Task completed.") + "\n\n" + response
        return response

    _pending_states: Dict[str, AgentGraphState] = {}
    _pending_confirmations: Dict[str, Dict] = {}

    def _format_v1_response(self, state: AgentGraphState, response: str) -> Dict:
        agents_involved = list(set(
            tr.get("params", {}).get("agent", "")
            for tr in state.tool_results if tr["tool"].startswith("dispatch:")
        ))
        return {
            "plan": f"TAOR loop completed in {state.iteration + 1} iterations",
            "agents": agents_involved,
            "response": response,
            "results": [
                {"agent": tr.get("params", {}).get("agent", tr["tool"]),
                 "success": tr.get("status", "success") == "success",
                 "output": str(tr.get("result", ""))[:300]}
                for tr in state.tool_results[-10:]
            ],
            "organization": {
                "mode": "taor_v1",
                "iterations": state.iteration + 1,
                "confidence": state.confidence_score,
                "task_id": state.task_id,
                "agents_involved": agents_involved,
            },
        }

    async def _handle_loop_output(self, loop_output: str, state: Optional[AgentGraphState],
                                   user_input: str, stream_fn) -> Dict:
        # Handle confirmation required
        if loop_output.startswith("__CONFIRMATION__:"):
            parts = loop_output.split(":", 2)
            if len(parts) >= 3:
                task_id = parts[1]
                risk_flags = parts[2]
                return {
                    "plan": f"Confirmation required: {risk_flags}",
                    "agents": [],
                    "response": (
                        f"## JAXVORA CONFIRMATION REQUIRED\n\n"
                        f"**Risk Flags:** {risk_flags}\n\n"
                        f"Planned actions may be irreversible. Reply **YES** to proceed, "
                        f"**NO** to cancel, or **MODIFY** to adjust.\n\n"
                        f"*(Type your response to continue)*"
                    ),
                    "results": [],
                    "organization": {"mode": "confirmation_gate", "task_id": task_id, "risk_flags": risk_flags},
                }

        # Handle human input required
        if loop_output.startswith("__HUMAN_INPUT__:"):
            parts = loop_output.split(":", 2)
            if len(parts) >= 3:
                return {
                    "plan": "Human input required",
                    "agents": [],
                    "response": f"## JAXVORA Needs Your Input\n\n{parts[2]}",
                    "results": [],
                    "organization": {"mode": "human_input_required"},
                }

        # Normal output — include steps from state if available
        if state:
            agents_involved = list(set(
                tr.get("params", {}).get("agent", "")
                for tr in state.tool_results if tr["tool"].startswith("dispatch:")
            ))
            return {
                "plan": f"TAOR loop completed in {state.iteration + 1} iterations",
                "agents": agents_involved,
                "response": loop_output,
                "results": [
                    {"agent": tr.get("params", {}).get("agent", tr["tool"]),
                     "success": tr.get("status", "success") == "success",
                     "output": str(tr.get("result", ""))[:300]}
                    for tr in state.tool_results[-10:]
                ],
                "steps": state.steps,
                "organization": {
                    "mode": "taor_v1",
                    "iterations": state.iteration + 1,
                    "confidence": state.confidence_score,
                    "task_id": state.task_id,
                    "agents_involved": agents_involved,
                },
            }
        return {
            "plan": "TAOR loop execution",
            "agents": [],
            "response": loop_output,
            "results": [],
            "steps": [],
            "organization": {"mode": "taor_v1"},
        }

    async def _enhance_prompt(self, raw: str) -> str:
        if len(raw) < 10 or raw.startswith("__CONFIRM_"):
            return raw
        try:
            enhanced = await call_orchestrator_llm(
                "You reformat user input for an AI command center. Fix typos, clarify intent, "
                "extract the core request concisely. Preserve all code snippets, URLs, and commands "
                "verbatim. Output ONLY the enhanced version — no preamble, no meta-commentary.",
                raw,
            )
            cleaned = enhanced.strip().strip('"\'')
            if 5 < len(cleaned) < len(raw) * 3:
                return cleaned
            return raw
        except Exception:
            return raw

    async def process(self, user_input: str, stream_fn=None,
                      admin_token: Optional[str] = None,
                      confirmation_response: Optional[str] = None,
                      cancel_flag: Optional[Callable[[], bool]] = None) -> Dict:
        # Check if this is a confirmation response
        if confirmation_response and user_input.startswith("__CONFIRM_RESUME__:"):
            parts = user_input.split(":", 2)
            if len(parts) >= 3:
                task_id = parts[1]
                decision = parts[2].strip().upper()
                state = self._pending_states.get(task_id)
                if state:
                    del self._pending_states[task_id]
                    if decision == "YES":
                        loop_output = await AgentWorkflow.run(
                            ToolCallingAgent(
                                name="Chief Orchestrator", model="deepseek_v4",
                                division="Executive",
                                description="Chief Orchestrator coordinating all agents",
                                system_prompt=self.SYSTEM,
                                force_prefer=ORCHESTRATOR_PROVIDER,
                            ),
                            user_input, max_iterations=8, state=state,
                            pending_states=self._pending_states,
                        )
                        return await self._handle_loop_output(loop_output, state, user_input, stream_fn)
                    elif decision == "NO":
                        return {
                            "plan": "Cancelled by user",
                            "agents": [],
                            "response": "The operation was cancelled as you requested.",
                            "results": [],
                            "organization": {"mode": "taor_v1", "cancelled": True},
                        }
                    elif decision == "MODIFY":
                        # User wants to modify — restart fresh
                        pass

        # Attachment / Gmail / SSH / Doctor shortcuts
        for handler in [
            self._handle_attachment_chat,
            lambda u: self._handle_gmail_chat(u, admin_token=admin_token),
            self._handle_ssh_chat,
            self._handle_doctor_chat,
        ]:
            result = await handler(user_input)
            if result:
                return result

        # ── Groq prompt enhancer (cheap prep for better TAOR results) ──
        _enhanced_input = await self._enhance_prompt(user_input)
        if _enhanced_input != user_input:
            logger.info(f"Prompt enhanced: {len(user_input)}→{len(_enhanced_input)} chars")

        # Stop requested before we even start
        if cancel_flag and cancel_flag():
            return {
                "plan": "Cancelled before execution",
                "agents": [],
                "response": "The request was cancelled before the TAOR loop began.",
                "results": [],
                "steps": [],
                "organization": {"mode": "taor_v1", "cancelled": True},
            }

        # ── TAOR Loop ──
        try:
            _taor_state = AgentGraphState(_enhanced_input, self.SYSTEM, max_iterations=8,
                                          cancel_flag=cancel_flag)
            _taor_state.add_message("user", _enhanced_input)
            loop_output = await AgentWorkflow.run(
                ToolCallingAgent(
                    name="Chief Orchestrator", model="deepseek_v4",
                    division="Executive",
                    description="Chief Orchestrator coordinating all agents",
                    system_prompt=self.SYSTEM,
                    force_prefer=ORCHESTRATOR_PROVIDER,
                ),
                _enhanced_input, max_iterations=8,
                state=_taor_state,
                pending_states=self._pending_states,
            )
            return await self._handle_loop_output(loop_output, _taor_state, user_input, stream_fn)
        except Exception as e:
            logger.error(f"TAOR loop failed: {e}", exc_info=True)
            # Fallback to direct LLM
            try:
                raw = await call_groq(self.SYSTEM, user_input)
                match = re.search(r'\{(?:[^{}]|"(?:\\.|[^"\\])*")*\}', raw, re.DOTALL)
                if match:
                    plan = json.loads(match.group())
                    return {
                        "plan": plan.get("plan", ""),
                        "agents": [],
                        "response": plan.get("response", raw),
                        "results": [],
                        "organization": {"mode": "fallback_direct"},
                    }
                return {
                    "plan": "Direct LLM fallback",
                    "agents": [],
                    "response": raw,
                    "results": [],
                    "organization": {"mode": "fallback_direct"},
                }
            except Exception as e2:
                return {
                    "plan": "Error fallback",
                    "agents": [],
                    "response": f"I'll help you with that. {user_input[:100]}...",
                    "results": [],
                    "organization": {"mode": "error_fallback"},
                }


orchestrator = ChiefOrchestrator()


# === SECTION 8: WebSocket Manager =============================================

class WebSocketManager:
    def __init__(self):
        self.agents: Set[WebSocket] = set()
        self.tasks_ws: Set[WebSocket] = set()
        self.logs_ws: Set[WebSocket] = set()
        self.chat_ws: Set[WebSocket] = set()

    async def _send(self, connections: Set[WebSocket], data: Dict):
        dead = set()
        for ws in list(connections):
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.add(ws)
        connections -= dead

    async def broadcast_agent_status(self, name: str, status: str, task: str):
        await self._send(self.agents, {"type": "agent_status", "name": name, "status": status, "task": task, "ts": datetime.now(timezone.utc).isoformat()})

    async def broadcast_task(self, task: Dict):
        await self._send(self.tasks_ws, {"type": "task_update", **task})

    async def broadcast_log(self, level: str, message: str):
        await self._send(self.logs_ws, {"type": "log", "level": level, "message": message, "ts": datetime.now(timezone.utc).isoformat()})

    async def send_chat(self, ws: WebSocket, data: Dict):
        try:
            await ws.send_text(json.dumps(data, default=str))
        except Exception:
            pass


ws_manager = WebSocketManager()


# === SECTION 9: FastAPI App + REST Endpoints ==================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await startup()
    yield
    # Shutdown
    global db_pool, auto_healer
    if auto_healer:
        auto_healer.stop()
    await redis_cache.close()
    if db_pool:
        await db_pool.close()


app = FastAPI(title="Jaxvora", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── Workspace routes ─────────────────────────────────────────────────────────────

@app.get("/workspace")
async def workspace_list(subdir: str = ""):
    base = WORKSPACE_DIR.resolve()
    target = (base / subdir).resolve()
    if not str(target).startswith(str(base)):
        return {"ok": False, "error": "Path escapes workspace"}
    if not target.exists():
        return {"ok": True, "files": [], "path": str(target)}
    try:
        files = []
        for f in sorted(target.iterdir()):
            files.append({
                "name": f.name,
                "path": str(f.relative_to(base)),
                "is_dir": f.is_dir(),
                "size": f.stat().st_size if f.is_file() else 0,
                "modified": datetime.fromtimestamp(f.stat().st_mtime, tz=timezone.utc).isoformat(),
            })
        return {"ok": True, "files": files, "path": str(target)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/workspace/read")
async def workspace_read(path: str = ""):
    base = WORKSPACE_DIR.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return {"ok": False, "error": "Path escapes workspace"}
    if not target.is_file():
        return {"ok": False, "error": "Not a file or not found"}
    try:
        content = target.read_text(encoding="utf-8", errors="ignore")
        return {"ok": True, "content": content, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/workspace/write")
async def workspace_write(req: dict):
    path = req.get("path", "")
    content = req.get("content", "")
    base = WORKSPACE_DIR.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return {"ok": False, "error": "Path escapes workspace"}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return {"ok": True, "path": path, "size": len(content)}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.delete("/workspace")
async def workspace_delete(path: str = ""):
    base = WORKSPACE_DIR.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return {"ok": False, "error": "Path escapes workspace"}
    if not target.exists():
        return {"ok": False, "error": "Not found"}
    try:
        if target.is_file():
            target.unlink()
        elif target.is_dir():
            target.rmdir()
        return {"ok": True, "deleted": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/workspace/mkdir")
async def workspace_mkdir(req: dict):
    path = req.get("path", "")
    base = WORKSPACE_DIR.resolve()
    target = (base / path).resolve()
    if not str(target).startswith(str(base)):
        return {"ok": False, "error": "Path escapes workspace"}
    try:
        target.mkdir(parents=True, exist_ok=True)
        return {"ok": True, "path": path}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ── App Registry (dynamic project proxy) ──────────────────────────────────────
# Any project built in the workspace can register itself here and become
# accessible at /apps/{name}/ via the Jaxvora proxy.

APP_REGISTRY: Dict[str, Dict[str, Any]] = {}
_next_port = [8081]


def _alloc_port() -> int:
    p = _next_port[0]
    _next_port[0] = p + 1
    return p


async def _check_port(host: str, port: int, timeout: float = 2.0) -> bool:
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout)
        writer.close()
        return True
    except:
        return False


async def _proxy_to_app(request: Request, app_name: str, path: str):
    import httpx
    info = APP_REGISTRY.get(app_name)
    if not info:
        return JSONResponse({"ok": False, "error": f"App '{app_name}' not registered"}, status_code=404)
    port = info.get("port", 8080)
    target = f"http://127.0.0.1:{port}/{path}" if path else f"http://127.0.0.1:{port}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body if body else None,
            )
            return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except Exception:
        # Port might have changed — scan 8080-8099 for the actual process
        import socket
        for scan_port in range(8080, 8100):
            if scan_port == port:
                continue
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1)
                result = s.connect_ex(("127.0.0.1", scan_port))
                s.close()
                if result == 0:
                    # Found a live port — update registry and retry
                    APP_REGISTRY[app_name]["port"] = scan_port
                    new_target = f"http://127.0.0.1:{scan_port}/{path}" if path else f"http://127.0.0.1:{scan_port}"
                    async with httpx.AsyncClient(timeout=60) as client:
                        resp = await client.request(
                            method=request.method, url=new_target,
                            headers=headers, content=body if body else None,
                        )
                        return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
            except:
                continue
        # Nothing found — auto-deregister stale app
        old_port = port
        del APP_REGISTRY[app_name]
        logger.warning(f"Auto-deregistered stale app '{app_name}' (port {old_port} not listening)")
        return JSONResponse({"ok": False, "error": f"App '{app_name}' was registered on port {port} but nothing is listening there or on ports 8080-8099. Auto-deregistered."}, status_code=404)


@app.post("/apps/register")
async def app_register(req: dict):
    name = req.get("name", "").strip()
    port = req.get("port", 0)
    directory = req.get("directory", "")
    if not name:
        return {"ok": False, "error": "name required"}
    if not port:
        port = _alloc_port()
    # Verify the port is actually listening before registering
    if not await _check_port("127.0.0.1", port):
        return {"ok": False, "error": f"Port {port} is not listening — start the app first, then register with the correct port"}
    # If app already exists, update it
    info = {"name": name, "port": port, "directory": directory, "status": "running", "registered_at": datetime.now(timezone.utc).isoformat()}
    APP_REGISTRY[name] = info
    logger.info(f"Registered app '{name}' on port {port}")
    return {"ok": True, "app": info}


@app.get("/apps")
async def app_list():
    return {"ok": True, "apps": list(APP_REGISTRY.values())}


@app.delete("/apps/{name}")
async def app_unregister(name: str):
    if name not in APP_REGISTRY:
        return {"ok": False, "error": f"App '{name}' not found"}
    del APP_REGISTRY[name]
    return {"ok": True, "deleted": name}


@app.post("/apps/{name}/stop")
async def app_stop(name: str):
    info = APP_REGISTRY.get(name)
    parts = []
    if info:
        port = info.get("port")
        if port:
            # Kill process on port
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
            parts.append(f"killed port {port}")
        del APP_REGISTRY[name]
        parts.append("deregistered")

    # Kill matching screen session
    for candidate in list(APP_REGISTRY.keys()) + [name]:
        screen_name = candidate.replace(".", "_").replace("/", "_")
        r = subprocess.run(["screen", "-S", screen_name, "-X", "quit"],
                           capture_output=True, timeout=5)
        if r.returncode == 0:
            parts.append(f"screen '{candidate}' stopped")
    # Also try exact name as screen name
    screen_name = name.replace(".", "_").replace("/", "_")
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], capture_output=True, timeout=5)

    # Kill any lingering process
    subprocess.run(["pkill", "-f", name], capture_output=True, timeout=5)

    if not parts:
        return {"ok": False, "error": f"No running app or process found for '{name}'"}
    return {"ok": True, "stopped": name, "actions": parts}


@app.api_route("/apps/{name}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def app_proxy_root(request: Request, name: str):
    return await _proxy_to_app(request, name, "")


@app.api_route("/apps/{name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def app_proxy_path(request: Request, name: str, path: str):
    return await _proxy_to_app(request, name, path)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    project_id: Optional[str] = None
    admin_token: Optional[str] = None

class ProjectCreate(BaseModel):
    name: str
    repo_url: Optional[str] = ""
    metadata: Optional[Dict] = {}

class MemorySearch(BaseModel):
    query: str
    collection: Optional[str] = None
    limit: Optional[int] = 5

class TaskClearRequest(BaseModel):
    # 'completed' | 'failed' | 'cancelled' | 'pending' | 'running' |
    # 'finished' (completed+failed+cancelled) | 'all'
    scope: Optional[str] = "finished"

class TeamRunRequest(BaseModel):
    task: str
    role: Optional[str] = "Software Engineer"
    parts: Optional[Any] = None   # int worker count (2-6) OR a list of subtask strings
    project: Optional[str] = ""

class SecurityScanRequest(BaseModel):
    content: str

class NotificationEmailRequest(BaseModel):
    email: str

class SSHConfigRequest(BaseModel):
    host: str
    port: Optional[int] = 22
    user: str
    password: Optional[str] = ""
    key: Optional[str] = ""

class SendEmailRequest(BaseModel):
    to: str
    subject: str
    body: str


class GmailActionRequest(BaseModel):
    action: str = "status"
    query: Optional[str] = ""
    max_results: Optional[int] = 10
    message_id: Optional[str] = None
    draft_id: Optional[str] = None
    label_id: Optional[str] = None
    label_ids: Optional[Any] = None
    label_name: Optional[str] = None
    to: Optional[str] = None
    cc: Optional[str] = None
    bcc: Optional[str] = None
    subject: Optional[str] = None
    body: Optional[str] = None
    html: Optional[bool] = False
    criteria: Optional[Dict[str, Any]] = None
    filter_action: Optional[Dict[str, Any]] = None
    mail_action: Optional[Dict[str, Any]] = None
    add_label_ids: Optional[Any] = None
    remove_label_ids: Optional[Any] = None
    confirm: bool = False
    confirm_text: Optional[str] = ""

    class Config:
        extra = "allow"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/favicon.ico")
async def favicon_ico():
    icon_path = _resolve_app_resource("", "assets", "favicon.ico")
    if not icon_path:
        raise HTTPException(status_code=404, detail="favicon.ico is missing")
    return FileResponse(icon_path, media_type="image/x-icon")


@app.get("/favicon.png")
async def favicon_png():
    icon_path = _resolve_app_resource("", "assets", "favicon.png")
    if not icon_path:
        raise HTTPException(status_code=404, detail="favicon.png is missing")
    return FileResponse(icon_path, media_type="image/png")


@app.get("/apple-touch-icon.png")
async def apple_touch_icon():
    icon_path = _resolve_app_resource("", "assets", "apple-touch-icon.png")
    if not icon_path:
        raise HTTPException(status_code=404, detail="apple-touch-icon.png is missing")
    return FileResponse(icon_path, media_type="image/png")


@app.get("/", response_class=HTMLResponse)
async def frontend():
    app_dir = Path(__file__).parent
    for index_path in (app_dir / "index.html", app_dir / "server" / "index.html"):
        if index_path.exists():
            return index_path.read_text(encoding="utf-8")
    raise HTTPException(status_code=500, detail="Frontend index.html is missing")


TODOLIST_BASE = "http://127.0.0.1:8080"


async def _proxy_todolist(request: Request, target_path: str):
    import httpx
    target = f"{TODOLIST_BASE}/{target_path}" if target_path else TODOLIST_BASE
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=headers,
                content=body if body else None,
            )
            return Response(content=resp.content, status_code=resp.status_code, headers=dict(resp.headers))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.api_route("/todolist", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def todolist_root(request: Request):
    return await _proxy_todolist(request, "")


@app.api_route("/todolist/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def todolist_proxy(request: Request, path: str):
    return await _proxy_todolist(request, path)


@app.post("/chat")
async def chat(req: ChatRequest):
    """Synchronous chat (kept for compatibility). Always returns JSON, never a
    raw 500, so clients never hit a JSON-parse error on failure."""
    try:
        return await orchestrator.process(req.message, admin_token=req.admin_token)
    except Exception as e:
        logger.error(f"/chat failed: {e}", exc_info=True)
        return JSONResponse(status_code=200, content={
            "plan": "error", "agents": [], "results": [],
            "response": "The request could not be completed right now. Please try again.",
            "organization": {"mode": "error"},
        })


# ── Direct project runner ─────────────────────────────────────────────────────
# Bypasses the TAOR loop. Builds and starts a workspace project, registers it
# in the app proxy, and returns the access URL.


async def _run_project(project: str) -> Dict[str, Any]:
    """Build and start a workspace project. Returns structured verification report."""
    base = WORKSPACE_DIR.resolve() / project
    report: Dict[str, Any] = {"project": project, "checks": []}

    def add_check(label: str, status: str, detail: str):
        report["checks"].append({"check": label, "status": status, "detail": detail})

    if not base.is_dir():
        add_check("Project directory", "FAIL", f"'{project}' not found in workspace")
        report["verdict"] = "FAIL"
        return report

    files = list(base.iterdir())
    add_check("Project directory", "PASS", f"{base} ({len(files)} entries)")

    # Detect project type
    has_main_go = (base / "main.go").is_file()
    has_package_json = (base / "package.json").is_file()
    has_main_py = (base / "main.py").is_file() or (base / "app.py").is_file()
    has_index_html = (base / "index.html").is_file() or (base / "static" / "index.html").is_file()
    binary_path = base / project
    static_dir = base / "static"

    project_type = "go" if has_main_go else "python" if has_main_py else "node" if has_package_json else "static"
    add_check("Project type", "PASS", project_type)

    # Kill old screen session
    screen_name = project.replace(".", "_").replace("/", "_")
    subprocess.run(["screen", "-S", screen_name, "-X", "quit"], capture_output=True, timeout=5)
    await asyncio.sleep(0.5)

    port = 8080
    for i in range(8080, 8100):
        if i not in [info["port"] for info in APP_REGISTRY.values()]:
            port = i
            break
    add_check("Port selected", "PASS", str(port))

    build_logs = ""
    cmd = ""
    cwd = str(base)

    if has_main_go and binary_path.exists():
        cmd = f"cd {cwd} && ./{project}"
        add_check("Build", "PASS", "binary already exists, skipping build")
    elif has_main_go:
        ret = subprocess.run(
            ["go", "build", "-o", project, "."],
            capture_output=True, text=True, timeout=60, cwd=cwd,
        )
        build_logs = f"exit_code={ret.returncode}\nstdout: {ret.stdout[:500]}\nstderr: {ret.stderr[:500]}"
        if ret.returncode != 0:
            add_check("Build", "FAIL", f"Go build failed (code {ret.returncode}): {ret.stderr[:300]}")
            report["build_logs"] = build_logs
            report["verdict"] = "FAIL"
            return report
        cmd = f"cd {cwd} && ./{project}"
        add_check("Build", "PASS", "Go build succeeded")
    elif has_main_py:
        py_file = "main.py" if (base / "main.py").is_file() else "app.py"
        cmd = f"cd {cwd} && python3 {py_file}"
        add_check("Build", "PASS", "Python — no build required")
    elif has_package_json:
        cmd = f"cd {cwd} && node server.js"
        add_check("Build", "PASS", "Node.js — no build required")
    elif has_index_html:
        static_path = static_dir if static_dir.is_dir() else base
        cmd = f"cd {cwd} && python3 -m http.server {port} --directory {static_path}"
        add_check("Build", "PASS", "Static — no build required")
    else:
        add_check("Build", "FAIL", "Cannot determine how to run this project")
        report["verdict"] = "FAIL"
        return report

    report["run_command"] = cmd
    report["build_logs"] = build_logs

    # Start in screen
    try:
        r = subprocess.run(
            ["screen", "-dmS", screen_name, "bash", "-c", cmd],
            capture_output=True, timeout=10,
        )
        if r.returncode != 0:
            add_check("Screen session start", "FAIL",
                      f"exit_code={r.returncode}, stderr: {r.stderr.decode(errors='replace')[:200]}")
            report["verdict"] = "FAIL"
            return report
    except Exception as e:
        add_check("Screen session start", "FAIL", f"exception: {e}")
        report["verdict"] = "FAIL"
        return report
    add_check("Screen session start", "PASS", "screen -dmS succeeded")

    await asyncio.sleep(2)

    # Verify process alive
    screen_check = subprocess.run(["screen", "-ls", screen_name],
                                  capture_output=True, text=True, timeout=5)
    pid = None
    if screen_check.returncode == 0 and screen_name in screen_check.stdout:
        pid_match = re.search(r'(\d+)\.' + re.escape(screen_name), screen_check.stdout)
        pid = pid_match.group(1) if pid_match else "unknown"
        add_check("Process alive", "PASS", f"screen PID {pid}")
    else:
        pg = subprocess.run(["pgrep", "-f", screen_name], capture_output=True, text=True, timeout=5)
        if pg.stdout.strip():
            pid = pg.stdout.strip().split("\n")[0]
            add_check("Process alive (pgrep)", "PASS", f"PID {pid}")
        else:
            add_check("Process alive", "FAIL", f"screen session '{screen_name}' not found")
            report["verdict"] = "FAIL"
            return report
    report["pid"] = pid

    # Detect actual listening port
    actual_port = port
    ports_found = []
    for try_port in range(8080, 8100):
        chk = subprocess.run(
            ["curl", "-sS", "-o", "/dev/null", "-w", "%{http_code}", f"http://127.0.0.1:{try_port}/"],
            capture_output=True, text=True, timeout=3,
        )
        code = chk.stdout.strip()
        if code in ("200", "301", "302", "308"):
            ports_found.append(try_port)
            if actual_port == port:
                actual_port = try_port
                break

    if not ports_found:
        add_check("Port detection", "FAIL", "no listening port found in range 8080-8099")
        report["verdict"] = "FAIL"
        return report

    ss_check = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True, timeout=5)
    port_confirmed = f":{actual_port}" in ss_check.stdout
    add_check("Port listening", "PASS" if port_confirmed else "WARN",
              f"port {actual_port} detected via HTTP, ss {'agrees' if port_confirmed else 'disagrees'}")

    # HTTP health check
    health = _http_health_check(actual_port, timeout_s=5)
    if health:
        http_ok = health["http_code"] in ("200", "301", "302", "308")
        add_check("HTTP health check", "PASS" if http_ok else "FAIL",
                  f"HTTP {health['http_code']}, body: {health['body'][:200]}")
    else:
        add_check("HTTP health check", "FAIL", "no response from port")
        report["verdict"] = "FAIL"
        return report

    info = {
        "name": project, "port": actual_port, "directory": project,
        "status": "running", "url": f"/apps/{project}/",
        "type": project_type, "pid": pid,
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }
    APP_REGISTRY[project] = info
    add_check("App registry", "PASS", f"registered as '/apps/{project}/'")

    # Verify proxy URL — real check via a worker thread so the event loop stays
    # free to serve this server's own /apps/ route (a sync self-curl deadlocks).
    proxy = await asyncio.to_thread(_verify_proxy_url, f"/apps/{project}/")
    add_check("Proxy URL verification", proxy["status"],
              f"GET {proxy['proxy_url']} -> HTTP {proxy.get('http_code', '?')}")
    if proxy["status"] != "PASS":
        report["verdict"] = "FAIL"
        report["reason"] = (
            f"proxy route /apps/{project}/ did not respond "
            f"(HTTP {proxy.get('http_code', '?')})"
        )
        return {"ok": False, "report": report}

    fails = [c for c in report["checks"] if c["status"] == "FAIL"]
    report["verdict"] = "PASS" if not fails else "FAIL"
    report["preview_url"] = f"https://jaxvora.vercel.app/apps/{project}/"

    return {"ok": report["verdict"] == "PASS", "report": report}


@app.post("/run/{name}")
async def run_project(name: str):
    result = await _run_project(name)
    if result.get("report"):
        report = result["report"]
        status = 200 if report.get("verdict") == "PASS" else 400
        return JSONResponse(content=report, status_code=status)
    return JSONResponse(content=result, status_code=400)


@app.post("/team")
async def team_run(req: TeamRunRequest):
    """Direct, reliable parallel-team run for ANY role (bypasses the Chief loop):
    splits the task into slices, runs workers of `role` in parallel, then a
    Head/Lead of that role reviews + merges. Returns the structured report."""
    task = (req.task or "").strip()
    if not task:
        return JSONResponse(status_code=400, content={"ok": False, "error": "task is required", "verdict": "FAIL"})
    report = await _run_parallel_team((req.role or "Software Engineer").strip(), task,
                                      req.parts, (req.project or "").strip())
    status = 200 if report.get("verdict") == "PASS" else 400
    return JSONResponse(content=report, status_code=status)


# ── Async chat jobs ───────────────────────────────────────────────────────────
# A long multi-agent run can exceed an upstream proxy's response timeout (e.g. the
# Vercel rewrite), which would return a non-JSON error page. /chat/start launches
# the run in the background and returns immediately; the client polls /chat/poll,
# which also reports the agents currently working so the flow graph stays live.
CHAT_JOBS: Dict[str, Dict[str, Any]] = {}
CHAT_JOBS_MAX = 200


def _prune_chat_jobs():
    if len(CHAT_JOBS) <= CHAT_JOBS_MAX:
        return
    for jid, _ in sorted(CHAT_JOBS.items(), key=lambda kv: kv[1].get("created", 0))[: len(CHAT_JOBS) - CHAT_JOBS_MAX]:
        CHAT_JOBS.pop(jid, None)


async def _run_chat_job(job_id: str, message: str, admin_token: Optional[str]):
    job = CHAT_JOBS.get(job_id)
    if job is None:
        return
    try:
        # Check if cancelled before starting
        if job.get("cancel"):
            job["status"] = "cancelled"
            job["response"] = "Cancelled by user."
            return
        result = await orchestrator.process(message, admin_token=admin_token,
                                            cancel_flag=lambda: job.get("cancel", False))
        if isinstance(result, dict):
            job["result"] = result
            job["response"] = result.get("response", "")
            job["agents"] = result.get("agents", []) or []
        else:
            job["result"] = {"response": str(result)}
            job["response"] = str(result)
        job["status"] = "cancelled" if job.get("cancel") else "done"
    except asyncio.CancelledError:
        job["status"] = "cancelled"
        job["response"] = "Cancelled by user."
    except Exception as e:
        logger.error(f"Chat job {job_id} failed: {e}", exc_info=True)
        job["status"] = "error"
        job["error"] = "The request could not be completed right now. Please try again."
    finally:
        job["finished"] = datetime.now(timezone.utc).timestamp()


@app.post("/chat/start")
async def chat_start(req: ChatRequest):
    _prune_chat_jobs()
    job_id = str(uuid.uuid4())
    CHAT_JOBS[job_id] = {
        "status": "running", "response": None, "error": None, "agents": [],
        "created": datetime.now(timezone.utc).timestamp(),
    }
    asyncio.create_task(_run_chat_job(job_id, req.message, req.admin_token))
    return {"job_id": job_id, "status": "running"}


@app.get("/chat/poll/{job_id}")
async def chat_poll(job_id: str):
    job = CHAT_JOBS.get(job_id)
    if job is None:
        return JSONResponse(status_code=404, content={"status": "not_found", "error": "Unknown or expired job."})
    working = [
        {"name": a.name, "division": a.division, "status": a._status, "task": a._current_task}
        for a in AGENT_REGISTRY.values() if a._status in ("running", "error")
    ]
    out: Dict[str, Any] = {"status": job["status"], "working": working}
    if job["status"] in ("done", "cancelled"):
        out["response"] = job.get("response", "")
        out["agents"] = job.get("agents", [])
        out["result"] = job.get("result", {})
        out["steps"] = job.get("result", {}).get("steps", [])
    elif job["status"] == "error":
        out["error"] = job.get("error", "Request failed.")
    return out


@app.post("/chat/stop/{job_id}")
async def chat_stop(job_id: str):
    job = CHAT_JOBS.get(job_id)
    if job is None:
        return {"ok": False, "error": "Job not found"}
    if job["status"] != "running":
        return {"ok": False, "error": f"Job is {job['status']}, not running"}
    job["cancel"] = True
    job["status"] = "cancelled"
    job["response"] = "Cancelled by user."
    # Kill all running agents
    for a in AGENT_REGISTRY.values():
        if a._status == "running":
            a._status = "idle"
            a._current_task = ""
    return {"ok": True, "stopped": job_id}


@app.get("/agents")
async def list_agents():
    return [a.status_dict() for a in AGENT_REGISTRY.values()]


@app.get("/organization")
async def get_organization():
    return organization_snapshot()


@app.get("/tasks")
async def list_tasks(status: Optional[str] = None, agent: Optional[str] = None, limit: int = 50):
    if db_pool is None:
        return []
    if status and agent:
        rows = await db_fetch("SELECT * FROM tasks WHERE status=$1 AND agent_name=$2 ORDER BY created_at DESC LIMIT $3", status, agent, limit)
    elif status:
        rows = await db_fetch("SELECT * FROM tasks WHERE status=$1 ORDER BY created_at DESC LIMIT $2", status, limit)
    elif agent:
        rows = await db_fetch("SELECT * FROM tasks WHERE agent_name=$1 ORDER BY created_at DESC LIMIT $2", agent, limit)
    else:
        rows = await db_fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT $1", limit)
    return [dict(r) for r in rows]


def _rowcount(tag: Optional[str]) -> int:
    """Parse the affected-row count from an asyncpg command tag (e.g. 'DELETE 5')."""
    try:
        return int(str(tag).strip().split()[-1])
    except Exception:
        return 0


def _valid_task_id(task_id: str) -> bool:
    """tasks.id is a UUID; reject malformed ids up front so we return a clean 404
    instead of a 500 from asyncpg."""
    try:
        uuid.UUID(str(task_id))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _signal_runtime_stop() -> Dict[str, int]:
    """Best-effort: flip running agents to idle and cancel active chat jobs so an
    in-flight TAOR loop actually winds down (the loop checks its cancel_flag)."""
    agents = 0
    for a in AGENT_REGISTRY.values():
        if a._status == "running":
            a._status = "idle"
            a._current_task = ""
            agents += 1
    jobs = 0
    for job in CHAT_JOBS.values():
        if job.get("status") == "running":
            job["cancel"] = True
            job["status"] = "cancelled"
            job["response"] = "Cancelled by user."
            jobs += 1
    return {"agents_signaled": agents, "chat_jobs_signaled": jobs}


@app.post("/tasks/{task_id}/stop")
async def stop_task(task_id: str):
    """Stop a single running/pending task: mark it 'cancelled' and signal the
    runtime to wind down. No fake success — reports what actually changed."""
    if db_pool is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "no database"})
    if not _valid_task_id(task_id):
        return JSONResponse(status_code=404, content={"ok": False, "error": "task not found"})
    row = await db_fetchrow("SELECT id, status FROM tasks WHERE id=$1", task_id)
    if row is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "task not found"})
    if row["status"] not in ("running", "pending"):
        return {"ok": False, "error": f"task is '{row['status']}', not running/pending", "status": row["status"]}
    await db_execute(
        "UPDATE tasks SET status='cancelled', completed_at=NOW(), "
        "output=COALESCE(output, '') || '\n[stopped by user]' WHERE id=$1", task_id)
    runtime = _signal_runtime_stop()
    return {"ok": True, "task_id": task_id, "new_status": "cancelled", **runtime}


@app.post("/tasks/stop-all")
async def stop_all_tasks():
    """Cancel ALL running + pending tasks and signal the runtime to stop."""
    if db_pool is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "no database"})
    tag = await db_execute(
        "UPDATE tasks SET status='cancelled', completed_at=NOW() WHERE status IN ('running','pending')")
    runtime = _signal_runtime_stop()
    return {"ok": True, "cancelled": _rowcount(tag), **runtime}


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str):
    """Delete a single task row (any status)."""
    if db_pool is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "no database"})
    if not _valid_task_id(task_id):
        return JSONResponse(status_code=404, content={"ok": False, "error": "task not found"})
    tag = await db_execute("DELETE FROM tasks WHERE id=$1", task_id)
    return {"ok": True, "task_id": task_id, "deleted": _rowcount(tag)}


@app.post("/tasks/clear")
async def clear_tasks(req: TaskClearRequest):
    """Clear tasks by scope. Running/pending scopes (and 'all') also signal the
    runtime to stop in-flight work before deleting the rows."""
    if db_pool is None:
        return JSONResponse(status_code=503, content={"ok": False, "error": "no database"})
    scope = (req.scope or "finished").lower().strip()
    runtime: Dict[str, int] = {}
    if scope == "all":
        runtime = _signal_runtime_stop()
        tag = await db_execute("DELETE FROM tasks")
    elif scope == "finished":
        tag = await db_execute("DELETE FROM tasks WHERE status IN ('completed','failed','cancelled')")
    elif scope in ("running", "pending"):
        runtime = _signal_runtime_stop()
        tag = await db_execute("DELETE FROM tasks WHERE status=$1", scope)
    elif scope in ("completed", "failed", "cancelled"):
        tag = await db_execute("DELETE FROM tasks WHERE status=$1", scope)
    else:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"unknown scope '{scope}'"})
    return {"ok": True, "scope": scope, "deleted": _rowcount(tag), **runtime}


@app.get("/projects")
async def list_projects():
    if db_pool is None:
        return []
    rows = await db_fetch("SELECT * FROM projects ORDER BY created_at DESC")
    return [dict(r) for r in rows]


@app.post("/projects")
async def create_project(req: ProjectCreate):
    row = await db_fetchrow(
        "INSERT INTO projects (name, repo_url, metadata) VALUES ($1, $2, $3) RETURNING *",
        req.name, req.repo_url, json.dumps(req.metadata) if req.metadata else None
    )
    return dict(row)


@app.get("/logs")
async def get_logs(level: Optional[str] = None, limit: int = 100, offset: int = 0):
    if db_pool is None:
        return []
    if level:
        rows = await db_fetch("SELECT * FROM logs WHERE level=$1 ORDER BY timestamp DESC LIMIT $2 OFFSET $3", level, limit, offset)
    else:
        rows = await db_fetch("SELECT * FROM logs ORDER BY timestamp DESC LIMIT $1 OFFSET $2", limit, offset)
    return [dict(r) for r in rows]


@app.get("/analytics")
async def get_analytics():
    total_n = completed_n = failed_n = today_n = 0
    by_agent_rows: List[Dict[str, Any]] = []
    audit_total = audit_rejected = 0
    db_available = db_pool is not None

    if db_available:
        try:
            total = await db_fetchrow("SELECT COUNT(*) as n FROM tasks")
            success = await db_fetchrow("SELECT COUNT(*) as n FROM tasks WHERE status='completed'")
            failed = await db_fetchrow("SELECT COUNT(*) as n FROM tasks WHERE status='failed'")
            today = await db_fetchrow(
                "SELECT COUNT(*) as n FROM tasks WHERE created_at > NOW() - INTERVAL '24 hours'"
            )
            by_agent = await db_fetch(
                "SELECT agent_name, COUNT(*) as calls FROM tasks GROUP BY agent_name ORDER BY calls DESC LIMIT 10"
            )
            audit_stats = await db_fetchrow(
                "SELECT COUNT(*) as total, COUNT(*) FILTER (WHERE approved = false) as rejected FROM audit"
            )

            total_n = int(total["n"] or 0)
            completed_n = int(success["n"] or 0)
            failed_n = int(failed["n"] or 0)
            today_n = int(today["n"] or 0)
            by_agent_rows = [dict(r) for r in by_agent]
            if audit_stats:
                audit_total = int(audit_stats["total"] or 0)
                audit_rejected = int(audit_stats["rejected"] or 0)
        except Exception as e:
            logger.warning(f"Analytics DB fallback: {e}")
            db_available = False

    agents = [a.status_dict() for a in AGENT_REGISTRY.values()]
    total_agents = len(agents)
    active_agents = sum(1 for a in agents if a.get("status") == "running")
    error_agents = sum(1 for a in agents if a.get("status") == "error")
    security_agents = [a for a in agents if a.get("division") == "Security"]
    security_errors = sum(1 for a in security_agents if a.get("status") == "error")

    success_rate = round(completed_n / max(total_n, 1) * 100, 1) if total_n else 100.0
    agent_coverage = min(100.0, total_agents / 22 * 100) if total_agents else 0.0
    runtime_health = 100.0 - (error_agents / max(total_agents, 1) * 30)
    db_health = 100.0 if db_available else 75.0
    project_health = round(
        max(0.0, min(100.0, agent_coverage * 0.35 + runtime_health * 0.30 + success_rate * 0.25 + db_health * 0.10))
    )

    security_agent_health = 100.0
    if security_agents:
        security_agent_health = 100.0 - (security_errors / len(security_agents) * 35)
    audit_health = 100.0 - (audit_rejected / max(audit_total, 1) * 30) if audit_total else 100.0
    security_score = round(max(0.0, min(100.0, security_agent_health * 0.55 + audit_health * 0.30 + runtime_health * 0.15)))

    return {
        "total_tasks": total_n,
        "completed": completed_n,
        "failed": failed_n,
        "tasks_today": today_n,
        "success_rate": success_rate,
        "by_agent": by_agent_rows,
        "project_health": project_health,
        "security_score": security_score,
        "active_agents": active_agents,
        "total_agents": total_agents,
        "error_agents": error_agents,
        "db_available": db_available,
    }


@app.post("/doctor/run")
async def doctor_run(max_iterations: int = 2):
    return await run_jaxvora_doctor(max_iterations=max_iterations)


@app.get("/memory/search")
async def memory_search(q: str = Query(...), collection: Optional[str] = None, limit: int = 5):
    results = await memory.search(q, collection, limit)
    return results


class RAGIngestRequest(BaseModel):
    text: str
    source: str = ""
    metadata: Optional[Dict] = None


class RAGSearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post("/rag/ingest")
async def rag_ingest(req: RAGIngestRequest):
    """Ingest text into RAG vector store."""
    count = await rag_engine.ingest(req.text, source=req.source, metadata=req.metadata)
    return {"ok": True, "chunks": count}


@app.post("/upload/rag")
async def upload_to_rag(file: UploadFile = File(...)):
    """Upload a file directly to RAG (chunk, embed, store)."""
    content = await file.read()
    extracted = _extract_upload_text(content, file.filename or "attachment", file.content_type or "")
    if not extracted["ok"]:
        return {"ok": False, "error": extracted["error"]}
    count = await rag_engine.ingest(extracted["content"], source=file.filename or "attachment")
    return {"ok": True, "chunks": count, "text_length": len(extracted["content"])}


@app.post("/rag/search")
async def rag_search(req: RAGSearchRequest):
    """Search RAG vector store with hybrid search."""
    results = await rag_engine.search(req.query, top_k=req.top_k)
    return {"ok": True, "results": results}


@app.get("/rag/status")
async def rag_status():
    """RAG engine status."""
    total = 0
    total_chunks = 0
    sources = []
    try:
        row = await db_fetchrow("SELECT COUNT(*) as n FROM rag_documents")
        total_chunks = int(row["n"]) if row else 0
        src_rows = await db_fetch(
            "SELECT source, COUNT(*) as chunks, MAX(created_at) as last_added FROM rag_documents GROUP BY source ORDER BY last_added DESC"
        )
        sources = [{"source": r["source"], "chunks": int(r["chunks"]), "last_added": str(r.get("last_added", ""))} for r in src_rows]
        row2 = await db_fetchrow("SELECT COUNT(DISTINCT source) as n FROM rag_documents")
        total = int(row2["n"]) if row2 else 0
    except Exception:
        pass
    return {
        "ok": True,
        "documents": total,
        "total_chunks": total_chunks,
        "sources": sources,
        "index_loaded": rag_engine._index_loaded,
        "index_size": len(rag_engine._index),
    }


@app.delete("/rag/documents/{doc_id}")
async def rag_delete_document(doc_id: str):
    """Delete a single RAG document chunk by id."""
    try:
        await db_execute("DELETE FROM rag_documents WHERE id = $1::uuid", doc_id)
        rag_engine._index.pop(doc_id, None)
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.delete("/rag/source")
async def rag_delete_source(source: str = Query(...)):
    """Delete an entire knowledge-base file (all chunks sharing a source)."""
    if db_pool is None:
        return {"ok": False, "error": "Database unavailable."}
    try:
        rows = await db_fetch("SELECT id FROM rag_documents WHERE source = $1", source)
        await db_execute("DELETE FROM rag_documents WHERE source = $1", source)
        for r in rows:
            rag_engine._index.pop(str(r["id"]), None)
        return {"ok": True, "deleted": len(rows), "source": source}
    except Exception as e:
        logger.error(f"/rag/source delete failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


# ── Social media connectors ───────────────────────────────────────────────────
@app.get("/social/connectors")
async def social_connectors():
    return {"ok": True, "platforms": social_public_view(await social_load())}


@app.post("/social/connect")
async def social_connect(req: dict):
    platform = (req.get("platform") or "").lower().strip()
    if platform not in SOCIAL_PLATFORMS:
        return {"ok": False, "error": "unknown platform"}
    data = await social_load()
    conn = data.get(platform, {}) or {}
    if "token" in req:
        conn["token"] = (req.get("token") or "").strip()
    if "auto_post" in req:
        conn["auto_post"] = bool(req.get("auto_post"))
    if isinstance(req.get("meta"), dict):
        conn["meta"] = {**(conn.get("meta") or {}), **req["meta"]}
    data[platform] = conn
    await social_save(data)
    return {"ok": True, "platforms": social_public_view(data)}


@app.post("/social/disconnect")
async def social_disconnect(req: dict):
    platform = (req.get("platform") or "").lower().strip()
    data = await social_load()
    if platform in data:
        data[platform] = {"auto_post": False}
        await social_save(data)
    return {"ok": True, "platforms": social_public_view(data)}


@app.post("/social/auto")
async def social_auto(req: dict):
    platform = (req.get("platform") or "").lower().strip()
    data = await social_load()
    if platform in SOCIAL_PLATFORMS:
        conn = data.get(platform, {}) or {}
        conn["auto_post"] = bool(req.get("auto_post"))
        data[platform] = conn
        await social_save(data)
    return {"ok": True, "platforms": social_public_view(data)}


@app.post("/social/post")
async def social_post_endpoint(req: dict):
    """Manual post (explicit user action) — publishes regardless of auto_post."""
    platform = (req.get("platform") or "").lower().strip()
    text = req.get("text") or ""
    if platform not in SOCIAL_PLATFORMS:
        return {"ok": False, "error": "unknown platform"}
    if not text:
        return {"ok": False, "error": "empty text"}
    data = await social_load()
    conn = data.get(platform, {}) or {}
    if not conn.get("token"):
        return {"ok": False, "error": f"{SOCIAL_LABELS[platform]} is not connected."}
    return await social_publish(platform, conn, text, req.get("link", ""), req.get("image_url", ""))


@app.post("/admin/reset")
async def admin_reset(req: dict):
    """Wipe Jaxvora's execution history so it starts fresh. Clears the task queue,
    logs, audit, agent history and all jaxvora_* run tables. Preserves the
    knowledge base (rag_documents), connectors and app settings. Guarded: requires
    {"confirm":"RESET"} and the admin token; not exposed via the Vercel proxy."""
    if (req.get("confirm") or "") != "RESET":
        return {"ok": False, "error": "Pass {\"confirm\":\"RESET\"} to wipe history."}
    expected = GMAIL_AUTOMATION_API_TOKEN
    if expected and (req.get("admin_token") or "") != expected:
        return {"ok": False, "error": "Valid admin_token required."}
    if db_pool is None:
        return {"ok": False, "error": "Database unavailable."}
    tables = [
        "jaxvora_ssh_audit", "jaxvora_operation_log", "jaxvora_subtask_log",
        "jaxvora_sessions", "agent_history", "audit", "logs", "tasks",
    ]
    cleared = {}
    for t in tables:
        try:
            await db_execute(f"TRUNCATE TABLE {t} RESTART IDENTITY CASCADE")
            cleared[t] = "cleared"
        except Exception as e:
            cleared[t] = f"skipped: {e}"
    for a in AGENT_REGISTRY.values():
        a._status = "idle"
        a._current_task = ""
    try:
        CHAT_JOBS.clear()
    except Exception:
        pass
    return {"ok": True, "cleared": cleared, "note": "Knowledge base, connectors and settings were preserved."}


@app.get("/web/search")
async def web_search(q: str = Query(...), max_results: int = 5):
    """Search the web using DuckDuckGo."""
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            r = await client.post(
                "https://lite.duckduckgo.com/lite/",
                data={"q": q},
                headers={"User-Agent": "Mozilla/5.0 (compatible; Jaxvora/1.0)"},
            )
            results = []
            lines = r.text.split("\n")
            current_title, current_snippet, current_url = "", "", ""
            in_result = False
            capturing_snippet = False
            for line in lines:
                if '"result-link"' in line or "'result-link'" in line or '"result__a"' in line or "'result__a'" in line:
                    if current_title:
                        results.append({"title": current_title.strip(), "snippet": html.unescape(re.sub(r'<[^>]+>', '', current_snippet)).strip(), "url": current_url.strip()})
                    current_title = current_snippet = current_url = ""
                    capturing_snippet = False
                    href_match = re.search(r'href="([^"]+)"', line)
                    if not href_match:
                        href_match = re.search(r"href='([^']+)'", line)
                    if href_match:
                        current_url = html.unescape(href_match.group(1))
                    title_match = re.search(r'>([^<]+)<', line)
                    if title_match:
                        current_title = html.unescape(title_match.group(1))
                    in_result = True
                elif in_result and ('class="result-snippet"' in line or "class='result-snippet'" in line or 'class="result__snippet"' in line or "class='result__snippet'" in line):
                    capturing_snippet = True
                    after_class = line.split(">", 1)[1] if ">" in line else ""
                    current_snippet += after_class + " "
                elif in_result and capturing_snippet:
                    if "</td>" in line or "</TD>" in line:
                        capturing_snippet = False
                    elif "<tr" not in line and line.strip() and not line.strip().startswith("<"):
                        current_snippet += line.strip() + " "
                elif in_result and line.strip() in ("</div>", "</div"):
                    if current_title:
                        results.append({"title": current_title.strip(), "snippet": html.unescape(re.sub(r'<[^>]+>', '', current_snippet)).strip(), "url": current_url.strip()})
                    current_title = current_snippet = current_url = ""
                    in_result = False
                    capturing_snippet = False
                if len(results) >= max_results:
                    break
            if current_title:
                results.append({"title": current_title.strip(), "snippet": html.unescape(re.sub(r'<[^>]+>', '', current_snippet)).strip(), "url": current_url.strip()})
            return {"ok": True, "results": results[:max_results], "query": q}
    except Exception as e:
        return {"ok": False, "error": str(e), "results": []}


@app.post("/security/scan")
async def security_scan(req: SecurityScanRequest):
    agent = AGENT_REGISTRY.get("Cybersecurity")
    if agent:
        result = await agent.run(f"Security scan requested:\n\n{req.content}")
        return {"findings": result.output, "task_id": result.task_id}
    return {"findings": "Cybersecurity agent unavailable"}


@app.get("/audit")
async def get_audit(limit: int = 50):
    if db_pool is None:
        return []
    rows = await db_fetch("SELECT * FROM audit ORDER BY timestamp DESC LIMIT $1", limit)
    return [dict(r) for r in rows]


@app.get("/tools")
async def list_tools():
    return tool_registry.list_tools()


@app.get("/gmail/status")
async def gmail_status():
    status = gmail_automation_status()
    return {"ok": status["configured"], **status}


@app.post("/gmail/action")
async def gmail_action(req: GmailActionRequest, x_jaxvora_admin_token: Optional[str] = Header(None)):
    payload = req.model_dump()
    action = str(payload.get("action", "status")).strip().lower()
    status = gmail_automation_status()
    if status["configured"] and action != "status":
        if not GMAIL_AUTOMATION_API_TOKEN:
            return {
                "ok": False,
                "error": "Gmail automation API is locked because GMAIL_AUTOMATION_API_TOKEN is not configured",
                "policy": status["policy"],
            }
        if not _gmail_action_authorized(x_jaxvora_admin_token):
            return {
                "ok": False,
                "error": "Gmail automation requires X-Jaxvora-Admin-Token",
                "policy": status["policy"],
            }
    return await run_gmail_automation(payload)


def _clean_upload_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"/?envel\S{0,12}pe(?=[A-Za-z0-9._%+-]+@)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?:\u2642|\u00b6|\?|/)?phone(?=\+?\d)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/?ap-\S{0,16}arker-alt", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/?gl\S{0,8}be(?=[A-Za-z0-9.-]+\.)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/github(?=github\.com)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/linkedin(?=linkedin\.com)", "", text, flags=re.IGNORECASE)
    text = text.replace("\u2642", " ").replace("\u00b6", " ").replace("\u2322", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _strip_binary_noise(text: str) -> str:
    """Remove non-printable binary characters but keep newlines, tabs, and readable text."""
    cleaned = []
    for ch in text:
        code = ord(ch)
        if code == 0:
            continue
        if code == 10 or code == 13 or code == 9:
            cleaned.append(ch)
        elif 32 <= code <= 126:
            cleaned.append(ch)
        elif code >= 160:
            cleaned.append(ch)
    return "".join(cleaned)


def _try_extract_archive(content: bytes, filename: str) -> Optional[Dict[str, Any]]:
    """Try to extract readable text from archive/bundle formats (ZIP, TAR, GZ)."""
    import io
    extracted_parts = []

    # Try ZIP
    try:
        import zipfile
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            names = zf.namelist()
            for name in names:
                try:
                    data = zf.read(name)
                    text = data.decode("utf-8", errors="replace")
                    text = _strip_binary_noise(text)
                    text = text.strip()
                    if text:
                        extracted_parts.append(f"### {name}\n{text}")
                except Exception:
                    pass
            if extracted_parts:
                return {"ok": True, "error": "", "content": "\n\n".join(extracted_parts), "pages": 0, "truncated": False}
    except Exception:
        pass

    # Try TAR
    try:
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(content), mode="r:*") as tf:
            for member in tf.getmembers():
                if member.isfile():
                    try:
                        data = tf.extractfile(member)
                        if data:
                            text = data.read().decode("utf-8", errors="replace")
                            text = _strip_binary_noise(text)
                            text = text.strip()
                            if text:
                                extracted_parts.append(f"### {member.name}\n{text}")
                    except Exception:
                        pass
            if extracted_parts:
                return {"ok": True, "error": "", "content": "\n\n".join(extracted_parts), "pages": 0, "truncated": False}
    except Exception:
        pass

    return None


def _looks_like_pdf_bytes(content: bytes, filename: str, content_type: str) -> bool:
    return (
        content.startswith(b"%PDF")
        or filename.lower().endswith(".pdf")
        or content_type.lower().startswith("application/pdf")
    )


def _extract_pdf_upload_text(content: bytes) -> Dict[str, Any]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        return {
            "ok": False,
            "error": f"PDF parser is not installed: {type(exc).__name__}",
            "content": "",
            "pages": 0,
            "truncated": False,
        }

    try:
        reader = PdfReader(BytesIO(content))
        page_texts: List[str] = []
        for idx, page in enumerate(reader.pages):
            try:
                extracted = page.extract_text() or ""
            except Exception as page_exc:
                extracted = f"[Page {idx + 1} text extraction failed: {type(page_exc).__name__}]"
            extracted = _clean_upload_text(extracted)
            if extracted:
                page_texts.append(f"--- Page {idx + 1} ---\n{extracted}")
        text = _clean_upload_text("\n\n".join(page_texts))
        truncated = len(text) > MAX_UPLOAD_TEXT_CHARS
        if truncated:
            text = text[:MAX_UPLOAD_TEXT_CHARS].rstrip()
        if not text:
            return {
                "ok": False,
                "error": "No readable text could be extracted from this PDF. It may be scanned or image-only.",
                "content": "",
                "pages": len(reader.pages),
                "truncated": False,
            }
        return {
            "ok": True,
            "error": "",
            "content": text,
            "pages": len(reader.pages),
            "truncated": truncated,
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"PDF extraction failed: {type(exc).__name__}: {exc}",
            "content": "",
            "pages": 0,
            "truncated": False,
        }


def _extract_upload_text(content: bytes, filename: str, content_type: str) -> Dict[str, Any]:
    if len(content) > MAX_UPLOAD_BYTES:
        return {
            "ok": False,
            "error": f"File is too large. Maximum supported size is {MAX_UPLOAD_BYTES // 1024 // 1024} MB.",
            "content": "",
            "pages": 0,
            "truncated": False,
        }

    if _looks_like_pdf_bytes(content, filename, content_type):
        return _extract_pdf_upload_text(content)

    archive_result = _try_extract_archive(content, filename)
    if archive_result:
        text = _clean_upload_text(archive_result["content"])
        truncated = len(text) > MAX_UPLOAD_TEXT_CHARS
        if truncated:
            text = text[:MAX_UPLOAD_TEXT_CHARS].rstrip()
        return {"ok": True, "error": "", "content": text, "pages": 0, "truncated": truncated}

    text_types = (
        "text/",
        "application/json",
        "application/xml",
        "application/x-yaml",
        "application/yaml",
        "application/javascript",
    )
    text_extensions = (
        ".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml",
        ".js", ".ts", ".py", ".html", ".css",
        ".skill", ".ps1", ".sh", ".bat", ".cmd", ".sql", ".log",
        ".env", ".toml", ".ini", ".cfg", ".conf",
        ".svg", ".gradle", ".kt", ".swift", ".go", ".rb", ".php",
        ".pl", ".r", ".lua", ".proto", ".graphql", ".prisma",
        ".tf", ".hcl", ".vue", ".svelte", ".astro", ".tsx", ".jsx",
        ".mjs", ".cjs", ".mts", ".cts", ".zig", ".nim", ".rs",
        ".dockerfile", ".makefile", ".cmake",
    )
    known_text = content_type.lower().startswith(text_types) or filename.lower().endswith(text_extensions)
    if known_text:
        text = _clean_upload_text(content.decode("utf-8", errors="replace"))
        truncated = len(text) > MAX_UPLOAD_TEXT_CHARS
        if truncated:
            text = text[:MAX_UPLOAD_TEXT_CHARS].rstrip()
        return {"ok": True, "error": "", "content": text, "pages": 0, "truncated": truncated}

    try:
        text = _clean_upload_text(content.decode("utf-8"))
        truncated = len(text) > MAX_UPLOAD_TEXT_CHARS
        if truncated:
            text = text[:MAX_UPLOAD_TEXT_CHARS].rstrip()
        return {"ok": True, "error": "", "content": text, "pages": 0, "truncated": truncated}
    except (UnicodeDecodeError, UnicodeError):
        pass

    return {
        "ok": False,
        "error": f"Could not read file: {filename}. Jaxvora supports text files (code, docs, data), PDFs, and archives (ZIP, TAR). Binary files are not supported.",
        "content": "",
        "pages": 0,
        "truncated": False,
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept file upload and return its text content for agent context."""
    content = await file.read()
    content_type = file.content_type or ""
    extracted = _extract_upload_text(content, file.filename or "attachment", content_type)
    return {
        "filename": file.filename,
        "size": len(content),
        "content_type": content_type,
        "ok": extracted["ok"],
        "error": extracted["error"],
        "content": extracted["content"],
        "text_length": len(extracted["content"]),
        "pages": extracted["pages"],
        "truncated": extracted["truncated"],
    }


class DownloadRequest(BaseModel):
    filename: str
    content: str
    content_type: Optional[str] = "text/plain"


@app.post("/download")
async def download_file(req: DownloadRequest):
    """Generate a downloadable file from LLM output. Used by agents to serve generated files."""
    import base64 as b64_mod
    safe_name = re.sub(r'[^\w.\-]', '_', req.filename) or "download"
    encoded = b64_mod.b64encode(req.content.encode("utf-8")).decode()
    return {
        "ok": True,
        "filename": safe_name,
        "content_type": req.content_type,
        "content_base64": encoded,
        "size": len(req.content),
    }


@app.get("/settings/notification-email")
async def get_notification_email():
    return {"email": NOTIFICATION_EMAIL}


@app.get("/settings/status")
async def get_settings_status():
    gmail_sender_ready = bool(GMAIL_SENDER)
    gmail_password_ready = bool(GMAIL_APP_PASSWORD)
    gmail_ready = gmail_sender_ready and gmail_password_ready
    gmail_missing = []
    if not gmail_sender_ready:
        gmail_missing.append("GMAIL_SENDER")
    if not gmail_password_ready:
        gmail_missing.append("GMAIL_APP_PASSWORD")
    gmail_api_status = gmail_automation_status()

    return {
        "keys": {
            "OPENROUTER_API_KEY": {"configured": bool(OPENROUTER_API_KEY)},
            "GROQ_API_KEY": {"configured": bool(GROQ_API_KEY)},
            "OPENCODE_ZEN_API_KEY": {"configured": bool(OPENCODE_ZEN_API_KEY), "primary": OPENCODE_ZEN_PRIMARY, "model": OPENCODE_ZEN_MODEL},
            "DATABASE_URL": {"configured": bool(DATABASE_URL), "connected": bool(db_pool)},
            "GMAIL_AUTOMATION": {
                "configured": gmail_api_status["configured"],
                "partial": bool(gmail_api_status["missing"]) and bool(GMAIL_CLIENT_ID or GMAIL_CLIENT_SECRET or GMAIL_REFRESH_TOKEN),
                "missing": gmail_api_status["missing"],
                "user": gmail_api_status["user"],
                "api_guard_configured": gmail_api_status["api_guard_configured"],
                "action_api_ready": gmail_api_status["action_api_ready"],
            },
            "EMAIL_DELIVERY": {
                "configured": gmail_ready,
                "partial": bool(gmail_sender_ready or gmail_password_ready) and not gmail_ready,
                "missing": gmail_missing,
            },
        },
        "llm_failover": llm_provider_status(),
        "email": {
            "notification_email": NOTIFICATION_EMAIL,
            "sender": GMAIL_SENDER,
            "sender_configured": gmail_sender_ready,
            "app_password_configured": gmail_password_ready,
            "delivery_configured": gmail_ready,
            "missing": gmail_missing,
        },
        "gmail_automation": gmail_api_status,
    }


@app.post("/settings/notification-email")
async def set_notification_email(req: NotificationEmailRequest):
    global NOTIFICATION_EMAIL
    NOTIFICATION_EMAIL = req.email
    if db_pool:
        try:
            await db_execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES ('notification_email', $1, NOW())
                ON CONFLICT (key) DO UPDATE SET value=$1, updated_at=NOW()
                """,
                NOTIFICATION_EMAIL,
            )
        except Exception as e:
            logger.warning(f"Failed to persist notification email: {e}")
    return {"email": NOTIFICATION_EMAIL, "status": "saved"}


@app.get("/settings/ssh")
async def get_ssh_config():
    return {
        "host": SSH_HOST,
        "port": SSH_PORT,
        "user": SSH_USER,
        "configured": bool(SSH_HOST and SSH_USER),
    }


@app.post("/settings/ssh")
async def set_ssh_config(req: SSHConfigRequest):
    global SSH_HOST, SSH_PORT, SSH_USER, SSH_PASSWORD, SSH_KEY
    SSH_HOST = req.host
    SSH_PORT = req.port or 22
    SSH_USER = req.user
    SSH_PASSWORD = req.password or ""
    SSH_KEY = req.key or ""
    return {"status": "saved", "host": SSH_HOST, "user": SSH_USER}


@app.post("/settings/ssh/test")
async def test_ssh():
    result = await SSHTool().run({"command": "echo SSH_OK && uname -a && uptime"})
    return {"result": result, "ok": "SSH_OK" in result}


@app.post("/send-email")
async def send_email_endpoint(req: SendEmailRequest):
    result = await send_gmail(req.to, req.subject, req.body)
    return {"result": result}


@app.post("/send-email/test")
async def test_email():
    to = NOTIFICATION_EMAIL
    if not to:
        return {"result": "No notification email configured", "ok": False}
    result = await send_gmail(to, "✅ Jaxvora Email Test", "Your Jaxvora email notifications are working correctly!")
    return {"result": result, "ok": "sent" in result.lower()}


# === SECTION 10: WebSocket Endpoints ==========================================

@app.websocket("/ws/agents")
async def ws_agents(ws: WebSocket):
    await ws.accept()
    ws_manager.agents.add(ws)
    try:
        # Send current state immediately
        await ws.send_text(json.dumps({
            "type": "snapshot",
            "agents": [a.status_dict() for a in AGENT_REGISTRY.values()]
        }, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.agents.discard(ws)


@app.websocket("/ws/tasks")
async def ws_tasks(ws: WebSocket):
    await ws.accept()
    ws_manager.tasks_ws.add(ws)
    try:
        if db_pool is None:
            await ws.send_text(json.dumps({"type": "error", "message": "Database unavailable"}))
            ws_manager.tasks_ws.discard(ws)
            return
        rows = await db_fetch("SELECT * FROM tasks ORDER BY created_at DESC LIMIT 20")
        await ws.send_text(json.dumps({"type": "snapshot", "tasks": [dict(r) for r in rows]}, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.tasks_ws.discard(ws)


@app.websocket("/ws/logs")
async def ws_logs(ws: WebSocket):
    await ws.accept()
    ws_manager.logs_ws.add(ws)
    try:
        if db_pool is None:
            await ws.send_text(json.dumps({"type": "error", "message": "Database unavailable"}))
            ws_manager.logs_ws.discard(ws)
            return
        rows = await db_fetch("SELECT * FROM logs ORDER BY timestamp DESC LIMIT 50")
        for r in reversed(rows):
            await ws.send_text(json.dumps({"type": "log", "level": r["level"], "message": r["message"], "ts": r["timestamp"].isoformat()}, default=str))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.logs_ws.discard(ws)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket):
    await ws.accept()
    ws_manager.chat_ws.add(ws)
    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                msg = payload.get("message", "")
                admin_token = payload.get("admin_token")
            except json.JSONDecodeError:
                await ws.send_json({"type": "error", "message": "Invalid JSON payload."})
                continue
            if not msg:
                await ws.send_json({"type": "error", "message": "Message is required."})
                continue
            await ws.send_json({"type": "thinking", "message": "Orchestrator is planning..."})

            async def stream(event):
                await ws.send_json(event)

            result = await orchestrator.process(msg, stream_fn=stream, admin_token=admin_token)
            await ws.send_json({"type": "response", "message": result["response"], "plan": result["plan"], "agents": result["agents"]})
    except WebSocketDisconnect:
        ws_manager.chat_ws.discard(ws)
    except Exception as e:
        logger.warning(f"WebSocket chat error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        ws_manager.chat_ws.discard(ws)


# === SECTION 11: Startup / Shutdown Events ====================================

auto_healer: Optional[AutoHealDaemon] = None

async def startup():
    global db_pool, NOTIFICATION_EMAIL, auto_healer
    await redis_cache.connect()

    # Init DB
    if DATABASE_URL:
        try:
            db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=2, max_size=10, command_timeout=30)
            async with db_pool.acquire() as conn:
                for stmt in SCHEMA.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        try:
                            await conn.execute(stmt)
                        except Exception as e:
                            logger.warning(f"Schema stmt skipped: {e}")
                row = await conn.fetchrow("SELECT value FROM app_settings WHERE key='notification_email'")
                if row and row["value"]:
                    NOTIFICATION_EMAIL = row["value"]
                elif NOTIFICATION_EMAIL:
                    await conn.execute(
                        """
                        INSERT INTO app_settings (key, value, updated_at)
                        VALUES ('notification_email', $1, NOW())
                        ON CONFLICT (key) DO NOTHING
                        """,
                        NOTIFICATION_EMAIL,
                    )
            logger.info("✓ PostgreSQL connected and schema ready")
        except Exception as e:
            logger.error(f"✗ PostgreSQL connection failed: {e}")
            db_pool = None
    else:
        logger.warning("DATABASE_URL not set — database features disabled")

    # Register agents and tools
    build_registry()
    tool_registry.register(FileSystemTool())
    tool_registry.register(TerminalTool())
    tool_registry.register(PostgreSQLTool())
    tool_registry.register(BrowserTool())
    tool_registry.register(SecurityScannerTool())
    tool_registry.register(CodeFormatterTool())
    tool_registry.register(EmailNotificationTool())
    tool_registry.register(GmailAutomationTool())
    tool_registry.register(SSHTool())
    tool_registry.register(WebSearchTool())
    tool_registry.register(AgentInvokeTool())
    tool_registry.register(SocialMediaTool())
    tool_registry.register(CodeRunnerTool())
    tool_registry.register(PlaywrightTool())
    tool_registry.register(FrontendPreviewTool())
    tool_registry.register(ServerRunnerTool())
    tool_registry.register(ParallelTeamTool())
    tool_registry.register(ParallelEngineeringTool())

    logger.info(f"✓ {len(AGENT_REGISTRY)} agents registered")
    logger.info(f"✓ {len(tool_registry._tools)} MCP tools registered")

    # Rebuild RAG index
    await rag_engine.rebuild_index()

    # Announce ready
    if db_pool:
        try:
            await db_execute(
                "INSERT INTO logs (level, message) VALUES ('INFO', 'Jaxvora system_ready')"
            )
        except Exception:
            pass

    # Start auto-heal daemon
    global auto_healer
    auto_healer = AutoHealDaemon(orchestrator)
    auto_healer.start()

    # Bootstrap sequence
    asyncio.create_task(_bootstrap_sequence())
    logger.info("🚀 Jaxvora ready")


async def _bootstrap_sequence():
    """Run bootstrap health-check after startup."""
    await asyncio.sleep(2)
    checks: List[Dict] = []
    # MCP health-check
    checks.append(_doctor_check("MCP tool registry",
        len(tool_registry.list_tools()) >= 8,
        f"{len(tool_registry.list_tools())} tools registered"))
    # DB verify
    db_ok = db_pool is not None
    checks.append(_doctor_check("Database", db_ok,
        "PostgreSQL pool connected" if db_ok else "Not available"))
    if db_ok:
        try:
            row = await db_fetchrow("SELECT COUNT(*) as c FROM rag_documents")
            doc_count = row["c"] if row else 0
            checks.append(_doctor_check("RAG index", True, f"{doc_count} documents indexed"))
        except Exception:
            checks.append(_doctor_check("RAG index", False, "Query failed"))
    # Agent registry
    checks.append(_doctor_check("Agent registry",
        len(AGENT_REGISTRY) >= 30,
        f"{len(AGENT_REGISTRY)} agents registered"))
    # Session resume
    pending_sessions = 0
    if db_ok:
        try:
            rows = await db_fetch(
                "SELECT COUNT(*) as c FROM jaxvora_sessions "
                "WHERE updated_at > NOW() - INTERVAL '24 hours'")
            pending_sessions = rows[0]["c"] if rows else 0
        except Exception:
            pass
    checks.append(_doctor_check("Pending sessions",
        pending_sessions == 0,
        f"{pending_sessions} active sessions in last 24h"))
    # Escalation backlog
    unresolved = error_escalation.unresolved_count()
    checks.append(_doctor_check("Error escalation",
        unresolved == 0,
        f"{unresolved} unresolved escalations"))
    # Compile
    try:
        import py_compile as _pc
        _pc.compile(__file__, doraise=True)
        checks.append(_doctor_check("Python compile", True, "main.py compiles cleanly"))
    except Exception as exc:
        checks.append(_doctor_check("Python compile", False, str(exc)))
    passed = sum(1 for c in checks if c["ok"])
    status = "all_ok" if passed == len(checks) else "issues_detected"
    report = {
        "type": "bootstrap_report",
        "status": status,
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    await ws_manager._send(ws_manager.agents, report)
    await log_to_db("INFO" if status == "all_ok" else "WARN",
        f"Bootstrap: {passed}/{len(checks)} checks passed")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
