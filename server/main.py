# === SECTION 1: Imports & Config ===

import os
import json
import asyncio
import traceback
import uuid
import logging
import re
import subprocess
import tempfile
import hmac
import html
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any, Set
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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, UploadFile, File, Form, Header
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jaxvora")

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "")
PORT = int(os.environ.get("PORT", 8080))

GMAIL_SENDER = os.environ.get("GMAIL_SENDER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
NOTIFICATION_EMAIL = os.environ.get("NOTIFICATION_EMAIL", "")
GMAIL_AUTOMATION_USER = os.environ.get("GMAIL_AUTOMATION_USER", os.environ.get("GMAIL_USER", "jaxvora@gmail.com"))
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

# ── LLM helpers ────────────────────────────────────────────────────────────────

async def call_openrouter(system: str, user: str, model: str = DEEPSEEK_MODEL) -> str:
    if not OPENROUTER_API_KEY:
        return f"[Mock response — OPENROUTER_API_KEY not set] Task: {user[:100]}"
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                f"{OPENROUTER_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "HTTP-Referer": "https://jaxvora.ai",
                    "X-Title": "Jaxvora",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": 2048,
                },
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[OpenRouter error: {e}]"


async def call_groq(system: str, user: str) -> str:
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — falling back to OpenRouter")
        return await call_openrouter(system, user)
    delays = [1, 3, 7]
    for attempt, delay in enumerate(delays + [None]):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                    json={
                        "model": LLAMA_MODEL,
                        "messages": [
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        "max_tokens": 2048,
                    },
                )
                if r.status_code == 429:
                    if delay is not None:
                        logger.warning(f"Groq rate-limited (429), retrying in {delay}s (attempt {attempt+1}/3)")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.warning("Groq rate-limited after 3 retries — falling back to OpenRouter")
                        return await call_openrouter(system, user)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError:
            raise
        except Exception as e:
            if delay is not None:
                await asyncio.sleep(delay)
                continue
            logger.error(f"Groq error after retries: {e} — falling back to OpenRouter")
            return await call_openrouter(system, user)
    return await call_openrouter(system, user)


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
        "{{year}}": str(datetime.utcnow().year),
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
    if not attachment_data and not attachment_name and gmail_automation_status().get("configured"):
        api_result = await run_gmail_automation({
            "action": "send",
            "to": to_email,
            "subject": subject,
            "body": body,
            "confirm": True,
        })
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
    msg = _gmail_mime_message(params)
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


# === SECTION 2: Database Layer ================================================

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
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def run(self, params: Dict[str, Any]) -> str:
        raise NotImplementedError


class FileSystemTool(MCPTool):
    def __init__(self):
        super().__init__("file_system", "Read and write files in the workspace")

    async def run(self, params: Dict[str, Any]) -> str:
        try:
            action = params.get("action", "read")
            path = params.get("path", "")
            if action == "read":
                with open(path) as f:
                    return f.read()
            elif action == "write":
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w") as f:
                    f.write(params.get("content", ""))
                return f"Written to {path}"
            return "Unknown action"
        except Exception as e:
            return f"file_system error: {e}"


class TerminalTool(MCPTool):
    ALLOWED = re.compile(r'^(ls|cat|echo|pwd|find|grep|wc|head|tail|python3? -c|pip show)\b')

    def __init__(self):
        super().__init__("terminal", "Run sandboxed shell commands (read-only)")

    async def run(self, params: Dict[str, Any]) -> str:
        cmd = params.get("command", "")
        if not self.ALLOWED.match(cmd):
            return f"Command blocked for safety: {cmd}"
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=10
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "Command timed out"
        except Exception as e:
            return f"terminal error: {e}"


class PostgreSQLTool(MCPTool):
    def __init__(self):
        super().__init__("postgresql", "Execute SQL queries against the database")

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
        super().__init__("browser", "Fetch and parse web pages")

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
        super().__init__("security_scanner", "Static analysis for secrets and vulnerabilities")

    async def run(self, params: Dict[str, Any]) -> str:
        content = params.get("content", "")
        findings = []
        for pattern, desc in self.PATTERNS:
            if re.search(pattern, content):
                findings.append(f"⚠ {desc}")
        return "\n".join(findings) if findings else "✓ No obvious issues found"


class CodeFormatterTool(MCPTool):
    def __init__(self):
        super().__init__("code_formatter", "Lint and format code suggestions")

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
        super().__init__("email_notify", "Send email notifications for bugs, issues, and alerts to the configured recipient")

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
        )

    async def run(self, params: Dict[str, Any]) -> str:
        result = await run_gmail_automation(params)
        return json.dumps(result, indent=2, default=str)


SSH_BLOCKED_COMMAND_PATTERNS = [
    r"\brm\s+-rf\s+/",
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
    r"\bchmod\s+-R\s+777\s+/",
    r"\bchown\s+-R\b.*\s+/",
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
        super().__init__("ssh_exec", "Execute commands on a remote server via SSH for 24/7 monitoring and management")

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
                "known_hosts": None,
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


class MCPToolRegistry:
    def __init__(self):
        self._tools: Dict[str, MCPTool] = {}

    def register(self, tool: MCPTool):
        self._tools[tool.name] = tool

    async def run(self, name: str, params: Dict[str, Any]) -> str:
        if name not in self._tools:
            return f"Tool '{name}' not found"
        return await self._tools[name].run(params)

    def list_tools(self) -> List[Dict]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]


tool_registry = MCPToolRegistry()


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
                collection, content, json.dumps(metadata or {})
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

    async def call_llm(self, system: str, user: str) -> str:
        if self.model == "groq":
            return await call_groq(system, user)
        else:
            return await call_openrouter(system, user)

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
            await db_execute(
                "UPDATE tasks SET status='completed', output=$1, completed_at=NOW() WHERE id=$2",
                output, task_id
            )
            await db_execute(
                "INSERT INTO agent_history (agent_name, task_summary, outcome) VALUES ($1, $2, 'success')",
                self.name, task[:120]
            )
            await log_to_db("INFO", f"[{self.name}] Completed ✓", task_id)
            self._status = "idle"
            self._current_task = ""
            await ws_manager.broadcast_agent_status(self.name, "idle", "")
            return AgentResult(self.name, task, output, True, task_id)
        except Exception as e:
            tb = traceback.format_exc()
            await db_execute(
                "UPDATE tasks SET status='failed', output=$1, completed_at=NOW() WHERE id=$2",
                str(e), task_id
            )
            await log_to_db("ERROR", f"[{self.name}] Error: {e}", task_id)
            self._status = "error"
            await ws_manager.broadcast_agent_status(self.name, "error", str(e)[:60])
            return AgentResult(self.name, task, f"Error: {e}\n{tb}", False, task_id)

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


# === SECTION 6: Agent Implementations =========================================

class AIEngineerAgent(BaseAgent):
    name = "AI Engineer"; model = "deepseek"; division = "Engineering"
    description = "AI features, RAG systems, MCP integrations, LLM workflows"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert AI engineer specialising in LLM integrations, RAG systems, "
            "embeddings, vector databases, MCP tool design, and production AI workflows. "
            "Provide detailed, actionable technical guidance.",
            task
        )

class SoftwareEngineerAgent(BaseAgent):
    name = "Software Engineer"; model = "deepseek"; division = "Engineering"
    description = "Backend/frontend dev, CRUD, API generation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior full-stack software engineer. You write clean, production-ready "
            "code with best practices, proper error handling, and comprehensive comments.",
            task
        )

class DebugAgent(BaseAgent):
    name = "Debug Agent"; model = "deepseek"; division = "Engineering"
    description = "Root-cause analysis, log investigation, automated bug fixing"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert debugger. You perform systematic root-cause analysis, read "
            "stack traces, investigate logs, and provide precise bug fixes with explanations.",
            task
        )

class QATestAgent(BaseAgent):
    name = "QA/Test Agent"; model = "deepseek"; division = "Engineering"
    description = "Unit, integration, E2E, regression tests"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a QA automation engineer. You write comprehensive test suites covering "
            "unit, integration, and E2E scenarios. Use pytest, Jest, or appropriate frameworks.",
            task
        )

class CodeReviewAgent(BaseAgent):
    name = "Code Review"; model = "groq"; division = "Engineering"
    description = "Code quality, best practices, risk & security review"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior code reviewer. You evaluate code for quality, security, "
            "performance, maintainability, and adherence to best practices. Be specific.",
            task
        )

class ArchitectureAgent(BaseAgent):
    name = "Architecture"; model = "groq"; division = "Engineering"
    description = "System design, scalability, technical debt"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a principal systems architect. You design scalable, resilient systems "
            "and identify technical debt, single points of failure, and improvement areas.",
            task
        )

class DatabaseAgent(BaseAgent):
    name = "Database"; model = "deepseek"; division = "Engineering"
    description = "Query optimisation, schema design, migrations"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a database expert specialising in PostgreSQL, query optimisation, "
            "schema design, indexing strategies, and zero-downtime migrations.",
            task
        )

class DevOpsAgent(BaseAgent):
    name = "DevOps"; model = "deepseek"; division = "Engineering"
    description = "CI/CD, Docker configs, Kubernetes manifests, deployments"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a DevOps/SRE engineer. You design CI/CD pipelines, write Dockerfiles, "
            "Kubernetes manifests, Terraform configs, and automate deployments reliably.",
            task
        )

class CybersecurityAgent(BaseAgent):
    name = "Cybersecurity"; model = "deepseek"; division = "Security"
    description = "Vulnerability scanning, secret detection, hardening"
    async def _execute(self, task):
        result = await self.call_llm(
            "You are a cybersecurity engineer. You identify vulnerabilities, detect exposed "
            "secrets, recommend hardening measures, and produce actionable security reports.",
            task
        )
        await memory.store("security_findings", f"Task: {task}\nFindings: {result[:500]}")
        return result

class RedTeamAgent(BaseAgent):
    name = "Red Team"; model = "deepseek"; division = "Security"
    description = "Threat modelling, attack simulation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a red team security expert. You perform threat modelling, identify "
            "attack vectors, and simulate adversarial scenarios to strengthen defences.",
            task
        )

class ComplianceAgent(BaseAgent):
    name = "Compliance"; model = "deepseek"; division = "Security"
    description = "GDPR, SOC2, ISO27001 checklists"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a compliance officer specialising in GDPR, SOC2, ISO27001, and HIPAA. "
            "Produce detailed compliance checklists and gap analysis reports.",
            task
        )

class DataAnalystAgent(BaseAgent):
    name = "Data Analyst"; model = "deepseek"; division = "Data"
    description = "SQL analysis, KPI tracking, business insights"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior data analyst. You write SQL queries, build KPI dashboards, "
            "interpret trends, and translate data into clear business insights.",
            task
        )

class BIAgent(BaseAgent):
    name = "BI Agent"; model = "deepseek"; division = "Data"
    description = "Power BI reports, DAX generation, semantic model analysis"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a Business Intelligence expert specialising in Power BI, Tableau, "
            "DAX formulas, semantic models, and executive dashboard design.",
            task
        )

class DataEngineerAgent(BaseAgent):
    name = "Data Engineer"; model = "deepseek"; division = "Data"
    description = "ETL pipelines, data quality, warehouse optimisation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a data engineer. You design ETL/ELT pipelines with dbt, Spark, or "
            "Airflow, enforce data quality contracts, and optimise warehouse performance.",
            task
        )

class MLEngineerAgent(BaseAgent):
    name = "ML Engineer"; model = "deepseek"; division = "Data"
    description = "Feature engineering, model training, evaluation pipelines"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an ML engineer. You design feature pipelines, train and evaluate models, "
            "handle model versioning, and deploy ML systems to production.",
            task
        )

class ResumeAgent(BaseAgent):
    name = "Resume Agent"; model = "deepseek"; division = "Career"
    description = "ATS-optimised resume and portfolio generation"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a professional resume writer and career coach. You create ATS-optimised "
            "resumes and portfolios that highlight achievements with quantified impact.",
            task
        )

class InterviewCoachAgent(BaseAgent):
    name = "Interview Coach"; model = "deepseek"; division = "Career"
    description = "Technical, behavioural, and mock interview prep"
    async def _execute(self, task):
        return await self.call_llm(
            "You are an expert interview coach for tech roles. You prepare candidates for "
            "system design, coding, and behavioural interviews with detailed coaching.",
            task
        )

class CareerCoachAgent(BaseAgent):
    name = "Career Coach"; model = "deepseek"; division = "Career"
    description = "Learning plans, skill-gap analysis, career guidance"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior tech career coach. You analyse skill gaps, design 90-day "
            "learning plans, and provide strategic career progression guidance.",
            task
        )

class ProductManagerAgent(BaseAgent):
    name = "Product Manager"; model = "deepseek"; division = "Product"
    description = "Roadmaps, feature planning, user stories"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a senior product manager. You create roadmaps, write user stories with "
            "acceptance criteria, prioritise backlogs, and align stakeholders.",
            task
        )

class DocumentationAgent(BaseAgent):
    name = "Documentation"; model = "deepseek"; division = "Product"
    description = "Technical docs, API docs, architecture docs"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a technical writer. You produce clear, comprehensive documentation "
            "including API references, architecture docs, and developer guides.",
            task
        )

class ResearchAgent(BaseAgent):
    name = "Research"; model = "deepseek"; division = "Product"
    description = "Technology research, framework comparison"
    async def _execute(self, task):
        return await self.call_llm(
            "You are a technology researcher. You compare frameworks, evaluate libraries, "
            "assess trade-offs, and produce well-structured research reports.",
            task
        )

class ProjectIntelligenceAgent(BaseAgent):
    name = "Project Intelligence"; model = "groq"; division = "Executive"
    description = "Dependency graph, architecture graph, impact analysis"
    async def _execute(self, task):
        result = await self.call_llm(
            "You are a project intelligence system. You analyse codebases, build dependency "
            "graphs, assess change impact, and provide architectural context.",
            task
        )
        await memory.store("architecture_knowledge", f"Analysis: {result[:600]}")
        return result


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
        match = re.search(r'\{.*\}', raw, re.DOTALL)
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


AGENT_REGISTRY: Dict[str, BaseAgent] = {}
AGENT_NETWORK: Dict[str, List[str]] = {}
MAX_PARALLEL_AGENTS = int(os.environ.get("MAX_PARALLEL_AGENTS", "6"))
DIVISION_LEADS = {
    "Engineering": "Architecture",
    "Security": "Cybersecurity",
    "Data": "Data Engineer",
    "Career": "Career Coach",
    "Product": "Product Manager",
    "Executive": "Project Intelligence",
}

def build_registry():
    global AGENT_NETWORK
    agents = [
        AIEngineerAgent(), SoftwareEngineerAgent(), DebugAgent(), QATestAgent(),
        CodeReviewAgent(), ArchitectureAgent(), DatabaseAgent(), DevOpsAgent(),
        CybersecurityAgent(), RedTeamAgent(), ComplianceAgent(),
        DataAnalystAgent(), BIAgent(), DataEngineerAgent(), MLEngineerAgent(),
        ResumeAgent(), InterviewCoachAgent(), CareerCoachAgent(),
        ProductManagerAgent(), DocumentationAgent(), ResearchAgent(),
        ProjectIntelligenceAgent(),
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

    SYSTEM = """You are Jaxvora's Chief Orchestrator powered by Llama 3.3 70B.
Your role: parse user intent, create execution plans, route to specialist agents, and synthesise results.
Write user-facing responses in clean markdown with short headings, bullets, and fenced code blocks when useful.
Do not return raw JSON, internal traces, or long unstructured paragraphs in the final response.
Never say Jaxvora cannot use a configured tool; route tool-specific requests before giving generic advice.

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
        (("deploy", "server", "docker", "ci", "cd", "vercel", "vm", "ssh"), ["DevOps", "Architecture", "QA/Test Agent"]),
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

    async def _handle_doctor_chat(self, user_input: str) -> Optional[Dict[str, Any]]:
        if not self._is_doctor_chat_intent(user_input):
            return None
        iterations = 3 if any(word in user_input.lower() for word in ("until fixed", "loop", "continuously", "monitor")) else 2
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
        response = await call_groq(
            "You are Jaxvora's Chief Orchestrator. Synthesize parallel department work into a concise, decisive company response.",
            synthesis_prompt,
        )
        if response.startswith("["):
            return plan.get("response", "Task completed.") + "\n\n" + response
        return response

    async def process(self, user_input: str, stream_fn=None, admin_token: Optional[str] = None) -> Dict:
        gmail_result = await self._handle_gmail_chat(user_input, admin_token=admin_token)
        if gmail_result:
            return gmail_result
        ssh_result = await self._handle_ssh_chat(user_input)
        if ssh_result:
            return ssh_result
        doctor_result = await self._handle_doctor_chat(user_input)
        if doctor_result:
            return doctor_result

        try:
            raw = await call_groq(self.SYSTEM, user_input)
            # Extract JSON
            match = re.search(r'\{.*\}', raw, re.DOTALL)
            if match:
                plan = json.loads(match.group())
            else:
                plan = {"plan": "Direct response", "agents": [], "response": raw}
        except Exception as e:
            plan = {
                "plan": "Fallback direct response",
                "agents": [],
                "response": f"I'll help you with that. {user_input[:100]}... (Orchestrator note: {e})"
            }

        squad = self._build_company_squad(user_input, plan.get("agents", []))
        results = await self._run_parallel_squad(user_input, plan, squad, stream_fn=stream_fn)

        # Log to audit and auto-decide
        try:
            row = await db_fetchrow(
                "INSERT INTO audit (agent_name, action, payload) VALUES ($1, $2, $3) RETURNING id",
                "Chief Orchestrator", "process_request", user_input[:500]
            )
            if row:
                asyncio.create_task(auto_decide_audit(str(row["id"]), "Chief Orchestrator", "process_request", user_input[:500]))
        except Exception:
            pass

        final = await self._synthesise_company_response(user_input, plan, results)
        return {
            "plan": plan.get("plan", ""),
            "agents": squad,
            "response": final,
            "results": results,
            "organization": {
                "mode": "parallel_company",
                "max_parallel_agents": MAX_PARALLEL_AGENTS,
            },
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
        for ws in connections:
            try:
                await ws.send_text(json.dumps(data, default=str))
            except Exception:
                dead.add(ws)
        connections -= dead

    async def broadcast_agent_status(self, name: str, status: str, task: str):
        await self._send(self.agents, {"type": "agent_status", "name": name, "status": status, "task": task, "ts": datetime.now().isoformat()})

    async def broadcast_task(self, task: Dict):
        await self._send(self.tasks_ws, {"type": "task_update", **task})

    async def broadcast_log(self, level: str, message: str):
        await self._send(self.logs_ws, {"type": "log", "level": level, "message": message, "ts": datetime.now().isoformat()})

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
    global db_pool
    if db_pool:
        await db_pool.close()


app = FastAPI(title="Jaxvora", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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


@app.post("/chat")
async def chat(req: ChatRequest):
    result = await orchestrator.process(req.message, admin_token=req.admin_token)
    return result


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
        req.name, req.repo_url, json.dumps(req.metadata)
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
    payload = req.dict()
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


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Accept file upload and return its text content for agent context."""
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        text = base64.b64encode(content).decode()
    return {
        "filename": file.filename,
        "size": len(content),
        "content": text[:8000],
        "truncated": len(text) > 8000,
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

async def startup():
    global db_pool, NOTIFICATION_EMAIL

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

    logger.info(f"✓ {len(AGENT_REGISTRY)} agents registered")
    logger.info(f"✓ {len(tool_registry._tools)} MCP tools registered")

    # Announce ready
    if db_pool:
        try:
            await db_execute(
                "INSERT INTO logs (level, message) VALUES ('INFO', 'Jaxvora system_ready')"
            )
        except Exception:
            pass

    asyncio.create_task(_broadcast_ready())
    logger.info("🚀 Jaxvora ready")


async def _broadcast_ready():
    await asyncio.sleep(1)
    await ws_manager._send(ws_manager.agents, {"type": "system_ready", "ts": datetime.now().isoformat()})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=PORT, reload=False)
