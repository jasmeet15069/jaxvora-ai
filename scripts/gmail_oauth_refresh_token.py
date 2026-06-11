#!/usr/bin/env python3
"""Generate or exchange a Gmail OAuth consent code for a refresh token.

This helper intentionally has no third-party dependencies. It reads a Google
OAuth client-secret JSON file, prints a consent URL, and can exchange the
returned redirect URL/code for tokens.
"""

import argparse
import json
import secrets
import sys
import urllib.parse
import urllib.request
from pathlib import Path


DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
    "https://www.googleapis.com/auth/gmail.settings.basic",
]


def load_client(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    client = data.get("installed") or data.get("web") or data
    required = ["client_id", "client_secret"]
    missing = [key for key in required if not client.get(key)]
    if missing:
        raise SystemExit(f"Client secret JSON is missing: {', '.join(missing)}")
    return client


def redirect_uri(client: dict, override: str = "") -> str:
    if override:
        return override
    uris = client.get("redirect_uris") or []
    if not uris:
        raise SystemExit("No redirect URI found. Pass --redirect-uri explicitly.")
    return uris[0]


def auth_url(client: dict, redirect: str, scopes: list[str]) -> str:
    params = {
        "client_id": client["client_id"],
        "redirect_uri": redirect,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
        "state": secrets.token_urlsafe(18),
    }
    return "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)


def extract_code(code: str, redirect_url: str) -> str:
    if code:
        return code
    parsed = urllib.parse.urlparse(redirect_url)
    params = urllib.parse.parse_qs(parsed.query)
    values = params.get("code")
    if not values:
        raise SystemExit("No OAuth code found. Pass --code or --redirect-url.")
    return values[0]


def exchange_code(client: dict, redirect: str, code: str) -> dict:
    payload = urllib.parse.urlencode(
        {
            "client_id": client["client_id"],
            "client_secret": client["client_secret"],
            "redirect_uri": redirect,
            "code": code,
            "grant_type": "authorization_code",
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        "https://oauth2.googleapis.com/token",
        data=payload,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Token exchange failed ({exc.code}): {detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Gmail OAuth refresh token helper")
    parser.add_argument("--client-secret", required=True, type=Path)
    parser.add_argument("--redirect-uri", default="")
    parser.add_argument("--code", default="")
    parser.add_argument("--redirect-url", default="")
    parser.add_argument("--output", type=Path, help="Optional JSON file for the token response")
    parser.add_argument("--scope", action="append", dest="scopes", help="Override/add scope. Repeatable.")
    args = parser.parse_args()

    client = load_client(args.client_secret)
    redirect = redirect_uri(client, args.redirect_uri)
    scopes = args.scopes or DEFAULT_SCOPES

    if not args.code and not args.redirect_url:
        print(auth_url(client, redirect, scopes))
        return 0

    tokens = exchange_code(client, redirect, extract_code(args.code, args.redirect_url))
    public = {
        "has_refresh_token": bool(tokens.get("refresh_token")),
        "token_type": tokens.get("token_type"),
        "expires_in": tokens.get("expires_in"),
        "scope": tokens.get("scope"),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(tokens, indent=2), encoding="utf-8")
        public["saved_to"] = str(args.output)
    else:
        public["refresh_token"] = tokens.get("refresh_token")
    print(json.dumps(public, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
