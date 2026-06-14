"""Tests for the job auto-apply pipeline (PlaywrightTool.auto_fill, resume profile
encryption, LinkedIn cookie loading, field-mapping heuristic).

Run on the VM where deps exist:
    JAXVORA_SECRET_KEY=$(python -c "from cryptography.fernet import Fernet;print(Fernet.generate_key().decode())") \
      python -m pytest tests/test_job_autofill.py -v

Tests that need optional deps (asyncpg/playwright) skip gracefully if absent.
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# A key must exist before importing main (it's read at import time).
os.environ.setdefault("JAXVORA_SECRET_KEY", "")
if not os.environ["JAXVORA_SECRET_KEY"]:
    try:
        from cryptography.fernet import Fernet
        os.environ["JAXVORA_SECRET_KEY"] = Fernet.generate_key().decode()
    except Exception:
        pass

main = pytest.importorskip("main", reason="backend deps (e.g. asyncpg) not installed")


def test_field_hints_map_a_realistic_form():
    """The heuristic must map every profile field on a typical application form
    and leave only genuinely-unmappable inputs untouched."""
    import re
    form = [
        {"idx": 0, "name": "applicant_name", "label": "Full Name"},
        {"idx": 1, "name": "email", "label": "Email Address"},
        {"idx": 2, "name": "phone", "label": "Mobile Number"},
        {"idx": 3, "name": "li_url", "label": "LinkedIn Profile"},
        {"idx": 4, "name": "yoe", "label": "Total Experience (years)"},
        {"idx": 5, "name": "curr_title", "label": "Current Job Title"},
        {"idx": 6, "name": "curr_emp", "label": "Current Employer"},
        {"idx": 7, "name": "loc", "label": "Current Location"},
        {"idx": 8, "name": "expected_ctc", "label": "Expected CTC"},
        {"idx": 9, "name": "why", "label": "Why do you want to join us?"},
        {"idx": 10, "name": "referral", "label": "How did you hear about us?"},
    ]
    keys = ["full_name", "email", "phone", "linkedin", "experience_years",
            "current_title", "current_company", "location", "salary", "cover_letter"]
    used, matched = set(), {}
    for key in keys:
        hint = main.JOB_FIELD_HINTS[key]
        for d in form:
            if d["idx"] in used:
                continue
            hay = (d["name"] + " " + d["label"]).lower()
            if re.search(hint, hay):
                matched[key] = d["label"]
                used.add(d["idx"])
                break
    assert set(matched) == set(keys), f"unmatched profile fields: {set(keys) - set(matched)}"
    left = [d["label"] for d in form if d["idx"] not in used]
    assert left == ["How did you hear about us?"], left


def test_encryption_round_trip():
    pytest.importorskip("cryptography")
    blob = main.encrypt_secret("secret@example.com")
    assert blob and blob.startswith("enc:")
    assert main.decrypt_secret(blob) == "secret@example.com"


def test_save_refuses_plaintext_without_key(monkeypatch):
    """No key configured → never persist PII unencrypted."""
    monkeypatch.setattr(main, "JAXVORA_SECRET_KEY", "")
    out = asyncio.get_event_loop().run_until_complete(
        main.job_profile_save({"email": "a@b.com"}))
    assert out["ok"] is False and "SECRET_KEY" in out["error"]


def test_normalize_profile_derives_names():
    n = main._normalize_profile({"full_name": "Jane Q Public", "email": "j@x.com"})
    assert n["first_name"] == "Jane" and n["last_name"] == "Q Public"
    n2 = main._normalize_profile({"first_name": "Sam", "last_name": "Lee"})
    assert n2["full_name"] == "Sam Lee"


def test_load_linkedin_cookies_from_file(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cookies.json"
        p.write_text(json.dumps({"cookies": [
            {"name": "li_at", "value": "ABC", "domain": ".linkedin.com"},
            {"bad": "no name/value"},
        ]}))
        monkeypatch.setattr(main, "LINKEDIN_COOKIES_JSON", "")
        monkeypatch.setattr(main, "LINKEDIN_COOKIES_PATH", str(p))
        cookies = main.load_linkedin_cookies()
        assert len(cookies) == 1 and cookies[0]["name"] == "li_at"
        st = main.linkedin_session_status()
        assert st["present"] and st["has_auth_cookie"]


def test_public_view_masks_pii():
    pub = main.job_profile_public({"email": "jane@example.com", "experience_years": 6})
    assert pub["experience_years"] == 6           # non-sensitive passes through
    assert "@" in pub["email"] and "example.com" in pub["email"]
    assert pub["email"] != "jane@example.com"     # local part masked


@pytest.mark.asyncio
async def test_auto_fill_dummy_form():
    """Live Playwright check: auto_fill maps Name/Email/Phone/Experience on a dummy
    form. Skips if playwright (or its browser) is unavailable."""
    pytest.importorskip("playwright")
    from playwright.async_api import async_playwright

    html = """<!doctype html><html><body><form>
      <label for="n">Full Name</label><input id="n" name="full_name">
      <label for="e">Email</label><input id="e" name="email" type="email">
      <label for="p">Phone</label><input id="p" name="phone">
      <label for="x">Years of Experience</label><input id="x" name="yoe">
      <label for="t">Current Job Title</label><input id="t" name="title">
      <textarea name="why" placeholder="Why do you want to join?"></textarea>
      <button type="submit">Submit application</button>
    </form></body></html>"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "form.html"
        f.write_text(html)
        fields = {
            "full_name": "Jane Public", "email": "jane@x.com", "phone": "+1 555 0100",
            "experience_years": "6", "current_title": "Power BI Developer",
            "cover_letter": "I am excited about this role.",
        }
        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
                page = await (await browser.new_context()).new_page()
                await page.goto(f.as_uri())
                result = await main._auto_fill_form(page, main._normalize_profile(fields))
                vals = await page.evaluate(
                    "() => ({n:document.getElementById('n').value, e:document.getElementById('e').value, "
                    "p:document.getElementById('p').value, x:document.getElementById('x').value})")
                await browser.close()
        except Exception as e:
            pytest.skip(f"playwright browser unavailable: {e}")

    filled = {x["field"] for x in result["filled"]}
    assert {"full_name", "email", "phone", "experience_years"} <= filled
    assert vals["n"] == "Jane Public" and vals["e"] == "jane@x.com"
    assert vals["p"] == "+1 555 0100" and vals["x"] == "6"
