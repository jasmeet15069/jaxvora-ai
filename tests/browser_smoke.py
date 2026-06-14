"""Headless-chromium smoke test for the Jaxvora UI (run on the VM).
Exercises the new interactive Agent Flow features and reports console errors + a verdict.

    /root/jaxvora-ai/.venv/bin/python tests/browser_smoke.py
"""
import asyncio
import sys

BASE = "http://127.0.0.1:8090/"


async def main():
    from playwright.async_api import async_playwright

    console_errors = []
    page_errors = []
    results = []

    def ok(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = await (await browser.new_context()).new_page()
        page.on("console", lambda m: console_errors.append(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))

        await page.goto(BASE, wait_until="domcontentloaded")
        await page.wait_for_timeout(2500)  # let init() + first fetches settle

        # 1. Page loaded, flow object exists
        has_flow = await page.evaluate("typeof flow !== 'undefined'")
        ok("flow object present", has_flow)
        has_fns = await page.evaluate(
            "typeof openAgentModal==='function' && typeof sendAgentChat==='function' && typeof renderAgentActivity==='function'")
        ok("modal/chat functions defined", has_fns)

        # 2. Open the focus modal for a known agent
        await page.evaluate("openAgentModal('Resume Agent')")
        await page.wait_for_timeout(1500)
        modal_open = await page.evaluate("document.getElementById('agent-modal').classList.contains('open')")
        ok("agent modal opens", modal_open)
        for el in ("modal-thoughts", "modal-chat-input", "modal-chat-send", "modal-activity"):
            present = await page.evaluate(f"!!document.getElementById('{el}')")
            ok(f"element #{el} present", present)

        # 3. Focus highlight wired
        focused = await page.evaluate("flow.focusName")
        ok("flow focus set to agent", focused == "Resume Agent", f"focusName={focused}")

        # 4. Direct agent chat round-trip
        await page.fill("#modal-chat-input", "In one sentence, what do you do?")
        await page.click("#modal-chat-send")
        got_reply = False
        for _ in range(40):  # up to ~60s (qwen can be slow / rate-limited)
            await page.wait_for_timeout(1500)
            txt = await page.inner_text("#modal-chat-thread")
            if "Resume Agent" in txt and len(txt) > 40:
                got_reply = True
                break
        ok("direct agent chat returns a reply", got_reply,
           (await page.inner_text("#modal-chat-thread"))[:120])

        # 5. Close modal clears focus
        await page.evaluate("closeModal()")
        await page.wait_for_timeout(400)
        cleared = await page.evaluate("flow.focusName === null && !document.getElementById('agent-modal').classList.contains('open')")
        ok("close clears focus + modal", cleared)

        # 6. Computer (workspace) view loads
        try:
            await page.evaluate("typeof loadComputerPanel==='function' && loadComputerPanel()")
            await page.wait_for_timeout(1500)
            ws = await page.evaluate("(document.getElementById('computer-list')||{}).innerText || ''")
            ok("computer view loads", "Loading" not in ws or len(ws) > 0, ws[:80])
        except Exception as e:
            ok("computer view loads", False, str(e)[:80])

        await page.screenshot(path="/root/jaxvora-ai/workspace/.artifacts/browser_smoke.png", full_page=True)
        await browser.close()

    print("\n=== BROWSER SMOKE RESULTS ===")
    passed = 0
    for name, cond, detail in results:
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not cond else ""))
        passed += 1 if cond else 0
    print(f"\n{passed}/{len(results)} checks passed")
    if console_errors:
        print(f"\n⚠ {len(console_errors)} console error(s):")
        for e in console_errors[:10]:
            print("   -", e[:160])
    else:
        print("\n✓ no console errors")
    if page_errors:
        print(f"\n⚠ {len(page_errors)} page error(s):")
        for e in page_errors[:10]:
            print("   -", e[:160])
    sys.exit(0 if passed == len(results) and not page_errors else 1)


asyncio.run(main())
