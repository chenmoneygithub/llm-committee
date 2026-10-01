"""Isolated, offline browser check for the published forced-feedback supplement."""

from __future__ import annotations

import json
import re

from playwright.sync_api import sync_playwright

from llm_committee.pivot.models import digest
from llm_committee.pivot.study import atomic_json
from scripts.forced_feedback_public_history import OUTPUT
from scripts.report_forced_feedback_public_history import PAGE, SECTION
from scripts.scale_public_history import ROOT, read


def check():
    base = (OUTPUT / "base-report-before-supplement.html").read_text()
    stripped = re.sub(
        r"<!-- ff-supplement:(\w+):start -->.*?<!-- ff-supplement:\1:end -->", "", PAGE.read_text(), flags=re.S
    )
    assert stripped == base, "Main report HTML changed outside supplement blocks"
    before, after = read(OUTPUT / "base-report-before-supplement.json"), read(PAGE.with_suffix(".json"))
    assert digest({k: v for k, v in before.items() if k != "supplements"}) == digest(
        {k: v for k, v in after.items() if k != "supplements"}
    )
    errors, external = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", headless=True
        )
        context = browser.new_context(viewport={"width": 1280, "height": 980})
        context.route("http://**/*", lambda route: (external.append(route.request.url), route.abort()))
        context.route("https://**/*", lambda route: (external.append(route.request.url), route.abort()))
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(PAGE.as_uri(), wait_until="load")
        root = page.locator(f"#{SECTION}")
        assert root.count() == 1 and root.locator(".ff-case").count() == 300
        assert root.locator(".ff-timing").count() == 2
        page.locator("#ff-supplement-link").click()
        page.screenshot(path=str(OUTPUT / "browser-report.png"))
        root.locator("summary").filter(has_text="逐例查看：全部 300").click()
        assert root.locator(".ff-case:not([hidden])").count() == 300
        root.locator(".ff-filter-time").select_option("3")
        root.locator(".ff-filter-model").select_option("Qwen3.8-27B")
        assert root.locator(".ff-case:not([hidden])").count() == 50
        example = root.locator(".ff-case:not([hidden])").first
        example.locator(":scope > summary").click()
        example.locator("summary").filter(has_text="D：完整选项与认同概率").click()
        assert example.get_by_text("认同档位", exact=True).is_visible()
        assert example.get_by_text("G = 7", exact=True).is_visible()
        example.locator(":scope > summary").scroll_into_view_if_needed()
        page.screenshot(path=str(OUTPUT / "browser-example.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), "Mobile document overflow"
        page.locator("#ff-supplement-link").click()
        page.screenshot(path=str(OUTPUT / "browser-mobile.png"))
        page.set_viewport_size({"width": 1280, "height": 980})
        dashboard = ROOT / "docs/committee-experiment-dashboard.html"
        page.goto(dashboard.as_uri(), wait_until="load")
        assert page.get_by_role("tab").count() == 6
        page.locator("#tab-mixed-family-2").click()
        frame = page.frame_locator("#panel-mixed-family-2 iframe")
        frame.locator(f"#{SECTION}").wait_for(state="attached")
        assert frame.locator(".ff-case").count() == 300
        with page.expect_popup() as popup:
            page.locator("#panel-mixed-family-2 .source-link").click()
        popup.value.wait_for_load_state()
        assert popup.value.locator(f"#{SECTION}").count() == 1
        popup.value.close()
        assert not errors and not external, (errors, external)
        browser.close()
    result = {
        "status": "passed",
        "cases": 300,
        "filter_combination_cases": 50,
        "six_tabs_preserved": True,
        "main_html_unchanged_outside_insertions": True,
        "main_json_observations_unchanged": True,
        "desktop_and_mobile_checked": True,
        "console_errors": errors,
        "external_requests": external,
    }
    atomic_json(OUTPUT / "browser-audit.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(check()))
