"""Offline, six-setting entry point; embeds reports without changing their results.

Run with: .venv/bin/python -m scripts.report_experiment_dashboard
The explicit source map deliberately excludes archived protocols and dry runs.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

from llm_committee.pivot.models import ROSTERS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAB = "same-model-3"
TITLES = {"same_model": "Same model", "same_family": "Same family", "mixed_family": "Mixed family"}
PILOT_REPORTS = {
    ("same_model", 2): (
        "turn-tone-dyadic-same-model-public-history-2026-09-28.html",
        "public-history-dyadic-paired-2026-09-27-v1",
    ),
    ("same_model", 3): (
        "turn-tone-triadic-same-model-public-history-2026-09-28.html",
        "public-history-triadic-ABE-2026-09-28-v1",
    ),
    ("same_family", 2): (
        "turn-tone-dyadic-same-family-public-history-2026-09-28.html",
        "public-history-dyadic-paired-2026-09-27-v1",
    ),
    ("same_family", 3): (
        "turn-tone-triadic-same-family-public-history-2026-09-28.html",
        "public-history-triadic-ABE-2026-09-28-v1",
    ),
    ("mixed_family", 2): (
        "turn-tone-dyadic-public-history-2026-09-27.html",
        "public-history-dyadic-paired-2026-09-27-v1",
    ),
    ("mixed_family", 3): (
        "turn-tone-triadic-public-history-2026-09-28.html",
        "public-history-triadic-ABE-2026-09-28-v1",
    ),
}
SCALED_REPORTS = {
    (roster, members): (
        f"turn-tone-{'dyadic' if members == 2 else 'triadic'}-{roster.replace('_', '-')}-50-public-history.html",
        protocol,
    )
    for (roster, members), (_, protocol) in PILOT_REPORTS.items()
}
# Switch atomically at the cohort level: never present an unfinished expansion
# alongside a completed fifty-question cohort as if sample sizes matched.
REPORTS = (
    SCALED_REPORTS.copy()
    if all(
        (ROOT / "docs" / name).exists() and (ROOT / "docs" / name).with_suffix(".json").exists()
        for name, _ in SCALED_REPORTS.values()
    )
    else PILOT_REPORTS.copy()
)


def settings():
    """Six model-configuration × routing settings, not six tone conditions."""
    return [
        {
            "id": f"{roster.replace('_', '-')}-{members}",
            "roster": roster,
            "title": TITLES[roster],
            "members": members,
            "models": ROSTERS[roster],
            "layers": (
                "A · B · E" if members == 3 else ("A · B · C · D · E" if roster == "mixed_family" else "A · B · C · E")
            ),
        }
        for roster in TITLES
        for members in (2, 3)
    ]


def relative_url(path: Path, output: Path) -> str:
    return quote(Path(os.path.relpath(path, output.parent)).as_posix(), safe="/")


def load_report(spec, docs: Path, output: Path):
    source = REPORTS.get((spec["roster"], spec["members"]))
    if source is None:
        return None
    name, protocol = source
    path = docs / name
    raw = path.read_bytes()
    data = json.loads(path.with_suffix(".json").read_text())
    run = data["run_report"]
    if data.get("roster", spec["roster"]) != spec["roster"] or data.get("models", list(spec["models"])) != list(
        spec["models"]
    ):
        raise ValueError(f"Wrong model roster in {name}")
    if data["protocol_version"] != protocol or run["protocol_version"] != protocol:
        raise ValueError(f"Unexpected protocol in {name}; do not mix archived results")
    finished = ("completed", "completed_with_failures")
    if run["status"] not in finished or not run["kind"].startswith("paid_"):
        raise ValueError(f"Need a completed real run, not a partial run or mock: {name}")
    statuses = [run["status"]]
    quality_note = ""
    if spec["members"] == 2:
        quality = data["length_controlled_E"]
        if quality["report"]["status"] not in finished:
            raise ValueError(f"Final-answer evaluation is not complete: {name}")
        statuses.append(quality["report"]["status"])
        if "primary_rows" in quality:
            rows = quality["primary_rows"]
            available = sum(r["debate_score"] is not None for r in rows)
            quality_note = f"E2: {available}/{len(rows)} final-answer comparisons available."
    return {
        "html": raw.decode("utf-8"),
        "href": relative_url(path, output),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "questions": run["completed_questions"],
        "has_missing": "completed_with_failures" in statuses,
        "quality_note": quality_note,
    }


CSS = """
:root { color-scheme:light; --ink:#20342f; --muted:#61716c; --line:#d6dfda; --accent:#185d4c; }
* { box-sizing:border-box; }
body { margin:0; background:#f4f6f3; color:var(--ink); font:16px/1.55 system-ui,-apple-system,sans-serif; }
a { color:var(--accent); text-underline-offset:3px; }
button { font:inherit; }
[hidden] { display:none !important; }
.page-head, .tablist, .workspace, .footer { max-width:1280px; margin:auto; }
.page-head { padding:32px 32px 22px; }
.eyebrow { font-size:11px; font-weight:750; text-transform:uppercase; letter-spacing:.14em; color:var(--accent); }
h1 { font:700 36px/1.2 Georgia,serif; margin:8px 0 12px; }
.page-head p { margin:6px 0; color:var(--muted); }
.status-count { font-size:13px; }
.tabs-bar { position:sticky; top:0; z-index:2; background:#f4f6f3; border-bottom:1px solid var(--line); }
.tablist { display:grid; grid-template-columns:repeat(6,minmax(0,1fr)); gap:8px; padding:12px 32px; }
.tab { border:1px solid var(--line); border-radius:8px; background:white; color:var(--ink); text-align:left; padding:12px 14px; cursor:pointer; }
.tab:hover { border-color:var(--accent); }
.tab[aria-selected=true] { background:var(--accent); color:white; border-color:var(--accent); }
.tab:focus-visible, a:focus-visible { outline:3px solid #cd8527; outline-offset:3px; }
.tab-title { display:block; font-size:13px; font-weight:700; }
.tab-members { display:block; font-size:17px; margin:2px 0 7px; }
.tab-status { display:block; font-size:11px; }
.tab-status::before { content:''; display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:6px; background:#9aa5a0; }
.tab[data-status=completed] .tab-status::before { background:#328566; }
.tab[data-status=completed-with-gaps] .tab-status::before { background:#bf761e; }
.tab[aria-selected=true] .tab-status::before { background:#a8e5c8; }
.workspace { padding:22px 32px 36px; }
.panel { border:1px solid var(--line); border-radius:10px; overflow:hidden; background:white; }
.panel-head { padding:22px 28px; border-bottom:1px solid var(--line); }
.panel-title { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
h2 { font-size:22px; line-height:1.3; margin:0; }
.source-link { font-size:13px; white-space:nowrap; }
.models { margin:14px 0 10px; display:flex; gap:8px 20px; flex-wrap:wrap; font-size:13px; }
.models span b { display:inline-block; color:var(--accent); margin-right:5px; }
.scope, .routing { font-size:13px; color:var(--muted); margin:6px 0; }
.missing-note { font-size:13px; color:#754411; background:#fff5e5; padding:9px 12px; border-radius:5px; }
.pending { max-width:850px; padding:28px; }
.pending h3 { margin:0 0 10px; font-size:20px; }
.pending p { margin:10px 0; }
.scope-note { color:var(--muted); font-size:14px; }
iframe { display:block; width:100%; height:800px; border:0; }
.loading { padding:0 28px; color:var(--muted); }
.footer { padding:0 32px 30px; font-size:13px; color:var(--muted); }
noscript p { padding:12px 28px; background:#fff4dc; }
@media(max-width:900px) { .tablist { grid-template-columns:repeat(3,minmax(0,1fr)); } }
@media(max-width:600px) {
  .page-head { padding:24px 16px 12px; } h1 { font-size:30px; }
  .tablist { padding:10px 12px; gap:6px; } .tab { padding:9px; }
  .tab-title { font-size:11px; } .tab-members { font-size:15px; }
  .workspace { padding:14px 10px 24px; } .panel-head, .pending { padding:18px; }
  .footer { padding:0 18px 24px; }
}
@media print {
  .tabs-bar, .page-head, .footer, iframe { display:none; }
  .workspace { padding:0; } .source-link::after { content:' — open this report to print the full results'; }
}
"""

# Each report has its own document: no global CSS or section-ID collisions.
# Inline snapshots work over file:// without fetch, a server, or network access.
JS = """
(() => {
  const tabs = [...document.querySelectorAll('[role="tab"]')];
  const bar = document.querySelector('.tabs-bar');
  const observers = new Map();
  function mount(panel) {
    const frame = panel.querySelector('iframe');
    if (!frame || frame.dataset.mounted) return;
    frame.dataset.mounted = 'true';
    const snapshot = JSON.parse(document.getElementById(frame.dataset.snapshot).textContent);
    frame.addEventListener('load', () => {
      const doc = frame.contentDocument;
      if (!doc || !doc.body) return;
      const base = doc.createElement('base');
      base.href = new URL(snapshot.href, document.baseURI).href;
      doc.head.prepend(base);
      const style = doc.createElement('style');
      style.textContent = `
        html { scroll-behavior:auto; }
        body { display:flow-root; margin:0; background:white; }
        main { margin:0 auto; border:0; padding:24px 28px 40px; }
        @media(max-width:600px) { main { padding:18px 14px 30px; } }
      `;
      doc.head.append(style);
      const resize = () => {
        if (panel.hidden) return;
        frame.style.height = Math.ceil(doc.body.getBoundingClientRect().height + 4) + 'px';
      };
      const observer = new ResizeObserver(resize);
      observer.observe(doc.body);
      observers.set(panel.id, {observer, resize});
      doc.querySelectorAll('a[href]').forEach(link => {
        if (!link.getAttribute('href').startsWith('#')) {
          link.target = '_blank';
          link.rel = 'noopener';
        }
      });
      doc.addEventListener('click', event => {
        const link = event.target.closest('a[href^="#"]');
        if (!link) return;
        const target = doc.getElementById(decodeURIComponent(link.getAttribute('href').slice(1)));
        if (!target) return;
        event.preventDefault();
        const top = window.scrollY + frame.getBoundingClientRect().top
          + target.getBoundingClientRect().top - bar.offsetHeight - 16;
        window.scrollTo({top, behavior:'instant'});
      });
      panel.querySelector('.loading').hidden = true;
      resize();
    }, {once:true});
    frame.srcdoc = snapshot.html;
  }
  function activate(id, focus = false) {
    const selected = tabs.find(tab => tab.dataset.setting === id) || tabs.find(tab => tab.dataset.default);
    tabs.forEach(tab => {
      const active = tab === selected;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      document.getElementById(tab.getAttribute('aria-controls')).hidden = !active;
    });
    const panel = document.getElementById(selected.getAttribute('aria-controls'));
    mount(panel);
    requestAnimationFrame(() => observers.get(panel.id)?.resize());
    if (focus) {
      selected.focus({preventScroll:true});
      window.scrollTo({top:0, behavior:'instant'});
    }
    if (location.hash !== '#' + selected.dataset.setting) location.hash = selected.dataset.setting;
    document.title = selected.dataset.title + ' · Committee debate';
  }
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => activate(tab.dataset.setting, true));
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      activate(tabs[next].dataset.setting, true);
    });
  });
  window.addEventListener('hashchange', () => activate(location.hash.slice(1)));
  activate(location.hash.slice(1));
})();
"""


def render(docs: Path, output: Path) -> str:
    tabs, panels = [], []
    complete = missing = 0
    for spec in settings():
        sid = spec["id"]
        title = f"{spec['title']} · {spec['members']} members"
        report = load_report(spec, docs, output)
        active = sid == DEFAULT_TAB
        default_attr = ' data-default="true"' if active else ""
        complete += report is not None
        gaps = bool(report and report["has_missing"])
        missing += gaps
        status = "Finished · gaps" if gaps else "Completed" if report else "Not run"
        status_key = "completed-with-gaps" if gaps else "completed" if report else "not-run"
        tabs.append(
            f'<button class="tab" role="tab" id="tab-{sid}" aria-controls="panel-{sid}" '
            f'aria-selected="{str(active).lower()}" tabindex="{0 if active else -1}" '
            f'data-setting="{sid}" data-title="{html.escape(title)}" '
            f'data-status="{status_key}"'
            f"{default_attr}>"
            f'<span class="tab-title">{spec["title"]}</span>'
            f'<span class="tab-members">{spec["members"]} members</span>'
            f'<span class="tab-status">{status}</span></button>'
        )
        models = "".join(
            f"<span><b>{slot}</b> {html.escape(model)}</span>"
            for slot, model in zip("ABC", spec["models"], strict=True)
        )
        routing = (
            "Exclusive dyads AB, CA and BC drawn from the three slots above; each path uses two members."
            if spec["members"] == 2
            else "Three-member routing; members can change along a discussion path."
        )
        if spec["roster"] == "same_model":
            routing += " A, B and C are separate member identities using the same model."
        source_link = (
            f'<a class="source-link" target="_blank" rel="noopener" '
            f'href="{html.escape(report["href"])}">Open standalone report ↗</a>'
            if report
            else ""
        )
        scope = f'{report["questions"]} questions · Reported layers: ' if report else "Planned layers: "
        head = (
            f'<header class="panel-head"><div class="panel-title"><h2>{title}</h2>{source_link}</div>'
            f'<div class="models" aria-label="Model slots">{models}</div>'
            f'<p class="scope">{scope}{spec["layers"]}</p><p class="routing">{routing}</p>'
        )
        if report and report["quality_note"]:
            head += f'<p class="scope">{html.escape(report["quality_note"])}</p>'
        if gaps:
            head += '<p class="missing-note">Run finished with missing results after bounded technical retries. Unavailable comparisons are excluded, not counted as ties or losses. See the report for counts and failure logs.</p>'
        head += "</header>"
        if report:
            # Escape '<' so even report text containing '</script>' stays inert.
            payload = json.dumps(report, ensure_ascii=False).replace("<", "\\u003c")
            content = (
                '<p class="loading" role="status">Loading the saved report…</p>'
                f'<iframe title="{title} — complete results" data-snapshot="snapshot-{sid}" '
                'sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"></iframe>'
                f'<script type="application/json" id="snapshot-{sid}">{payload}</script>'
            )
        else:
            note = (
                "Local C/D are outside the three-member analysis plan: the preceding label need not target "
                "the next receiver. E still includes member outcomes and final-answer comparison."
                if spec["members"] == 3
                else "C tracks position changes within the exclusive dyads. Probability readouts (D) are reserved "
                "for the mixed-family, two-member setting."
            )
            content = (
                '<div class="pending"><h3>Not run yet</h3>'
                "<p>This tab reserves the setting. No current-protocol results are available here; "
                "older pilots and dry runs are not substituted.</p>"
                f'<p class="scope-note">{note}</p></div>'
            )
        panels.append(
            f'<section class="panel" role="tabpanel" id="panel-{sid}" aria-labelledby="tab-{sid}" '
            f'tabindex="0" data-layers="{spec["layers"]}"{" hidden" if not active else ""}>'
            f"{head}{content}</section>"
        )
    fallback_links = " · ".join(
        f'<a href="{relative_url(docs / name, output)}">{TITLES[roster]} · {members} members</a>'
        for (roster, members), (name, _) in REPORTS.items()
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Committee debate · experiment reports</title><style>{CSS}</style></head><body>"
        '<header class="page-head"><div class="eyebrow">Experiment reports</div>'
        "<h1>Committee debate</h1><p>Three model configurations × two routing settings. "
        "Turn-level tone throughout; original and alternate tone assignments stay inside each report.</p>"
        f'<p class="status-count">{complete} finished · {missing} with missing results · {6 - complete} not run</p></header>'
        '<div class="tabs-bar"><div class="tablist" role="tablist" aria-label="Experiment setting">'
        f'{"".join(tabs)}</div></div><noscript><p>Enable JavaScript to switch tabs, or open the reports: '
        f'{fallback_links}</p></noscript><main class="workspace">{"".join(panels)}</main>'
        '<footer class="footer">Results stay separate across settings. The two- and three-member designs '
        "use different discussion budgets, so their contrast does not isolate committee size. "
        "These tabs embed saved reports; opening this page does not run experiments.</footer>"
        f"<script>{JS}</script></body></html>\n"
    )


def publish(docs: Path = ROOT / "docs", output: Path | None = None) -> Path:
    docs = docs.resolve()
    output = (output or docs / "committee-experiment-dashboard.html").resolve()
    sources = {p.resolve() for name, _ in REPORTS.values() for p in (docs / name, (docs / name).with_suffix(".json"))}
    if output in sources:
        raise ValueError("Dashboard must not overwrite a source report")
    document = render(docs, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=ROOT / "docs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(publish(args.docs, args.output))


if __name__ == "__main__":
    main()
