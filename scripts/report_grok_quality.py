"""Add the independent judge beside Gemini in the existing mixed-triadic E table."""

import argparse
import shutil
from pathlib import Path

from bs4 import BeautifulSoup

from llm_committee.pivot.study import atomic_json
from scripts.grok_quality_check import OUTPUT as JUDGE_OUTPUT
from scripts.grok_quality_check import verify_source
from scripts.scale_public_history import ROOT, read, sha

REPORT = ROOT / "docs/turn-tone-triadic-mixed-family-50-public-history.html"


def percent_interval(estimate):
    if estimate["mean"] is None:
        return "Unavailable"
    low, high = estimate["ci"]
    return f"{100 * estimate['mean']:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def attach(report=REPORT, judge_output=JUDGE_OUTPUT):
    report, judge_output = Path(report), Path(judge_output)
    judged, manifest = read(judge_output / "report.json"), read(judge_output / "manifest.json")
    if judged["status"] != "completed":
        raise ValueError("Do not publish an incomplete second-judge cohort")
    verify_source(manifest)
    data = read(report.with_suffix(".json"))
    if data["manifest_sha256"] != sha(Path(manifest["source_directory"]) / "manifest.json"):
        raise ValueError("Report and Grok refer to different cohorts")
    for suffix in (".html", ".json"):
        backup = judge_output / f"report-before-grok{suffix}"
        if not backup.exists():
            shutil.copyfile(report.with_suffix(suffix), backup)
    soup = BeautifulSoup(report.read_text(), "html.parser")
    section = soup.select_one("#e2")
    table = section.select_one("table")
    header = table.select_one("thead tr")
    gemini = next(th for th in header.select("th") if "Debate wins" in th.get_text() and "Grok" not in th.get_text())
    gemini.string = "Gemini · Debate wins (%) [95% interval]"
    if not header.select_one('[data-judge="grok"]'):
        th = soup.new_tag("th", attrs={"data-judge": "grok"})
        th.string = "Grok · Debate wins (%) [95% interval]"
        header.append(th)
    for tr, old, new in zip(table.select("tbody tr"), data["E2"], judged["summary"], strict=True):
        assert old["assignment"] == new["assignment"]
        assert old["estimate"] == new["gemini_same_pairs_estimate"]
        cell = tr.select_one('[data-judge="grok"]')
        if cell is None:
            cell = soup.new_tag("td", attrs={"data-judge": "grok"})
            tr.append(cell)
        cell.string = percent_interval(new["grok_estimate"])
    for node in soup.select("#independent-grok-note"):
        node.decompose()
    note = soup.new_tag("div", id="independent-grok-note")
    lines = [
        "Grok 4.6 is an independent second judge of these exact same 100 answer pairs (50 questions, two continuations), with the same rubric and both answer orders: 200 calls. No answer was regenerated. The three outcome-count columns above describe Gemini; the last two columns compare judge preference rates, not percentage quality improvement.",
        "Each judge's 95% interval resamples questions. The two continuations share questions and prefixes; neither the 100 pairs nor 200 decisions are independent questions. Judges are reported separately, not combined into a majority vote.",
    ]
    for row in judged["summary"]:
        lines.append(
            f"{row['assignment'].title()}: Grok prefers debate in both orders for {row['grok_both']}/{row['complete_pairs']} pairs, baseline in both for {row['baseline_both']}, and changes its preference with order for {row['order_inconsistent']}. "
            f"The judges select the same answer in {row['same_order_decisions']}/{row['compared_decisions']} matched-order decisions; their two-order verdicts match on {row['same_pair_verdict']}/{row['complete_pairs']} pairs. Agreement between judges is not human-validated accuracy."
        )
    lines.append(
        "The constructed diagnostics below belong to Gemini only; Grok has not been evaluated on that battery. Human review uses a separate outcome-blind sample of 20 distinct questions, one pair per question; no human results have been collected yet. Humans may report a tie or inability to judge, unlike the forced-choice model judges."
    )
    for text in lines:
        p = soup.new_tag("p")
        p.string = text
        note.append(p)
    table.insert_after(note)
    for heading in section.select("h3"):
        if heading.get_text().startswith("Small judge diagnostics"):
            heading.string = "Small judge diagnostics · Gemini only"
    report.write_text(str(soup))
    data.setdefault("additional_E2_judges", {})["grok-4-6"] = judged
    atomic_json(report.with_suffix(".json"), data)
    return report.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument("--judge-output", type=Path, default=JUDGE_OUTPUT)
    args = parser.parse_args()
    print(attach(args.report, args.judge_output))
    from scripts.report_experiment_dashboard import publish

    print(publish())
