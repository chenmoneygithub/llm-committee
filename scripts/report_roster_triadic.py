"""Publish a roster-labeled triadic report with the corresponding dyadic link.

Presentation-only wrapper: the frozen experiment implementation remains intact.
"""

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from bs4 import BeautifulSoup

from llm_committee.pivot.models import PilotConfig
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.triadic_report import full_case
from llm_committee.pivot.triadic_report import publish as publish_triadic

TITLES = {"same_family": "GPT same-family", "same_model": "Same-model", "mixed_family": "Mixed-family"}


def publish(source, output, *, dyadic_report):
    source, output, dyadic_report = (Path(p).resolve() for p in (source, output, dyadic_report))
    manifest = json.loads((source / "manifest.json").read_text())
    config = PilotConfig(**manifest["config"])
    publish_triadic(source, output)
    summary = json.loads(output.with_suffix(".json").read_text())
    soup = BeautifulSoup(output.read_text(), "html.parser")
    for node in (soup.title, soup.h1):
        node.string = f"{TITLES[config.roster]} · {node.get_text()}"
    link = soup.find("a", href="turn-tone-dyadic-public-history-2026-09-27.html")
    assert link is not None
    link["href"] = quote(Path(os.path.relpath(dyadic_report, output.parent)).as_posix(), safe="/")
    link.string = "Same roster · two-member report"
    if config.roster == "same_model":
        members = [f"{chr(65 + m)} (member {m}) · {model}" for m, model in enumerate(config.members)]
        note = soup.new_tag("p")
        note.string = (
            "A, B and C are three separate member identities using GPT-5.6 Terra. "
            "Each has its own initial answer and branch-local participation history. "
            "Numeric routing IDs 0, 1 and 2 correspond to A, B and C."
        )
        soup.h1.insert_after(note)
        for row, group in zip(soup.select("#e1 tbody tr"), summary["E1"], strict=True):
            row.select_one("td").string = members[group["member"]]
        records = [
            json.loads(path.read_text())
            for plan in manifest["plans"]
            if (path := source / "questions" / f"{plan['question_id']}.json").exists()
        ]
        for case, record in zip(soup.select("#cases .question-case"), records, strict=True):
            rendered = BeautifulSoup(full_case(record, members), "html.parser")
            case.replace_with(rendered.select_one(".question-case"))
    output.write_text(str(soup))
    summary.update(roster=config.roster, models=list(config.members))
    atomic_json(output.with_suffix(".json"), summary)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dyadic-report", required=True, type=Path)
    args = parser.parse_args()
    print(publish(args.source, args.output, dyadic_report=args.dyadic_report))
