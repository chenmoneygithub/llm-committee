"""Explicit, network-free recomputation of saved runs under the zero-fill policy."""

from __future__ import annotations

import json
import tarfile
from collections.abc import Callable
from pathlib import Path

from .continuation import load_checkpoint
from .models import PilotConfig, Question
from .probabilities import TOPK_ZERO_FILL_POLICY
from .runner import manifest_for, run_pilot
from .storage import Journal


class OfflineProvider:
    def __init__(self, token_counter: Callable[[str, str], int]):
        self.token_count = token_counter

    def generate(self, request):
        raise AssertionError("Offline reanalysis must never generate or rescore")


def reanalyze_topk(source: Path, output: Path, *, token_counter: Callable[[str, str], int]) -> dict:
    """Replay exact saved requests; preserve legacy failures and costs as provenance.

    A changed prompt, input, runtime or absent response blocks replay. The journal also
    disables dispatch before reservation, independently of the no-generation provider.
    """
    source = source.resolve()
    if source == output.resolve():
        raise ValueError("Reanalysis requires a new directory; source is read-only")
    old = json.loads((source / "manifest.json").read_text())
    questions = tuple(Question.from_dict(q) for q in old["questions"])
    config = PilotConfig(**old["config"])
    manifest = manifest_for(questions, config, mock=old["kind"] == "offline_mock")
    descriptor, rows = load_checkpoint(source, manifest, reanalyze_topk_zero_fill=True)
    manifest["continuation"] = descriptor
    manifest["execution"].update(dispatch_enabled=False, mode="offline_topk_reanalysis")
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with tarfile.open(output / "implementation-start.tar.gz", "x:gz") as archive:
        for path in sorted(Path(__file__).parent.glob("*.py")):
            archive.add(path, arcname=f"llm_committee/pivot/{path.name}")
    journal = Journal(output / "requests.sqlite3", manifest, 1.0)
    try:
        journal.import_checkpoint(descriptor, rows)
        result = run_pilot(questions, config, manifest, journal, OfflineProvider(token_counter))
        if journal.charged_usd != 0:
            raise AssertionError("Reanalysis incurred a new charge")
        result["probability_readout"] = dict(TOPK_ZERO_FILL_POLICY)
        result["offline_reanalysis"] = {
            "api_calls": 0,
            "new_api_spend_usd": 0.0,
            "source_directory": str(source),
            "unused_supplemental_score_records": sum(
                json.loads(row["request"])["purpose"] == "candidate_scores" for row in rows
            ),
            "source_invalid_records": [
                {"key": row["key"], "status": row["status"], "error": row["error"]}
                for row in rows
                if row["status"] != "completed"
            ],
            "note": "New derived analysis, not repair of old scores. All source responses and failures are unchanged.",
        }
        (output / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (output / "calls.json").write_text(json.dumps(journal.audit(), ensure_ascii=False, indent=2) + "\n")
        return result
    finally:
        journal.close()
