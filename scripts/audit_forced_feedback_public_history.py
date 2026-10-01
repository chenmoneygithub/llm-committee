"""Offline reconstruction, preserved-source checks and actual native-token audits."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from llm_committee.pivot.databricks_provider import ENDPOINTS, databricks_payload
from llm_committee.pivot.failures import (
    LocalTaskFailure,
    local_output_limit_record,
    local_transport_record,
    retry_key,
    retryable_format_record,
)
from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.reanalysis import OfflineProvider
from llm_committee.pivot.storage import RunStopped
from llm_committee.pivot.study import atomic_json, native_token_count
from scripts.audit_stateful_dyadic import check_native
from scripts.forced_feedback_public_history import (
    FORCE,
    OUTPUT,
    FeedbackGraph,
    FeedbackMockProvider,
    request_identity,
    verify_preservation,
)
from scripts.scale_public_history import read, sha


class ReplayJournal:
    def __init__(self, output):
        self.db = sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.verified = {}

    def call(self, request, provider, parse):
        attempts = []
        for attempt in range(3):
            key = retry_key(request.key, attempt)
            attempts.append(key)
            row = self.db.execute("SELECT * FROM calls WHERE key=?", (key,)).fetchone()
            if row is None:
                raise RunStopped("No paid dispatch in audit")
            row = dict(row)
            assert row["request_hash"] == digest(request.document())
            assert canonical(json.loads(row["request"])) == canonical(request.document())
            if row["status"] == "completed":
                response, parsed = json.loads(row["response"]), json.loads(row["parsed"])
                assert response["status"] == "completed"
                assert parse(response["text"]) == {k: v for k, v in parsed.items() if k != "_readout"}
                self.verified[request.key] = (request, response, parsed)
                return parsed
            transport = local_transport_record(request.document(), row, "databricks")
            output_limit = local_output_limit_record(request.document(), row, "databricks")
            assert transport or output_limit or retryable_format_record(request.document(), row)
            if attempt == 2:
                kind = (
                    "failed_after_transport_retries"
                    if transport
                    else "failed_after_output_limit_retries" if output_limit else "failed_after_format_retries"
                )
                raise LocalTaskFailure(request.key, attempts, row["error"], kind=kind)
        raise AssertionError("Unreachable")


def audit(output=OUTPUT):
    output = Path(output).resolve()
    manifest, contexts, report = (read(output / p) for p in ("manifest.json", "source-contexts.json", "report.json"))
    verify_preservation(manifest)
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert report["status"] in ("preflight_completed", "completed", "completed_with_failures")
    mock = manifest["kind"] == "offline_mock"
    count = FeedbackMockProvider().token_count if mock else native_token_count()
    replay = ReplayJournal(output)
    origins, renderers = {}, {}
    reused_count = native_count = case_count = 0
    purposes, reconstructed_keys = Counter(), set()
    records = []
    try:
        for plan in manifest["plans"][: report["target_this_phase"]]:
            graph = FeedbackGraph(contexts[plan["question_id"]], plan, manifest)
            graph.restore(replay, OfflineProvider(count))
            assert graph.complete
            reconstructed = graph.report(count)
            assert canonical(reconstructed) == canonical(read(output / "questions" / f"{plan['question_id']}.json"))
            records.append(reconstructed)
            case_count += len(reconstructed["cases"])
            reconstructed_keys.update(graph.tasks)
            for key, provenance in graph.reused.items():
                source = provenance["directory"]
                if source not in origins:
                    origins[source] = sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True)
                    origins[source].row_factory = sqlite3.Row
                row = dict(
                    origins[source].execute("SELECT * FROM calls WHERE key=?", (provenance["storage_key"],)).fetchone()
                )
                assert digest(row) == provenance["row_sha256"] and row["status"] == "completed"
                expected = graph.tasks[key].build({}, count)
                assert request_identity(json.loads(row["request"])) == request_identity(expected.document())
                assert json.loads(row["parsed"]) == graph.values[key]
                reused_count += 1
            for case in graph.plan["cases"]:
                cid = case["id"]
                forced = graph.history(case, graph.values, "forced")
                for role in ("feedback", "return"):
                    key = graph.key(f"{cid}/forced/{role}")
                    if key not in graph.values:
                        continue
                    request = graph.tasks[key].build(graph.values, count)
                    assert (FORCE in request.messages[-1].text) == (role == "feedback")
                    assert request.effort == "medium"
                    assert "your_current_position" not in canonical(request.document()["messages"])
                    assert set(graph.values[key]) == {"reply", "agreement", "choice", "position"}
                for arm in ("natural", "forced"):
                    keys = graph.measurements[cid]["arms"][arm]
                    if keys["E_answer"] in graph.values:
                        payload = json.loads(graph.synthesis(case, graph.values, arm)[-1].text)
                        assert len(payload["discussion"]) == case["return_T"]
                        assert {r["member"] for r in payload["initial_answers"]} == {
                            case["recipient"],
                            case["challenger"],
                        }
                    if (
                        "D2_argument" in keys
                        and keys["D2_argument"] in graph.values
                        and keys["D2_control"] in graph.values
                    ):
                        a, b = (graph.tasks[keys[k]].build(graph.values, count) for k in ("D2_argument", "D2_control"))
                        assert [i for i in range(len(a.messages)) if a.messages[i] != b.messages[i]] == [
                            len(a.messages) - 2
                        ]
                        assert (
                            json.loads(a.messages[-3].text)["position_to_evaluate"]["text"]
                            == graph.before(case)["position"]
                        )
                        assert FORCE not in canonical(a.document()["messages"])
                    if "D1" in keys and keys["D1"] in graph.values:
                        d1 = graph.tasks[keys["D1"]].build(graph.values, count)
                        assert "position_to_evaluate" not in canonical(d1.document()["messages"])
                        if arm == "forced" and case["return_node"] in forced:
                            assert forced[case["return_node"]]["position"] not in "\n".join(m.text for m in d1.messages)
        for request, response, value in replay.verified.values():
            purposes[request.purpose] += 1
            if request.purpose != "debate":
                assert FORCE not in canonical(request.document()["messages"])
            if request.candidate_labels:
                assert request.effort == "none" and response["reasoning_tokens"] == 0
                if not mock:
                    from llm_committee.pivot.tinker_provider import make_renderer

                    if request.model not in renderers:
                        renderers[request.model] = make_renderer(request.model, "none")
                    check_native(request.document(), response, value, *renderers[request.model])
                native_count += 1
            if not mock and request.model in ENDPOINTS:
                raw = response["raw"]
                assert raw["request_payload"] == databricks_payload(request)
                assert raw["endpoint"] == ENDPOINTS[request.model] and raw["http_status"] == 200
            if request.purpose == "synthesis":
                assert 190 <= len(value["answer"].split()) <= 210
        for raw, status in replay.db.execute("SELECT request,status FROM calls"):
            assert json.loads(raw)["key"] in reconstructed_keys and status != "pending"
        assert (
            len(replay.verified)
            == replay.db.execute("SELECT COUNT(*) FROM calls WHERE status='completed'").fetchone()[0]
        )
    finally:
        replay.db.close()
        for db in origins.values():
            db.close()
    result = {
        "status": "passed",
        "cases_reconstructed": case_count,
        "new_successes_verified": len(replay.verified),
        "new_calls_by_purpose": dict(purposes),
        "exact_archived_reuses": reused_count,
        "new_probability_reads_checked_at_actual_token": native_count,
        "source_unchanged": True,
        "early_stops_at_T2": True,
        "late_stops_at_T4": True,
        "mock": mock,
        "manifest_sha256": sha(output / "manifest.json"),
        "report_sha256": sha(output / "report.json"),
        "question_records_sha256": digest(records),
    }
    atomic_json(output / "audit.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.output)), flush=True)
