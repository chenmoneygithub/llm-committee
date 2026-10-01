"""Independent Grok E2 judging of frozen mixed-family triadic answers only.

No debate, synthesis or calibration generation is dispatched. Existing Gemini
messages, rubric, answer order, schema, reasoning effort and token limit are
reused verbatim. Provider registration is process-local, leaving old code
snapshots and experiment journals untouched.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

from llm_committee.pivot import prompts
from llm_committee.pivot.databricks_provider import ENDPOINTS, DatabricksProvider
from llm_committee.pivot.failures import FORMAT_RETRY_POLICY, LocalTaskFailure
from llm_committee.pivot.models import Message, Request, canonical, digest
from llm_committee.pivot.providers import PRICES, Price
from llm_committee.pivot.results_report import question_estimate
from llm_committee.pivot.storage import Journal
from llm_committee.pivot.study import atomic_json
from scripts.scale_public_history import OUTPUT as EXTENSION
from scripts.scale_public_history import ROOT, freeze, read, sha

SOURCE = EXTENSION / "combined/mixed_family/triadic"
OUTPUT = ROOT / "runs/grok-mixed-triadic-E2-50-20260928"
MODEL = "grok-4-6"
ENDPOINT = "databricks-grok-4-6"
CAP_USD = 8.0
PRICE = {
    "input_usd_per_million": 2.5,
    "cached_usd_per_million": 0.6251,
    "output_usd_per_million": 7.5001,
    "regional_multiplier": 1.1,
    "source": "https://www.databricks.com/product/pricing/proprietary-foundation-model-serving",
    "checked": "2026-09-28",
    "note": "Listed 35.714/8.929/107.143 DBU per million at $0.07/DBU, rounded upward; includes 10% regional uplift, ignores promotional discounts. Estimate, not invoice.",
}


def register_provider():
    ENDPOINTS[MODEL] = ENDPOINT
    PRICES[MODEL] = Price(
        PRICE["input_usd_per_million"], PRICE["cached_usd_per_million"], PRICE["output_usd_per_million"], 500000
    )


def request_from(document):
    return Request(**{**document, "messages": tuple(Message(**m) for m in document["messages"])})


def prepare(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("The independent judge must use a separate journal")
    original = read(source / "manifest.json")
    assert original["protocol_version"] == "public-history-triadic-ABE-2026-09-28-v1"
    assert original["config"]["roster"] == "mixed_family" and len(original["plans"]) == 50
    tasks, files = [], [source / name for name in ("manifest.json", "report.json", "requests.sqlite3")]
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        saved = {
            json.loads(r)["key"]: (json.loads(r), json.loads(v))
            for r, v in db.execute("SELECT request,parsed FROM calls WHERE status='completed'")
        }
    for plan in original["plans"]:
        path = source / "questions" / f"{plan['question_id']}.json"
        files.append(path)
        record = read(path)
        for row in record["E2"]:
            assert row["status"] == "success" and row["debate_score"] is not None
            for order, judgment in enumerate(row["preferences"]["orders"]):
                source_key = f"{record['question_id']}/{original['protocol_version']}/mixed_family/{row['assignment']}/E2/order{order}"
                old, value = saved[source_key]
                assert old["purpose"] == "judge_e" and value == judgment["judgment"]
                data = json.loads(old["messages"][-1]["text"])
                side = judgment["target_side"]
                assert data[side] == row["debated"]["answer"]
                assert data["right" if side == "left" else "left"] == row["baseline"]["answer"]
                assert old["schema"] == prompts.E_SCHEMA and old["effort"] == "low" and old["max_output_tokens"] == 2048
                request = replace(request_from(old), model=MODEL, key=f"grok-E2/{source_key}")
                tasks.append(
                    {
                        "request": json.loads(canonical(request.document())),
                        "source_key": source_key,
                        "question_id": record["question_id"],
                        "assignment": row["assignment"],
                        "order": order,
                        "target_side": side,
                        "gemini_judgment": value,
                    }
                )
    assert len(tasks) == len({t["request"]["key"] for t in tasks}) == 200
    files_to_hash = [
        Path(__file__),
        ROOT / "llm_committee/pivot/prompts.py",
        ROOT / "llm_committee/pivot/storage.py",
        ROOT / "llm_committee/pivot/databricks_provider.py",
        ROOT / "llm_committee/pivot/providers.py",
    ]
    manifest = {
        "version": "grok-independent-E2-v1",
        "kind": "paid_external_judge",
        "source_directory": str(source),
        "source_sha256": {str(p.relative_to(source)): sha(p) for p in files},
        "implementation_sha256": {str(p.relative_to(ROOT)): sha(p) for p in files_to_hash},
        "model": MODEL,
        "endpoint": ENDPOINT,
        "price": PRICE,
        "cap_usd": CAP_USD,
        "config": {"closed_provider": "databricks"},
        "execution": {"failure_policy": FORMAT_RETRY_POLICY, "max_inflight": 4},
        "scope": "100 existing mixed-family triadic final-answer pairs, both orders; no other layers or generations",
        "question_count": 50,
        "planned_pairs": 100,
        "planned_calls": 200,
        "tasks_sha256": digest(tasks),
        "diagnostics": "Existing constructed judge diagnostics not rejudged by Grok",
    }
    freeze(output / "manifest.json", manifest)
    freeze(output / "tasks.json", tasks)
    return manifest, tasks


def verify_source(manifest):
    source = Path(manifest["source_directory"])
    for name, expected in manifest["source_sha256"].items():
        if sha(source / name) != expected:
            raise ValueError(f"Source changed: {name}")


def summarize(output):
    output = Path(output)
    manifest, tasks = read(output / "manifest.json"), read(output / "tasks.json")
    assert digest(tasks) == manifest["tasks_sha256"]
    verify_source(manifest)
    values, ledger, attempts = {}, [], []
    with sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True) as db:
        for request, value in db.execute("SELECT request,parsed FROM calls WHERE status='completed'"):
            key = json.loads(request)["key"]
            assert key not in values
            values[key] = json.loads(value)
        ledger = db.execute("SELECT status,COUNT(*),SUM(charge) FROM calls GROUP BY status").fetchall()
        attempts = [
            {"key": k, "status": s, "error": e}
            for k, s, e in db.execute("SELECT key,status,error FROM calls WHERE status!='completed'")
        ]
    pairs = {}
    for task in tasks:
        key = task["question_id"], task["assignment"]
        row = pairs.setdefault(key, {"question_id": key[0], "assignment": key[1], "orders": []})
        value = values.get(task["request"]["key"])
        row["orders"].append(
            {
                "order": task["order"],
                "target_side": task["target_side"],
                "grok": value,
                "gemini": task["gemini_judgment"],
            }
        )
    for row in pairs.values():
        row["grok_score"] = (
            sum(o["grok"]["preference"] == o["target_side"] for o in row["orders"]) / 2
            if all(o["grok"] for o in row["orders"])
            else None
        )
        row["gemini_score"] = sum(o["gemini"]["preference"] == o["target_side"] for o in row["orders"]) / 2
    summaries = []
    for arm in ("original", "alternate"):
        rows = [r for r in pairs.values() if r["assignment"] == arm]
        complete = [r for r in rows if r["grok_score"] is not None]
        summaries.append(
            {
                "assignment": arm,
                "planned_pairs": len(rows),
                "complete_pairs": len(complete),
                "grok_estimate": question_estimate(rows, "grok_score"),
                "gemini_same_pairs_estimate": question_estimate(complete, "gemini_score"),
                "grok_both": sum(r["grok_score"] == 1 for r in complete),
                "baseline_both": sum(r["grok_score"] == 0 for r in complete),
                "order_inconsistent": sum(r["grok_score"] == 0.5 for r in complete),
                "same_pair_verdict": sum(r["grok_score"] == r["gemini_score"] for r in complete),
                "same_order_decisions": sum(
                    o["grok"]["preference"] == o["gemini"]["preference"] for r in complete for o in r["orders"]
                ),
                "compared_decisions": 2 * len(complete),
            }
        )
    result = {
        "version": manifest["version"],
        "source_directory": str(output.resolve()),
        "completed_calls": len(values),
        "planned_calls": len(tasks),
        "rows": list(pairs.values()),
        "summary": summaries,
        "status": "completed" if len(values) == len(tasks) else "incomplete",
        "cost_accounting": {
            "received_response_estimate_usd": sum(
                cost for s, _, cost in ledger if s in ("completed", "invalid", "received")
            ),
            "unresolved_reservations_usd": sum(
                cost for s, _, cost in ledger if s not in ("completed", "invalid", "received")
            ),
            "price": PRICE,
        },
        "failed_attempts": attempts,
        "source_unchanged": True,
        "interpretation": "Single additional judge; blind binary preference, both orders. 0.5 means order-inconsistent, not a tie. Bootstrap questions, not 200 independent decisions. Judges kept separate; no human validation yet.",
    }
    atomic_json(output / "report.json", result)
    return result


def run(output=OUTPUT, limit=200, provider_factory=DatabricksProvider):
    output = Path(output)
    manifest, tasks = read(output / "manifest.json"), read(output / "tasks.json")
    assert digest(tasks) == manifest["tasks_sha256"]
    assert 1 <= limit <= len(tasks)
    verify_source(manifest)
    for path, expected in manifest["implementation_sha256"].items():
        assert sha(ROOT / path) == expected, "Frozen implementation changed"
    register_provider()
    with (output / "run.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal = Journal(output / "requests.sqlite3", manifest, CAP_USD)
        journal.close()
        stop = threading.Event()

        def execute(task):
            provider = provider_factory()
            journal = Journal(output / "requests.sqlite3", manifest, CAP_USD, should_stop=stop.is_set)
            try:
                value = journal.call(
                    request_from(task["request"]), provider, lambda s: prompts.parse_json(s, prompts.E_SCHEMA)
                )
                print(json.dumps({"event": "completed", "key": task["request"]["key"]}), flush=True)
                return value
            except LocalTaskFailure as exc:
                print(json.dumps({"event": "missing", **exc.document()}), flush=True)
            except BaseException:
                stop.set()
                raise
            finally:
                journal.close()
                provider.close()

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(execute, tasks[:limit]))
        report = summarize(output)
        print(json.dumps({k: report[k] for k in ("status", "completed_calls", "cost_accounting")}), flush=True)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    prepare(args.source, args.output)
    if args.live:
        run(args.output, args.limit)
    else:
        print(args.output.resolve())
