"""Bounded, resumable three-judge screening; never generates committee debates.

The frozen plan supplies original questions, rubric, exact endpoints and a fixed
candidate order. SQLite retains every request/response, including failures.
No credentials are stored; no uncertain or invalid call is automatically retried.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def make_payload(plan, candidate, judge):
    payload = {
        "messages": [
            {"role": "system", "content": plan["screening_prompt"]},
            {
                "role": "user",
                "content": "Question:\n" + candidate["question"]
                + "\n\nOriginal answer options:\n" + json.dumps(candidate["options"], ensure_ascii=False)
                + '\n\nReturn only one JSON object with two fields: "question_class" '
                + '(exactly "debatable_opinion", "personal_report", or "unclear") and '
                + '"reason" (one brief sentence explaining the classification). Do not answer the survey question.',
            },
        ],
        "max_tokens": plan["max_output_tokens"],
        "stream": False,
    }
    if "reasoning_effort" in judge:
        payload["reasoning_effort"] = judge["reasoning_effort"]
    return payload


def parse_verdict(raw):
    choices = raw.get("choices") or []
    if len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError("Incomplete or non-single response")
    message = choices[0].get("message", {})
    if message.get("role") != "assistant" or message.get("tool_calls"):
        raise ValueError("Unexpected message")
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if p.get("type") == "text")
    if not isinstance(content, str):
        raise ValueError("No final text")
    content = content.strip()
    if content.startswith("```json\n") and content.endswith("```"):
        content = content[8:-3].strip()
    value = json.loads(content)
    if set(value) != {"question_class", "reason"}:
        raise ValueError("Unexpected verdict fields")
    if value["question_class"] not in {"debatable_opinion", "personal_report", "unclear"}:
        raise ValueError("Unknown question class")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Missing classification reason")
    return value


def usage_cost(raw, judge):
    usage = raw.get("usage") or {}
    prompt, total, visible = (usage.get(k) for k in ("prompt_tokens", "total_tokens", "completion_tokens"))
    if any(type(v) is not int or v < 0 for v in (prompt, total, visible)):
        raise ValueError("Missing usage; retain reservation")
    output = total - prompt
    if output < visible:
        raise ValueError("Inconsistent usage; retain reservation")
    # Count all non-input tokens, including separately reported Gemini reasoning.
    # Charge input at uncached rate; no promotional discount is assumed.
    return 1.1 * (prompt * judge["input_usd_per_million"] + output * judge["output_usd_per_million"]) / 1e6


def reservation(payload, judge):
    # Byte-length plus framing allowance conservatively bounds these short inputs.
    input_bound = len(canonical(payload).encode("utf-8")) + 1024
    # Gemini may report thinking outside visible completion_tokens; hold its full
    # output window rather than assuming max_tokens covers every billed token.
    output_bound = 65536 if judge["model"].startswith("gemini-") else payload["max_tokens"]
    return 1.1 * (input_bound * judge["input_usd_per_million"] + output_bound * judge["output_usd_per_million"]) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--limit", type=int, help="Stop after this many candidate decisions, including cached ones")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if not args.live:
        raise SystemExit("Explicit --live is required")
    if [j["model"] for j in plan["jury"]] != ["gpt-5.5", "opus-4.8", "gemini-3.5-flash"]:
        raise ValueError("Original screening jury required")
    if not 1 <= plan["requested_additions"] <= 20 or not 0 < plan["estimated_cost_cap_usd"] <= 3:
        raise ValueError("This runner is limited to 20 additions and a $3 estimated-cost cap")
    if not 1 <= plan["max_screened_candidates"] <= 40 or plan["max_parallel_calls"] != 3:
        raise ValueError("Unexpected screening scope")
    candidates = plan["candidates"]
    order = [q["candidate_order_index"] for q in candidates]
    if order != sorted(set(order)):
        raise ValueError("Candidate order must be fixed and unique")
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(args.journal, isolation_level=None)
    db.execute("PRAGMA synchronous=FULL")
    db.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("""CREATE TABLE IF NOT EXISTS calls (
        candidate INTEGER, model TEXT, endpoint TEXT, payload TEXT,
        status TEXT, charge REAL, raw TEXT, verdict TEXT, error TEXT,
        PRIMARY KEY(candidate,model))""")
    identity = hashlib.sha256(canonical(plan).encode()).hexdigest()
    prior = db.execute("SELECT value FROM metadata WHERE key='plan_sha256'").fetchone()
    if prior and prior[0] != identity:
        raise ValueError("Plan changed; cannot reuse this journal")
    db.execute("INSERT OR IGNORE INTO metadata VALUES('plan_sha256',?)", (identity,))

    import httpx
    from databricks.sdk.config import Config

    config = Config(profile="un", http_timeout_seconds=20, retry_timeout_seconds=1)
    config.authenticate()
    accepted, processed = [], 0
    with httpx.Client(timeout=90, follow_redirects=False, transport=httpx.HTTPTransport(retries=0)) as client:
        for candidate in candidates:
            if not candidate["context_review"]["passed"]:
                continue
            if processed >= plan["max_screened_candidates"] or (args.limit and processed >= args.limit):
                break
            index, pending = candidate["candidate_order_index"], []
            for judge in plan["jury"]:
                payload = make_payload(plan, candidate, judge)
                old = db.execute("SELECT payload,status FROM calls WHERE candidate=? AND model=?", (index, judge["model"])).fetchone()
                if old:
                    if old != (canonical(payload), "completed"):
                        raise RuntimeError("Changed, failed or uncertain call exists; no automatic retry")
                else:
                    pending.append((judge, payload, reservation(payload, judge)))
            db.execute("BEGIN IMMEDIATE")
            try:
                charged = db.execute("SELECT COALESCE(SUM(charge),0) FROM calls").fetchone()[0]
                if charged + sum(p[2] for p in pending) > plan["estimated_cost_cap_usd"]:
                    raise RuntimeError("Estimated-cost cap cannot cover next candidate")
                for judge, payload, hold in pending:
                    db.execute("INSERT INTO calls VALUES(?,?,?,?,?,?,NULL,NULL,NULL)",
                               (index, judge["model"], judge["endpoint"], canonical(payload), "pending", hold))
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise
            headers = {**config.authenticate(), "Content-Type": "application/json"}
            if config.workspace_id:
                headers["X-Databricks-Workspace-Id"] = str(config.workspace_id)
            failed = False
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(client.post,
                                    config.host.rstrip("/") + f"/serving-endpoints/{j['endpoint']}/invocations",
                                    headers=headers, json=p): (j, hold)
                    for j, p, hold in pending
                }
                for future in as_completed(futures):
                    judge, hold = futures[future]
                    key = (index, judge["model"])
                    try:
                        response = future.result()
                        raw = response.json()
                        db.execute("UPDATE calls SET raw=?,status='received' WHERE candidate=? AND model=?",
                                   (canonical({"http_status": response.status_code, "response": raw}), *key))
                        if response.status_code != 200:
                            raise ValueError(f"HTTP {response.status_code}")
                        cost = usage_cost(raw, judge)
                        if cost > hold:
                            raise ValueError("Usage exceeded estimate reservation")
                        db.execute("UPDATE calls SET charge=? WHERE candidate=? AND model=?", (cost, *key))
                        verdict = parse_verdict(raw)
                        db.execute("UPDATE calls SET verdict=?,status='completed' WHERE candidate=? AND model=?",
                                   (canonical(verdict), *key))
                    except Exception as exc:
                        failed = True
                        # Exception messages may contain provider URLs; only persist the type.
                        db.execute("UPDATE calls SET status='failed',error=? WHERE candidate=? AND model=?",
                                   (type(exc).__name__, *key))
            if failed:
                raise RuntimeError(f"Candidate {index}: incomplete judge evidence; inspect saved raw responses")
            verdicts = {model: json.loads(v) for model, v in db.execute(
                "SELECT model,verdict FROM calls WHERE candidate=? AND status='completed'", (index,))}
            if len(verdicts) != 3:
                raise RuntimeError("All three judge decisions are required")
            votes = sum(v["question_class"] == "debatable_opinion" for v in verdicts.values())
            if votes >= 2:
                accepted.append(index)
            processed += 1
            total = db.execute("SELECT SUM(charge) FROM calls").fetchone()[0]
            print(json.dumps({"candidate": index, "opinion_votes": votes,
                              "labels": {m: v["question_class"] for m, v in verdicts.items()},
                              "accepted_count": len(accepted), "estimated_usd": round(total, 6)}), flush=True)
            if len(accepted) == plan["requested_additions"]:
                break
    print(json.dumps({"accepted": accepted, "screened_candidates": processed,
                      "complete": len(accepted) == plan["requested_additions"]}), flush=True)
    db.close()


if __name__ == "__main__":
    main()
