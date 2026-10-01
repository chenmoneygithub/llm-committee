"""Isolated low-reasoning diagnostic for the six exhausted HLE Qwen requests."""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime

from llm_committee.pivot.models import Message, Request, canonical, digest
from llm_committee.pivot.prompts import parse_json
from llm_committee.pivot.storage import Journal, RunBlocked
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import freeze
from llm_committee.pivot.tinker_provider import TinkerProvider, make_renderer, message_content
from scripts.hle_diamond_study import RUN as SOURCE
from scripts.hle_diamond_study import implementation_hash, read, sha

MODEL = "Qwen/Qwen3.8-27B"
VERSION = "hle-qwen-low-retry-20260929-v1"
RUN = SOURCE.parent.parent / "hle-diamond-qwen-low-retry-20260929"
PAGE = RUN / "hle-qwen-low-retry.html"


def low_request(document):
    assert document["model"] == MODEL and document["effort"] == "medium"
    assert document["purpose"] in ("initial", "debate") and not document["candidate_labels"]
    request = Request(
        **{
            **document,
            "key": f"{VERSION}/{digest(document)[:20]}",
            "effort": "low",
            "messages": tuple(Message(**m) for m in document["messages"]),
            "candidate_labels": tuple(document["candidate_labels"]),
        }
    )
    ignored = {"key", "effort"}
    assert canonical({k: v for k, v in document.items() if k not in ignored}) == canonical(
        {k: v for k, v in request.document().items() if k not in ignored}
    )
    return request


def sources(directory=SOURCE):
    progress = read(directory / "progress.json")
    contexts = read(directory / "source-contexts.json")
    result = []
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        for key, failure in sorted(progress["task_failures"].items()):
            assert failure["model"] == MODEL and failure["reason"] == "Response status: incomplete"
            attempts = [
                dict(db.execute("SELECT * FROM calls WHERE key=?", (k,)).fetchone()) for k in failure["attempt_keys"]
            ]
            documents = [json.loads(r["request"]) for r in attempts]
            assert all(d == documents[0] for d in documents)
            assert len(attempts) == 3 and all(r["status"] == "invalid" for r in attempts)
            last = json.loads(attempts[-1]["response"])
            assert last["raw"]["stop_reason"] == "length"
            qid = key.split("/", 1)[0]
            result.append(
                {
                    "source_key": key,
                    "source_attempts_sha256": digest(attempts),
                    "source_attempt_count": len(attempts),
                    "request": documents[0],
                    "reference_answer": contexts[qid]["answer_letter"],
                    "previous_output_tokens": last["output_tokens"],
                    "previous_stop_reason": last["raw"]["stop_reason"],
                }
            )
    return result


class TimedProvider:
    def __init__(self, provider):
        self.provider = provider

    def generate(self, request):
        start = time.monotonic()
        started = datetime.now(UTC).isoformat()
        response = self.provider.generate(request)
        return replace(
            response,
            raw={
                **(response.raw or {}),
                "diagnostic_started_at_utc": started,
                "diagnostic_elapsed_seconds": time.monotonic() - start,
            },
        )


def execute_case(source, provider, manifest, directory):
    request = low_request(source["request"])
    journal = Journal(directory / "requests.sqlite3", manifest, None)
    try:
        try:
            journal.call(request, provider, lambda text: parse_json(text, request.schema))
        except RunBlocked:
            pass  # Preserve the failed attempt, never silently retry in this diagnostic.
        row = journal.db.execute(
            "SELECT status,response,parsed,charge,error FROM calls WHERE key=?", (request.key,)
        ).fetchone()
        if row is None:
            raise RuntimeError("Diagnostic failed before journaling; inspect before retrying")
        status, response, parsed, charge, error = row
        response, parsed = json.loads(response) if response else None, json.loads(parsed) if parsed else None
        result = {
            **source,
            "low_request": request.document(),
            "status": status,
            "response": response,
            "parsed": parsed,
            "charge_or_reservation_usd": charge,
            "error": error,
            "correct": parsed["choice"] == source["reference_answer"] if parsed else None,
        }
        print(
            canonical(
                {
                    "key": request.key,
                    "status": status,
                    "output_tokens": response["output_tokens"] if response else None,
                    "seconds": (response["raw"].get("diagnostic_elapsed_seconds") if response else None),
                }
            ),
            flush=True,
        )
        return result
    finally:
        journal.close()


def native_audit(results):
    tokenizer, low = make_renderer(MODEL, "low")
    _, medium = make_renderer(MODEL, "medium")
    checked = 0
    for case in results:
        response = case["response"]
        if not response:
            continue
        messages = [
            {"role": "system" if m["role"] == "developer" else m["role"], "content": m["text"]}
            for m in case["request"]["messages"]
        ]
        low_ids = low.build_generation_prompt(messages).to_ints()
        assert low_ids != medium.build_generation_prompt(messages).to_ints()
        assert low_ids == response["raw"]["prompt_token_ids"]
        case["effective_low_prompt"] = tokenizer.decode(low_ids, skip_special_tokens=False)
        if case["status"] == "completed":
            parsed, termination = low.parse_response(response["raw"]["sampled_token_ids"])
            visible, thinking = message_content(parsed)
            assert termination.is_clean and response["raw"]["stop_reason"] == "stop"
            assert visible == response["text"]
            assert parse_json(visible, case["request"]["schema"]) == case["parsed"]
            assert response["reasoning_tokens"] == len(tokenizer.encode(thinking, add_special_tokens=False))
        checked += 1
    return {"native_prompts_checked": checked, "only_effort_changed": True, "original_study_untouched": True}


def render(result):
    def esc(x):
        return html.escape(str(x))

    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>HLE · Qwen low diagnostic</title>',
        "<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#24343e}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ccd5dd;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:16px 0;padding:12px;border:1px solid #ccd5dd}</style>",
        "<h1>Qwen：medium → low，独立失败请求诊断</h1>",
        "<p>私有报告，包含受限 benchmark 内容，请勿公开。只取原实验用完两次重试仍失败的全部 6 个请求，每个新增一次 low 生成。"
        "原题、历史、tone、schema、temperature 和 16,384-token 上限不变；不生成后续轨迹，不修改或补入原实验统计。"
        "这不是随机样本，没有新增 medium 对照，不能据此估计整体准确率或单独识别 reasoning 强度的因果作用。</p>",
        f'<p>完整可解析：{result["completed"]}/{len(result["cases"])}；其中答案正确：{result["correct"]}/{result["completed"]}。'
        f'收到响应的费用估算 US${result["received_response_estimate_usd"]:.4f}；未知费用预留 US${result["unresolved_reservations_usd"]:.4f}。</p>',
        "<table><thead><tr><th>案例</th><th>任务</th><th>结果</th><th>耗时</th><th>全部输出 tokens</th><th>选择 / 答案键</th></tr></thead><tbody>",
    ]
    for i, case in enumerate(result["cases"], 1):
        response, parsed = case["response"], case["parsed"]
        seconds = response["raw"].get("diagnostic_elapsed_seconds") if response else None
        values = [
            i,
            case["request"]["purpose"],
            case["status"],
            f"{seconds:.1f}s" if seconds is not None else "—",
            response["output_tokens"] if response else "—",
            f'{parsed["choice"] if parsed else "—"} / {case["reference_answer"]}',
        ]
        parts.append("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in values) + "</tr>")
    parts.append(
        "</tbody></table><p>旧请求最终一次均在 16,384 tokens 截断。新输出长度包括推理与最终答案；完成不等于答对。</p>"
    )
    for i, case in enumerate(result["cases"], 1):
        parts.append(f'<details><summary>案例 {i}：完整请求及最终输出</summary><p>{esc(case["source_key"])}</p>')
        parts.append("<p>low 在渲染时额外加入简短推理指令；以下原始消息不变。</p>")
        for m in case["request"]["messages"]:
            parts.append(f'<h3>{esc(m["role"])}</h3><pre>{esc(m["text"])}</pre>')
        if case.get("effective_low_prompt"):
            parts.append(
                "<details><summary>实际渲染后的完整输入（含 low 指令和模型模板）</summary>"
                f'<pre>{esc(case["effective_low_prompt"])}</pre></details>'
            )
        parts.append(f'<pre>{esc(json.dumps(case["parsed"], ensure_ascii=False, indent=2))}</pre>')
        if case["error"]:
            parts.append(f'<p>{esc(case["error"])}</p>')
        parts.append("</details>")
    return "".join(parts) + "</html>"


def saved_results(cases, directory):
    results = []
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        for case in cases:
            request = low_request(case["request"])
            row = db.execute(
                "SELECT request,status,response,parsed,charge,error FROM calls WHERE key=?", (request.key,)
            ).fetchone()
            if row is None:
                raise ValueError("Missing saved request; report-only never dispatches")
            document, status, response, parsed, charge, error = row
            assert canonical(json.loads(document)) == canonical(request.document())
            response = json.loads(response) if response else None
            parsed = json.loads(parsed) if parsed else None
            results.append(
                {
                    **case,
                    "low_request": request.document(),
                    "status": status,
                    "response": response,
                    "parsed": parsed,
                    "charge_or_reservation_usd": charge,
                    "error": error,
                    "correct": parsed["choice"] == case["reference_answer"] if parsed else None,
                }
            )
    return results


def write_report(results, manifest):
    report = {
        "cases": results,
        "completed": sum(c["status"] == "completed" for c in results),
        "correct": sum(c["correct"] is True for c in results),
        "received_response_estimate_usd": sum(
            c["charge_or_reservation_usd"] for c in results if c["response"] and c["status"] != "billing_unknown"
        ),
        "unresolved_reservations_usd": sum(
            c["charge_or_reservation_usd"] for c in results if not c["response"] or c["status"] == "billing_unknown"
        ),
        "audit": native_audit(results),
        "generation_script_sha256": manifest["script_sha256"],
        "report_script_sha256": sha(__file__),
    }
    atomic_json(RUN / "results.json", report)
    PAGE.write_text(render(report))
    print(canonical({k: v for k, v in report.items() if k != "cases"}), flush=True)
    print(str(PAGE), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--live", action="store_true")
    actions.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    cases = sources()
    assert len(cases) == 6
    if args.report_only:
        manifest = read(RUN / "manifest.json")
        assert digest(cases) == manifest["source_rows_sha256"]
        assert implementation_hash() == manifest["implementation_sha256"]
        snapshot = RUN / "runner-at-execution.py"
        assert sha(snapshot if snapshot.exists() else __file__) == manifest["script_sha256"]
        write_report(saved_results(cases, RUN), manifest)
        return
    manifest = {
        "kind": VERSION,
        "planned_new_calls": len(cases),
        "repeats_per_input": 1,
        "source": str(SOURCE),
        "source_rows_sha256": digest(cases),
        "script_sha256": sha(__file__),
        "implementation_sha256": implementation_hash(),
        "execution": {"budget_policy": "no_limit_user_requested"},
        "scope": "Diagnostic only; no new downstream turns, D1, synthesis or original-statistic replacement",
    }
    freeze(RUN / "manifest.json", manifest)
    freeze(RUN / "source-snapshot.json", cases)
    print(canonical({"prepared": len(cases), "effort": "low", "output": str(RUN)}), flush=True)
    if not args.live:
        return
    provider = TinkerProvider()
    try:
        provider.renderer_for(MODEL, "low")
        provider.client_for(MODEL)
        timed = TimedProvider(provider)
        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda c: execute_case(c, timed, manifest, RUN), cases))
    finally:
        try:
            provider.close("success")
        except Exception as exc:
            print(canonical({"cleanup_warning": type(exc).__name__, "inference_records_preserved": True}), flush=True)
    assert digest(sources()) == manifest["source_rows_sha256"]
    write_report(results, manifest)


if __name__ == "__main__":
    main()
