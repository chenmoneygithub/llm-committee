"""Two exact, separately journaled Qwen D1 repeats; never regenerate debate."""

from __future__ import annotations

import argparse
import html
import json
import sqlite3
from importlib.metadata import version

from llm_committee.pivot.failures import FORMAT_RETRY_POLICY
from llm_committee.pivot.models import Message, Question, Request, digest
from llm_committee.pivot.probabilities import distribution
from llm_committee.pivot.stateful_prompts import parse_choice
from llm_committee.pivot.storage import Journal
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.tinker_provider import TinkerProvider, make_renderer
from scripts.audit_stateful_dyadic import check_native
from scripts.forced_feedback_public_history import FORCE, OUTPUT, request_identity
from scripts.scale_public_history import ROOT, freeze, read, sha

QID = "archived-global-65"
RUN = ROOT / "runs/germany-d1-exact-replay-20260928"
PAGE = ROOT / "docs/germany-d1-exact-replay-2026-09-28.html"


def load_sources():
    record = read(OUTPUT / "questions" / f"{QID}.json")
    case = next(c for c in record["cases"] if c["id"] == "BC-T1")
    sources = {}
    for name, key in (
        ("before", case["measurement_keys"]["D1_before"]),
        ("after", case["arms"]["forced"]["requests"]["D1"]),
    ):
        provenance = record["reused_measurements"].get(key)
        directory = provenance["directory"] if provenance else str(OUTPUT)
        storage_key = provenance["storage_key"] if provenance else key
        with sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True) as db:
            db.row_factory = sqlite3.Row
            rows = db.execute("SELECT * FROM calls WHERE key=? AND status='completed'", (storage_key,)).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        if provenance:
            assert digest(row) == provenance["row_sha256"]
        sources[name] = {
            "directory": directory,
            "storage_key": storage_key,
            "row_sha256": digest(row),
            **{k: json.loads(row[k]) for k in ("request", "response", "parsed")},
        }
    return record, case, sources


def repeat_request(source, name):
    doc = source["request"]
    request = Request(
        **{
            **doc,
            "key": f"germany-d1-exact-replay-20260928/{name}",
            "messages": tuple(Message(**m) for m in doc["messages"]),
            "candidate_labels": tuple(doc["candidate_labels"]),
        }
    )
    assert request_identity(doc) == request_identity(request.document())
    assert request.model == "Qwen/Qwen3.8-27B" and request.effort == "none"
    return request


def render(result):
    def esc(x):
        return html.escape(str(x))

    question, case = result["question"], result["case"]
    arm = case["arms"]["forced"]
    columns = [(stage, run) for stage in ("before", "after") for run in ("original", "repeat")]
    rows = []
    for label, option in zip(question["labels"], question["options"], strict=True):
        values = [result["readings"][stage][run]["distribution"]["probabilities"][label] for stage, run in columns]
        rows.append(
            f"<tr><td>{label}: {esc(option)}</td>" + "".join(f"<td>{100*v:.4f}%</td>" for v in values) + "</tr>"
        )
    parts = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Germany · exact D1 replay</title><style>body{font:16px/1.65 system-ui;margin:32px auto;padding:0 20px;max-width:1050px;color:#24332f}"
        "table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ccd6d1;text-align:left;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        ".scroll{overflow-x:auto}details{border:1px solid #ccd6d1;padding:16px;margin:16px 0}summary{cursor:pointer;font-weight:650}blockquote{border-left:3px solid #368574;padding-left:18px;margin-left:0}</style></head><body>",
        "<h1>德国案例：同一 Qwen、原样 D1 复测</h1>",
        f'<p>{esc(question["text"])}</p>',
        "<p>仅复跑原题前测、收到强制反驳后的后测各一次。所有 model/settings/messages 与对应原请求一致；未重新生成 debate，未加入任何诊断提示。"
        "Qwen3.8-27B，reasoning off，temperature 1。后测看到 T0 和 T1，不看到随后正式生成的 T2 回复或新观点。</p>",
        "<h2>全部选项概率</h2><p>在 A–E 选项间归一化，每列合计 100%。抽样输出的字母不是整个概率分布。</p>",
        '<div class="scroll"><table><thead><tr><th>选项</th><th>原始前测</th><th>复测前测</th><th>原始后测</th><th>复测后测</th></tr></thead><tbody>',
        "".join(rows),
        "</tbody></table></div>",
        "<h2>实际保存的轨迹（没有重生成）</h2>",
        "<h3>T0 · Qwen 初始观点 · B: Somewhat favorable</h3>",
        f'<blockquote>{esc(arm["position_before"]["position"])}</blockquote>',
        f'<h3>T1 · Inkling 反驳 · tone: {esc(case["tones"]["BC-1"])} · {esc(arm["challenger_label"])}</h3>',
        f'<blockquote>{esc(arm["feedback"]["reply"])}</blockquote>',
        f'<h3>T2 · Qwen 正式回应 · reasoning on · tone: {esc(case["tones"]["BC-2"])} · {esc(arm["recipient_label"])}</h3>',
        f'<blockquote>{esc(arm["return"]["reply"])}</blockquote>',
        f'<h3>T2 新观点 · {esc(arm["return"]["choice"])}: Very favorable</h3><blockquote>{esc(arm["return"]["position"])}</blockquote>',
        "<details><summary>Inkling 的私有反驳指令（不在 Qwen 的 D1 输入中）</summary>",
        f"<pre>{esc(FORCE)}</pre></details>",
        "<h2>完整 D1 输入和输出</h2>",
    ]
    for stage in ("before", "after"):
        readings = result["readings"][stage]
        parts.append(f'<details><summary>{"前测" if stage=="before" else "后测"}：逐条消息、原始输出和概率</summary>')
        for m in readings["repeat"]["request"]["messages"]:
            parts.append(f'<h3>{esc(m["role"])}</h3><pre>{esc(m["text"])}</pre>')
        for run in ("original", "repeat"):
            entry = readings[run]
            parts.append(f'<h3>{run} · 输出 {esc(entry["response"]["text"])}</h3>')
            parts.append(
                "<pre>"
                + esc(
                    json.dumps(
                        {
                            "reasoning_tokens": entry["response"]["reasoning_tokens"],
                            "sampled_output_index": entry["parsed"]["_readout"]["sampled_output_index"],
                            "candidate_logprobs": entry["distribution"]["candidate_logprobs"],
                            "candidate_mass": entry["distribution"]["candidate_mass"],
                            "missing_candidate_labels": entry["distribution"]["missing_candidate_labels"],
                            "probabilities": entry["distribution"]["probabilities"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
        parts.append("</details>")
    parts.append(
        f'<p>新的请求费用估算 ${result["received_response_estimate_usd"]:.6f}。实际 token 位置、选项 token 映射、完整 prompt token 序列已核对。</p></body></html>'
    )
    return "".join(parts)


def run(*, live=False):
    record, case, sources = load_sources()
    manifest = {
        "kind": "exact_D1_diagnostic",
        "question_id": QID,
        "planned_new_calls": 2,
        "repeats_per_input": 1,
        "execution": {"budget_policy": "no_limit_user_requested", "failure_policy": FORMAT_RETRY_POLICY},
        "source_rows": {name: entry["row_sha256"] for name, entry in sources.items()},
        "source_question_sha256": sha(OUTPUT / "questions" / f"{QID}.json"),
        "script_sha256": sha(__file__),
    }
    freeze(RUN / "manifest.json", manifest)
    freeze(RUN / "source-snapshot.json", {"record": record, "sources": sources})
    question = Question.from_dict(record["question"])
    native = make_renderer("Qwen/Qwen3.8-27B", "none")
    requests = {name: repeat_request(entry, name) for name, entry in sources.items()}
    for source in sources.values():
        check_native(source["request"], source["response"], source["parsed"], *native)
        assert all(version(package) == pinned for package, pinned in source["response"]["raw"]["packages"].items())
    journal = Journal(RUN / "requests.sqlite3", manifest, None, should_stop=lambda: not live)
    try:
        provider = TinkerProvider() if live else None
        for request in requests.values():
            journal.call(request, provider, lambda text: parse_choice(text, question))
        readings = {}
        for stage, request in requests.items():
            rows = journal.db.execute(
                "SELECT request,response,parsed FROM calls WHERE status='completed' AND json_extract(request,'$.key')=?",
                (request.key,),
            ).fetchall()
            assert len(rows) == 1
            req, response, parsed = map(json.loads, rows[0])
            old = sources[stage]
            check_native(req, response, parsed, *native)
            assert request_identity(req) == request_identity(old["request"])
            assert response["raw"]["prompt_token_ids"] == old["response"]["raw"]["prompt_token_ids"]
            assert parsed["_readout"]["prefix_token_ids"] == old["parsed"]["_readout"]["prefix_token_ids"]
            readings[stage] = {
                "original": {key: old[key] for key in ("request", "response", "parsed")},
                "repeat": {"request": req, "response": response, "parsed": parsed},
            }
            for entry in readings[stage].values():
                entry["distribution"] = distribution(entry["parsed"]["_readout"], question.labels, missing_as_zero=True)
        report = {
            "status": "completed_and_audited",
            "question": {**record["question"], "labels": question.labels},
            "case": case,
            "readings": readings,
            "received_response_estimate_usd": journal.charged_usd,
            "all_original_prompts_unchanged": True,
            "source_snapshot_sha256": sha(RUN / "source-snapshot.json"),
        }
    finally:
        journal.close()
    assert manifest["source_rows"] == {name: entry["row_sha256"] for name, entry in load_sources()[2].items()}
    atomic_json(RUN / "results.json", report)
    PAGE.write_text(render(report))
    print(
        json.dumps(
            {
                "status": report["status"],
                "cost_usd": report["received_response_estimate_usd"],
                "html": str(PAGE),
                "readings": {
                    stage: {
                        run: {"output": entry["response"]["text"], "distribution": entry["distribution"]}
                        for run, entry in readings.items()
                    }
                    for stage, readings in report["readings"].items()
                },
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Authorize only the two saved-prompt D1 repeats")
    run(live=parser.parse_args().live)
