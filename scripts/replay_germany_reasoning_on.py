"""One exact repeat of Germany's original reasoning-enabled formal T2 response."""

from __future__ import annotations

import argparse
import html
import json
import re
import sqlite3
from importlib.metadata import version

from llm_committee.pivot.models import Message, Request, digest
from llm_committee.pivot.prompts import parse_json
from llm_committee.pivot.storage import Journal
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.tinker_provider import TinkerProvider, make_renderer, message_content
from scripts.forced_feedback_public_history import OUTPUT, request_identity
from scripts.replay_germany_d1 import PAGE
from scripts.scale_public_history import ROOT, freeze, read, sha

RUN = ROOT / "runs/germany-reasoning-on-replay-20260928"
SOURCE_KEY = "archived-global-65/public-history-forced-feedback-2026-09-28-v1/BC-T1/forced/return"


def prepare():
    with sqlite3.connect(f"file:{OUTPUT}/requests.sqlite3?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute("SELECT * FROM calls WHERE key=? AND status='completed'", (SOURCE_KEY,)).fetchall()
    assert len(rows) == 1
    row = dict(rows[0])
    source = {key: json.loads(row[key]) for key in ("request", "response", "parsed")}
    doc = source["request"]
    request = Request(
        **{
            **doc,
            "key": "germany-reasoning-on-replay-20260928/T2",
            "messages": tuple(Message(**message) for message in doc["messages"]),
            "candidate_labels": tuple(doc["candidate_labels"]),
        }
    )
    assert request_identity(request.document()) == request_identity(doc)
    assert request.model == "Qwen/Qwen3.8-27B" and request.effort == "medium"
    assert request.max_output_tokens == 4096 and not request.candidate_labels
    return source, request, digest(row)


def run(live=False):
    source, request, row_hash = prepare()
    manifest = {
        "kind": "exact_reasoning_on_formal_T2_repeat",
        "planned_new_calls": 1,
        "source_row_sha256": row_hash,
        "script_sha256": sha(__file__),
        "execution": {"budget_policy": "no_limit_user_requested"},
        "note": "Exact original formal T2: medium reasoning, friendly tone, four-field response. Not a reasoning-only toggle of the neutral D1 prompt.",
    }
    freeze(RUN / "manifest.json", manifest)
    freeze(RUN / "source.json", source)
    tokenizer, renderer = make_renderer(request.model, request.effort)
    assert all(version(name) == pinned for name, pinned in source["response"]["raw"]["packages"].items())
    messages = [{"role": "system" if m.role == "developer" else m.role, "content": m.text} for m in request.messages]
    prompt_ids = renderer.build_generation_prompt(messages).to_ints()
    assert prompt_ids == source["response"]["raw"]["prompt_token_ids"]
    journal = Journal(RUN / "requests.sqlite3", manifest, None, should_stop=lambda: not live)
    try:
        parsed = journal.call(
            request, TinkerProvider() if live else None, lambda text: parse_json(text, request.schema)
        )
        response = json.loads(
            journal.db.execute("SELECT response FROM calls WHERE key=?", (request.key,)).fetchone()[0]
        )
        assert response["raw"]["prompt_token_ids"] == prompt_ids
        message, termination = renderer.parse_response(response["raw"]["sampled_token_ids"])
        visible, reasoning = message_content(message)
        assert termination.is_clean and response["raw"]["stop_reason"] == "stop"
        assert visible == response["text"] and parse_json(visible, request.schema) == parsed
        assert reasoning.strip() and response["reasoning_tokens"] == len(
            tokenizer.encode(reasoning, add_special_tokens=False)
        )
        report = {
            "status": "completed_and_audited",
            "original": source,
            "request": request.document(),
            "response": response,
            "parsed": parsed,
            "received_response_estimate_usd": journal.charged_usd,
            "exact_original_input_tokens": True,
            "reasoning_on_verified": True,
        }
    finally:
        journal.close()
    assert prepare()[2] == row_hash
    atomic_json(RUN / "results.json", report)
    labels = {
        "A": "Very favorable",
        "B": "Somewhat favorable",
        "C": "Somewhat unfavorable",
        "D": "Very unfavorable",
        "E": "DK/Refused",
    }
    esc = html.escape
    section = (
        '<section id="reasoning-on-replay"><h2>新增：原始正式 T2 的 reasoning-on 复跑（一次）</h2>'
        "<p>同一模型、同一 T0/T1、同一完整 prompt、medium reasoning、friendly tone、4096-token 上限。"
        "没有加入之前的 T2 回应。与原始正式 T2 完全一致；这不是仅打开 neutral D1 的 reasoning 开关。</p>"
        f'<p>原始选项：{source["parsed"]["choice"]}；本次选项：<strong>{parsed["choice"]}: {esc(labels.get(parsed["choice"], "No listed option"))}</strong>。'
        f'本次自报：{esc(str(parsed["agreement"]))}。只新增这一次样本，不据此估计 reasoning-on 的选项概率。</p>'
        f'<h3>本次公开回复</h3><blockquote>{esc(parsed["reply"])}</blockquote>'
        f'<h3>本次完整观点</h3><blockquote>{esc(parsed["position"])}</blockquote>'
        "<details><summary>原样复跑的完整 prompt</summary>"
        + "".join(f"<h3>{esc(m.role)}</h3><pre>{esc(m.text)}</pre>" for m in request.messages)
        + f'</details><p>记录到 reasoning tokens：{response["reasoning_tokens"]}；新增费用估算 ${report["received_response_estimate_usd"]:.6f}。</p></section>'
    )
    backup = RUN / "diagnostic-before-reasoning-on.html"
    if not backup.exists():
        backup.write_bytes(PAGE.read_bytes())
    page = re.sub(r'<section id="reasoning-on-replay">.*?</section>', "", PAGE.read_text(), flags=re.S)
    anchor = "<h2>实际保存的轨迹（没有重生成）</h2>"
    assert page.count(anchor) == 1
    PAGE.write_text(page.replace(anchor, section + anchor, 1))
    print(
        json.dumps(
            {
                "status": report["status"],
                "original_choice": source["parsed"]["choice"],
                "repeat": parsed,
                "reasoning_tokens": response["reasoning_tokens"],
                "cost_usd": report["received_response_estimate_usd"],
                "html": str(PAGE),
                "result_json": str(RUN / "results.json"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    run(parser.parse_args().live)
