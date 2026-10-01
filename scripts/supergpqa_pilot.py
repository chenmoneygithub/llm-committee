"""Prepare, run, audit and publish the authorized 20-question neutral SuperGPQA pilot."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sqlite3
import statistics
import urllib.request
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from llm_committee.pivot import supergpqa
from llm_committee.pivot.models import ROSTERS, canonical, digest
from llm_committee.pivot.providers import LiveProviders
from llm_committee.pivot.strong_run import run as run_journaled
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import (
    DATA_URL,
    PAIRS,
    VERSION,
    SuperGPQAGraph,
    SuperGPQAMockProvider,
    freeze,
    prepare,
    select_bank,
)

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / "runs/supergpqa-neutral-20-20260928-plan/question-bank.json"
RUN = ROOT / "runs/supergpqa-neutral-20-20260928/live-v2"
PAGE = ROOT / "docs/supergpqa-neutral-20-pilot-2026-09-28.html"
NAMES = dict(zip(ROSTERS["mixed_family"], ("Terra", "Qwen3.8-27B", "Inkling"), strict=True))
LABELS = {
    "strongly_agree": "Strongly agree",
    "leaning_agree": "Leaning agree",
    "leaning_disagree": "Leaning disagree",
    "strongly_disagree": "Strongly disagree",
    None: "未分类",
}


def read(path):
    return json.loads(path.read_text())


def prepare_bank(path=BANK):
    if path.exists():
        return read(path)
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        bank = select_bank(response.read())
    freeze(path, bank)
    return bank


def audit(directory, *, native=True):
    directory = Path(directory).resolve()
    manifest, contexts = read(directory / "manifest.json"), read(directory / "source-contexts.json")
    assert manifest["protocol_version"] == VERSION
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert (
        hashlib.sha256(Path(manifest["source"]["bank_file"]).read_bytes()).hexdigest()
        == manifest["source"]["bank_sha256"]
    )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(supergpqa.__file__).parent.glob("*.py"))
    }
    assert digest(hashes) == manifest["implementation_sha256"]
    mock = manifest["kind"] == "offline_mock"
    saved, renderers, checked_native, counts = {}, {}, Counter(), Counter()
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        for req, response, parsed in db.execute("SELECT request,response,parsed FROM calls WHERE status='completed'"):
            r, s, v = map(json.loads, (req, response, parsed))
            assert r["key"] not in saved
            saved[r["key"]] = (r, s, v)
            counts[r["purpose"]] += 1
            # Evaluation-only key, rationale and source metadata never enter prompts.
            text = canonical(r["messages"])
            assert '"answer_letter"' not in text and "screen_note" not in text
            assert "your_current_position" not in text
            assert "survey question" not in text
            if r["candidate_labels"]:
                assert r["purpose"] == "d_choice" and r["effort"] == "none" and s["reasoning_tokens"] == 0
                if not mock and native:
                    from llm_committee.pivot.tinker_provider import make_renderer
                    from scripts.audit_stateful_dyadic import check_native

                    if r["model"] not in renderers:
                        renderers[r["model"]] = make_renderer(r["model"], "none")
                    check_native(r, s, v, *renderers[r["model"]])
                    checked_native[r["model"]] += 1
            else:
                assert r["effort"] == "medium" and v["choice"] in r["schema"]["properties"]["choice"]["enum"]
    checked_requests = checked_questions = 0
    for plan in manifest["plans"]:
        path = directory / "questions" / f"{plan['question_id']}.json"
        if not path.exists():
            continue
        record = read(path)
        graph = SuperGPQAGraph(contexts[plan["question_id"]], plan, manifest)
        graph.failed = record["failed"]
        graph.blocked = record["blocked"]
        graph.submitted = set(graph.failed)
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            req = task.build(graph.values, lambda *_: 0)
            r, _, value = saved[task.key]
            assert canonical(req.document()) == canonical(r), task.key
            graph.accept(task, req, value)
            checked_requests += 1
        assert canonical(graph.report(None)) == canonical(record)
        for n in graph.route.nodes:
            formal = saved.get(graph.key(f"debate/{n.id}"))
            probe = saved.get(graph.key(f"D1/{n.id}"))
            if probe and formal:
                assert formal[0]["messages"][:-1] == probe[0]["messages"][:-1]
        checked_questions += 1
    if checked_questions == len(manifest["plans"]) and all(
        read(directory / "questions" / f"{p['question_id']}.json")["status"] == "completed" for p in manifest["plans"]
    ):
        assert dict(counts) == manifest["planned_counts"]["by_purpose"]
        assert checked_requests == len(saved) == manifest["planned_counts"]["logical_calls"]
    result = {
        "source": str(directory),
        "questions_reconstructed": checked_questions,
        "requests_reconstructed": checked_requests,
        "successful_requests": len(saved),
        "purposes": dict(counts),
        "native_probability_positions_checked": dict(checked_native),
        "mock": mock,
        "native_checks_requested": native,
        "no_ground_truth_in_model_input": True,
        "no_private_position_reinjection": True,
        "D1_context_equals_formal_input": True,
    }
    atomic_json(directory / "audit.json", result)
    return result


def summaries(records):
    models, chair = [], []
    for member, model in enumerate(ROSTERS["mixed_family"]):
        initial = [(r["initial_positions"].get(str(member)), r["answer_letter"]) for r in records]
        initial = [(v, a) for v, a in initial if v is not None]
        ends = [
            (e, r["answer_letter"])
            for r in records
            for e in r["endpoints"]
            if e["member"] == member and e["initial"] is not None and e["final"] is not None
        ]
        transitions = Counter(e["correctness_transition"] for e, _ in ends)
        models.append(
            {
                "model": model,
                "initial_n": len(initial),
                "initial_correct": sum(v["choice"] == a for v, a in initial),
                "endpoint_n": len(ends),
                "endpoint_correct": sum(e["final"]["choice"] == a for e, a in ends),
                "endpoint_initial_correct": sum(e["initial"]["choice"] == a for e, a in ends),
                "transitions": dict(transitions),
            }
        )
    for pair in PAIRS:
        cases = [
            (e, r["answer_letter"])
            for r in records
            for e in r["quality"]
            if e["pair"] == pair and e["baseline"] and e["debate"]
        ]
        chair.append(
            {
                "pair": pair,
                "n": len(cases),
                "baseline_correct": sum(e["baseline"]["choice"] == a for e, a in cases),
                "debate_correct": sum(e["debate"]["choice"] == a for e, a in cases),
                "transitions": dict(Counter(e["correctness_transition"] for e, _ in cases)),
            }
        )
    grouped, dgroups, dphases = defaultdict(list), defaultdict(list), defaultdict(list)
    labels = defaultdict(Counter)
    for r in records:
        for e in r["events"]:
            if e["after"]:
                labels[e["model"]][e["current_label"]] += 1
            if e["T"] >= 2 and e["after"] and e["before"]:
                grouped[e["previous_label"], e["current_label"]].append(e)
                if e["D1"]:
                    dgroups[e["model"], e["previous_label"], e["current_label"]].append(e)
                    phase = "first_participation" if "/initial/" in e["D1_before_request"] else "later_participation"
                    dphases[e["model"], e["previous_label"], e["current_label"], phase].append(e)
    conditional = [
        {
            "previous_label": a,
            "current_label": b,
            "n": len(es),
            "option_changes": sum(e["option_changed"] for e in es),
            "transitions": dict(Counter(e["correctness_transition"] for e in es)),
        }
        for (a, b), es in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]
    d1 = [
        {
            "model": model,
            "previous_label": a,
            "current_label": b,
            "n": len(es),
            "mean_tv": statistics.mean(e["D1"]["total_variation"] for e in es),
            "median_tv": statistics.median(e["D1"]["total_variation"] for e in es),
            "mean_correct_probability_change_pp": statistics.mean(e["D1"]["correct_probability_change_pp"] for e in es),
        }
        for (model, a, b), es in sorted(dgroups.items(), key=lambda item: str(item[0]))
    ]
    disagreements = []
    observed_disagreements = []
    for r in records:
        choices = [v["choice"] for v in r["initial_positions"].values()]
        if len(choices) == 3:
            disagreements.append(len(set(choices)) > 1)
        if len(choices) >= 2:
            observed_disagreements.append(len(set(choices)) > 1)
    return {
        "questions": len(records),
        "initial_disagreement_questions": sum(disagreements),
        "initial_disagreement_denominator": len(disagreements),
        "observed_initial_disagreement_questions": sum(observed_disagreements),
        "observed_initial_disagreement_denominator": len(observed_disagreements),
        "coverage": {
            "initial_answers": sum(len(r["initial_positions"]) for r in records),
            "replies": sum(e["after"] is not None for r in records for e in r["events"]),
            "complete_trajectories": sum(t["status"] == "completed" for r in records for t in r["trajectories"]),
            "failed_tasks": sum(len(r["failed"]) for r in records),
            "blocked_tasks": sum(len(r["blocked"]) for r in records),
        },
        "agreement_by_model": [
            {"model": model, "n": sum(labels[model].values()), "labels": dict(labels[model])}
            for model in ROSTERS["mixed_family"]
        ],
        "models": models,
        "chairman": chair,
        "C_conditional": conditional,
        "D1_conditional": d1,
        "D1_measurement_types": [
            {
                "model": model,
                "previous_label": a,
                "current_label": b,
                "measurement_type": phase,
                "n": len(es),
                "mean_tv": statistics.mean(e["D1"]["total_variation"] for e in es),
                "median_tv": statistics.median(e["D1"]["total_variation"] for e in es),
                "mean_correct_probability_change_pp": statistics.mean(
                    e["D1"]["correct_probability_change_pp"] for e in es
                ),
            }
            for (model, a, b, phase), es in sorted(dphases.items(), key=lambda item: str(item[0]))
        ],
    }


def publish(directory=RUN, page=PAGE):
    directory, page = Path(directory).resolve(), Path(page).resolve()
    manifest = read(directory / "manifest.json")
    records = [
        read(directory / "questions" / f"{p['question_id']}.json")
        for p in manifest["plans"]
        if (directory / "questions" / f"{p['question_id']}.json").exists()
    ]
    summary = summaries(records)
    progress = read(directory / "progress.json")
    summary.update(
        {
            "protocol_version": VERSION,
            "source": str(directory),
            "progress": progress,
            "scope": manifest["design"]["scope"],
        }
    )
    attempts = []
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        for req, response, parsed, status, charge in db.execute(
            "SELECT request,response,parsed,status,charge FROM calls"
        ):
            attempts.append(
                {
                    "request": json.loads(req),
                    "response": json.loads(response) if response else None,
                    "parsed": json.loads(parsed) if parsed else None,
                    "status": status,
                    "charge": charge,
                }
            )
    saved = {r["request"]["key"]: r for r in attempts if r["status"] == "completed"}
    summary["received_response_estimate_usd"] = sum(
        r["charge"] for r in attempts if r["response"] and r["status"] != "billing_unknown"
    )
    summary["unresolved_reserved_usd"] = sum(
        r["charge"] for r in attempts if not r["response"] or r["status"] == "billing_unknown"
    )
    summary["attempt_statuses"] = dict(Counter(r["status"] for r in attempts))
    summary["incomplete_response_attempts"] = sum(
        r["response"] is not None and r["response"]["status"] != "completed" for r in attempts
    )
    preflight = RUN.parent / "live/progress.json"
    if directory == RUN and preflight.exists():
        summary["excluded_v1_preflight_estimate_usd"] = read(preflight)["charged_or_reserved_usd"]
    atomic_json(page.with_suffix(".json"), summary)

    def esc(s):
        return html.escape(str(s))

    def ratio(n, d):
        return f"{n}/{d} ({100*n/d:.1f}%)" if d else "—"

    def table(headers, rows):
        return (
            '<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in rows)
            + "</tbody></table></div>"
        )

    def trans(t, name):
        return t.get(name, 0)

    out = [
        '<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>SuperGPQA · 20-question neutral pilot</title><style>",
        "body{font:16px/1.65 system-ui,sans-serif;background:#f5f7f8;color:#22303c;margin:0}main{max-width:1200px;margin:auto;padding:32px 24px}",
        "h1{font-size:30px}h2{margin-top:30px}h3{margin:24px 0 8px}.note{color:#536271}.card{background:white;border:1px solid #dce3e8;border-radius:12px;padding:20px;margin:18px 0}",
        "table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #dbe2e6;padding:11px;text-align:left;vertical-align:top}th{background:#edf3f5}",
        ".scroll{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 ui-monospace,monospace}.text{white-space:pre-wrap}",
        "summary{cursor:pointer;font-weight:650}details{border:1px solid #dce3e8;border-radius:8px;padding:14px;margin:12px 0}nav{display:flex;gap:12px;flex-wrap:wrap}a{color:#166c80}",
        ".good{color:#147147}.bad{color:#b23535}.chip{background:#e5edf1;padding:3px 9px;border-radius:5px}.key{background:#fff3ca}blockquote{border-left:3px solid #7fa1af;margin-left:0;padding-left:16px}",
        '</style><script>window.MathJax={tex:{inlineMath:[["$","$"],["\\\\(","\\\\)"]]},options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script>',
        '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script></head><body><main>',
        "<h1>SuperGPQA · 20 题闭卷 pilot</h1>",
        f'<p><span class="chip">{esc(progress["status"])}</span> 完成 {len(records)}/20 题 · 单一 neutral tone · mixed roster · 两人互辩</p>',
        '<nav><a href="#overview">正确率</a><a href="#a">自报态度</a><a href="#c">逐轮改对 / 改错</a><a href="#d">D1 概率</a><a href="#cases">20 题与完整讨论</a><a href="#setup">设置与限制</a></nav>',
        '<div class="card"><p>数学、物理、计算机、工程各 5 题。先筛题并冻结，再生成模型答案；题干、选项、答案键均保留原文。',
        "这是有质量筛选的诊断性子集，不是 SuperGPQA 全库准确率，也没有按模型是否答错挑题。</p>",
        "<p>A = GPT-5.6 Terra；B = Qwen3.8-27B；C = Inkling。每题运行 AB、CA、BC 三条独立路径，每条 4 次回复；",
        "没有第二套 tone assignment。每题 3 份初答在路径间复用；不同路径之后的讨论互不混合。",
        "成员保留自己的初答，并看到路径实际公开的消息，不额外读取对方未公开的初答。</p>",
        f'<p>三份初答齐全的题中，出现不同选项：{ratio(summary["initial_disagreement_questions"], summary["initial_disagreement_denominator"])} 题。',
        f'若也检查初答不齐全的题，已观察到分歧的题为 {summary["observed_initial_disagreement_questions"]}/{summary["observed_initial_disagreement_denominator"]}。',
        "后者只要求至少两份初答，不把缺失的第三份当成同意。</p></div>",
        '<div class="card"><h2>这轮 pilot 告诉我们什么？</h2>',
        "<p>独立初答正确："
        + "；".join(f'{NAMES[r["model"]]} {r["initial_correct"]}/{r["initial_n"]}' for r in summary["models"])
        + "。</p>",
    ]
    if all(r["initial_n"] and r["initial_correct"] / r["initial_n"] >= 0.95 for r in summary["models"]):
        out.append(
            "<p><strong>这批筛选题对当前阵容过于容易，错误与分歧太少，不适合直接扩大来检验 debate 的收益。</strong>"
            "它验证了运行与评分流程，但不能说明整个 SuperGPQA 已饱和。助手筛选可能偏向容易核验的题；"
            "更换题库或难度分层应预先定规则，不能事后只保留模型答错的题。</p>"
        )
    if all(r["n"] and r["baseline_correct"] == r["debate_correct"] == r["n"] for r in summary["chairman"]):
        out.append(
            "<p>三种搭配的 chairman 在无讨论和有讨论两侧均全部答对。这里的相同正确率存在天花板效应，"
            "不是“debate 没有作用”的有力证据。</p>"
        )
    out += [
        f'<p>覆盖：{summary["coverage"]["initial_answers"]}/60 份初答、{summary["coverage"]["replies"]}/240 条回复、',
        f'{summary["coverage"]["complete_trajectories"]}/60 条完整路径。缺失不算错误，表格使用各自实际分母。</p>',
    ]
    for i, r in enumerate(records, 1):
        if r["failed"]:
            failed_names = ", ".join(sorted({NAMES.get(f["model"], f["model"]) for f in r["failed"].values()}))
            out.append(
                f"<p>Q{i:02d} 的 {esc(failed_names)} 有请求在规定重试后仍失败；"
                f'{len(r["blocked"])} 个依赖请求未执行。原始尝试保留在日志中，没有人工补成有效答案。</p>'
            )
    out += [
        '</div><section id="overview"><h2>初答与讨论结束后的正确率</h2>',
        "<p>“正确”指选项与数据集的固定答案键一致，不使用 LLM judge。每名模型初答最多 20 份；",
        "它参加每题两条路径，因此最后一次参与的答案最多 40 份。40 个端点并非 40 道独立题；净变化按相同端点的初始答案配对计算。</p>",
    ]
    out.append(
        table(
            ["成员", "独立初答正确", "路径末次答案正确", "初错 → 末对", "初对 → 末错", "配对准确率变化"],
            [
                [
                    NAMES[r["model"]],
                    ratio(r["initial_correct"], r["initial_n"]),
                    ratio(r["endpoint_correct"], r["endpoint_n"]),
                    trans(r["transitions"], "wrong_to_right"),
                    trans(r["transitions"], "right_to_wrong"),
                    (
                        f"{100*(r['endpoint_correct']-r['endpoint_initial_correct'])/r['endpoint_n']:+.1f} pp"
                        if r["endpoint_n"]
                        else "—"
                    ),
                ]
                for r in summary["models"]
            ],
        )
    )
    out += [
        "<h2>Chairman：只有初答 vs 加入讨论</h2>",
        "<p>两边都是同一个 Terra chairman，使用同一对成员的同一份初答；debate 侧再加入该路径的四条公开回复。",
        "不是偏好胜率：“正确率”是 chairman 最终选项命中标准答案的比例。不做长度偏好 judging，也不重新生成 baseline 的成员初答。</p>",
    ]
    out.append(
        table(
            ["搭配", "配对题数", "无讨论 synthesis 正确", "讨论后 synthesis 正确", "改对", "改错"],
            [
                [
                    r["pair"],
                    r["n"],
                    ratio(r["baseline_correct"], r["n"]),
                    ratio(r["debate_correct"], r["n"]),
                    trans(r["transitions"], "wrong_to_right"),
                    trans(r["transitions"], "right_to_wrong"),
                ]
                for r in summary["chairman"]
            ],
        )
    )
    out += [
        '<p class="note">这里只描述这个小样本。讨论侧用了更多推理计算，不能据此隔离“互动”与“多做几次推理”的作用；',
        "三对搭配共享 20 道题，不能把它们当成 60 道独立题来检验。</p></section>",
        '<section id="a"><h2>A：成员如何描述自己对上一条消息的态度？</h2>',
        "<p>这是成员自己的 agreement 标签，不是外部 judge 的判断，也不等于它是否修改了自己的答案。所有轮次、单一 neutral tone。</p>",
        table(
            ["成员", "回复数", *[LABELS[k] for k in LABELS if k is not None]],
            [
                [NAMES[r["model"]], r["n"], *[ratio(r["labels"].get(k, 0), r["n"]) for k in LABELS if k is not None]]
                for r in summary["agreement_by_model"]
            ],
        ),
        "</section>",
        '<section id="c"><h2>C：同一成员逐次参与，答案如何变化？</h2>',
        "<p>按 [上条回复的自报 agreement，本条回复的自报 agreement] 分组。两人交替回复，所以两条 label 都对应彼此的贡献。",
        "比较的是本次选择与<strong>该成员自己上次参与</strong>的选择，不是简单比较两个相邻说话人的答案。",
        "T2–T4 合并；T1 的输入是初答，没有上条 debate label，不硬塞进这些组。</p>",
        "<p>“改对”分母是该组原来答错的次数；“改错”分母是原来答对的次数。两者不能使用同一个风险分母。",
        "小分组只供检查，不作为显著性结论。这里没有运行文本变化 judge。</p>",
    ]
    out.append(
        table(
            ["上条 label", "本条 label", "比较数", "换选项", "改对 / 原错", "改错 / 原对"],
            [
                [
                    LABELS[r["previous_label"]],
                    LABELS[r["current_label"]],
                    r["n"],
                    ratio(r["option_changes"], r["n"]),
                    ratio(
                        trans(r["transitions"], "wrong_to_right"),
                        trans(r["transitions"], "wrong_to_right") + trans(r["transitions"], "wrong_to_wrong"),
                    ),
                    ratio(
                        trans(r["transitions"], "right_to_wrong"),
                        trans(r["transitions"], "right_to_wrong") + trans(r["transitions"], "right_to_right"),
                    ),
                ]
                for r in summary["C_conditional"]
            ],
        )
    )
    out += [
        '</section><section id="d"><h2>D1：全部答案概率如何移动？</h2>',
        "<p>仅 Qwen 和 Inkling。正式答案启用 reasoning；D1 是对同一<strong>输入</strong>另做 reasoning-off 的单字母读数，",
        "不看到本次正式输出，也不额外注入上次保存的私有 position。分布在原始选项间归一化；top-20 未返回的选项按 0 计算。</p>",
        "<p><strong>TV</strong> = ½Σ|p<sub>after</sub>−p<sub>before</sub>|，这里显示为百分比：0% 表示分布没变，",
        "100% 表示前后支持完全不重叠；它描述移动量，不表示移动是否正确。",
        "<strong>ΔP(正确答案)</strong> = 后测减前测，单位是百分点（pp）；+10 pp 例如表示 40%→50%，−10 pp 表示远离答案键。",
        "每次前测取该成员上次参与的对应输入，首次参与取题目-only 初测。这是上下文读数，不是持久信念或单独的因果效应。</p>",
        "<p><strong>首次参与和后续参与的比较不完全相同：</strong>首次参与的前测还没有自己的 reasoning-on 初答，后测已有初答与收到的消息，",
        "因此较大的首次跳变不能只归因于同伴说服。下方每张表后附拆分检查，仍保留 agreement 条件，不改动原始测量。</p>",
    ]
    for model in ROSTERS["mixed_family"][1:]:
        out.append(f"<h3>{esc(NAMES[model])}</h3>")
        out.append(
            table(
                ["上条 label", "本条 label", "n", "平均 TV", "中位 TV", "平均 ΔP(正确答案)"],
                [
                    [
                        LABELS[r["previous_label"]],
                        LABELS[r["current_label"]],
                        r["n"],
                        f"{100*r['mean_tv']:.1f}%",
                        f"{100*r['median_tv']:.1f}%",
                        f"{r['mean_correct_probability_change_pp']:+.1f} pp",
                    ]
                    for r in summary["D1_conditional"]
                    if r["model"] == model
                ],
            )
        )
        out.append("<details><summary>检查大幅移动来自首次接触，还是后续讨论</summary>")
        out.append(
            table(
                ["上条 / 本条 label", "前后测量类型", "n", "平均 TV", "中位 TV", "平均 ΔP(正确答案)"],
                [
                    [
                        f'{LABELS[r["previous_label"]]} / {LABELS[r["current_label"]]}',
                        (
                            "题目-only → 首次参与输入"
                            if r["measurement_type"] == "first_participation"
                            else "上次参与输入 → 本次参与输入"
                        ),
                        r["n"],
                        f"{100*r['mean_tv']:.3f}%",
                        f"{100*r['median_tv']:.3f}%",
                        f"{r['mean_correct_probability_change_pp']:+.3f} pp",
                    ]
                    for r in summary["D1_measurement_types"]
                    if r["model"] == model
                ],
            )
        )
        out.append("</details>")
    out += [
        '<p class="note">D1 分布不必与 reasoning-on 的单次答案选择一致。每次前后完整分布在下方样例中，可检查平均数是否掩盖相反方向的移动。</p>',
        '</section><section id="cases"><h2>逐题检查：题目、初答、三条路径</h2>',
        "<p>参考答案只在本报告里显示，从未放入委员会或 chairman 的输入。</p>",
    ]
    for i, r in enumerate(records, 1):
        q, answer = r["question"], r["answer_letter"]
        out += [
            f'<details><summary>Q{i:02d} · {esc(r["domain"])} · {esc(r["difficulty"])} · {esc(q["text"][:130])}</summary>',
            f'<div class="text">{esc(q["text"])}</div><ol type="A">',
        ]
        out += [
            f'<li class="{"key" if chr(65+j)==answer else ""}">{esc(option)}</li>'
            for j, option in enumerate(q["options"])
        ]
        out += [f'</ol><p>答案键：{answer}。预运行检查：{esc(r["screen_note"])}</p>']
        for member, value in r["initial_positions"].items():
            model = ROSTERS["mixed_family"][int(member)]
            out += [
                f'<h3>T0 · {esc(NAMES[model])} · 选择 {esc(value["choice"])} · {"正确" if value["choice"]==answer else "错误"}</h3>',
                f'<div class="text">{esc(value["position"])}</div>',
            ]
        graph = SuperGPQAGraph(r, {"question_id": q["id"], "tone": "neutral"}, manifest)
        for pair in PAIRS:
            out.append(f"<details><summary>{pair} · 四轮讨论与最终 synthesis</summary>")
            for e in (e for e in r["events"] if e["pair"] == pair):
                v = e["after"]
                out.append(f'<h3>T{e["T"]} · {esc(NAMES[e["model"]])}</h3>')
                if v is None:
                    out.append("<p>缺失；见运行失败记录。</p>")
                    continue
                prior = e["before"]["choice"] if e["before"] else "—"
                out += [
                    f'<p>{esc(LABELS[v["agreement"]])} · 自己的选项 {prior} → {esc(v["choice"])} · {"正确" if v["choice"]==answer else "错误"}</p>',
                    f'<blockquote class="text">{esc(v["reply"])}</blockquote>',
                    f'<details><summary>同一次生成的 position</summary><div class="text">{esc(v["position"])}</div></details>',
                ]
                request = saved.get(graph.key(f'debate/{e["node"]}'))
                if request:
                    out.append("<details><summary>完整 formal prompt</summary>")
                    for m in request["request"]["messages"]:
                        out.append(f'<h4>{esc(m["role"])}</h4><pre>{esc(m["text"])}</pre>')
                    out.append("</details>")
                if e["D1"]:
                    d = e["D1"]
                    out.append(
                        f'<details><summary>D1 全分布 · TV {100*d["total_variation"]:.1f}% · ΔP(正确) {d["correct_probability_change_pp"]:+.1f} pp</summary>'
                    )
                    out.append(
                        table(
                            ["选项", "前测", "后测", "变化"],
                            [
                                [
                                    label + (" · 答案键" if label == answer else ""),
                                    f"{100*e['D1_before']['probabilities'][label]:.4f}%",
                                    f"{100*e['D1_after']['probabilities'][label]:.4f}%",
                                    f"{100*(e['D1_after']['probabilities'][label]-e['D1_before']['probabilities'][label]):+.4f} pp",
                                ]
                                for label in graph.question.labels
                            ],
                        )
                    )
                    for side in ("before", "after"):
                        dist = e[f"D1_{side}"]
                        req = saved.get(e[f"D1_{side}_request"])
                        out.append(
                            f'<p>{side} · 候选概率质量 {dist["candidate_mass"]:.6f} · 未返回选项 {esc(dist["missing_candidate_labels"])}</p>'
                        )
                        if req:
                            out.append(f"<details><summary>{side} · 完整 D1 prompt、输出与原始 logprobs</summary>")
                            for m in req["request"]["messages"]:
                                out.append(f'<h4>{esc(m["role"])}</h4><pre>{esc(m["text"])}</pre>')
                            out.append(
                                "<pre>"
                                + esc(
                                    json.dumps(
                                        {
                                            "text": req["response"]["text"],
                                            "reasoning_tokens": req["response"]["reasoning_tokens"],
                                            "sampled_output_index": req["parsed"]["_readout"]["sampled_output_index"],
                                            "candidate_logprobs": dist["candidate_logprobs"],
                                        },
                                        ensure_ascii=False,
                                        indent=2,
                                    )
                                )
                                + "</pre></details>"
                            )
                    out.append("</details>")
            quality = next(e for e in r["quality"] if e["pair"] == pair)
            for arm in ("baseline", "debate"):
                v = quality[arm]
                if v:
                    out += [
                        f'<h3>Chairman · {arm} · {esc(v["choice"])} · {"正确" if v["choice"]==answer else "错误"}</h3>',
                        f'<div class="text">{esc(v["position"])}</div>',
                    ]
            out.append("</details>")
        out.append("</details>")
    out += [
        '</section><section id="setup"><h2>设置、预算与限制</h2>',
        "<ul><li>正式初答、讨论和 synthesis：reasoning medium；最大输出 8192 tokens，包含 reasoning。不是强制写满。</li>",
        "<li>D1：reasoning off，最大 128 tokens（包含协议 framing）；输出只有一个选项字母。</li>",
        "<li>计划 60 初答 + 240 回复 + 200 D1 + 120 synthesis = 620 个逻辑请求；无独立重复，无 B/C 文本 judge，无 D2。</li>",
        "<li>接口/格式失败最多两次同请求重试；保留每次尝试，失败不会作为错误答案填 0，也不会伪造 agreement。</li>",
        "<li>20 道题是独立抽样单位，轮次和路径是重复观测。本报告不做显著性或全库泛化主张。</li>",
        "<li>筛题由助手在模型运行前完成，不声称经过独立专家验证。进一步争议需保留原始答案键并另行审计，不根据模型结果静默改答案。</li></ul>",
        f'<p>收到响应的 token 费用估算：US${summary["received_response_estimate_usd"]:.3f}；未解决的计费预留：US${summary["unresolved_reserved_usd"]:.3f}。预留不等于实际收费。</p>',
        (
            f'<p>另外，未纳入结果的 v1 工程检查估算 US${summary["excluded_v1_preflight_estimate_usd"]:.3f}；'
            f'两批合计约 US${summary["received_response_estimate_usd"] + summary["excluded_v1_preflight_estimate_usd"]:.3f}。不是供应商账单。</p>'
            if "excluded_v1_preflight_estimate_usd" in summary
            else ""
        ),
        f'<p>不完整输出尝试：{summary["incomplete_response_attempts"]}。调用状态：{esc(summary["attempt_statuses"])}。</p>',
        f"<p>原始数据与日志：<code>{esc(directory)}</code></p>",
        f'<p>冻结题库与筛选记录：<code>{esc(manifest["source"]["bank_file"])}</code></p>',
        "<p>公式显示可使用 MathJax CDN；即使离线，原始题目与公式文本仍保留。委员会运行本身没有浏览或调用工具。</p>",
        "</section></main></body></html>",
    ]
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("".join(out))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--html", type=Path, default=PAGE)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--mock", action="store_true")
    action.add_argument("--live", action="store_true")
    action.add_argument("--report-only", action="store_true")
    action.add_argument("--audit-only", action="store_true")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--max-inflight", type=int, default=32)
    args = parser.parse_args()
    if args.report_only:
        print(json.dumps(publish(args.output, args.html), ensure_ascii=False, indent=2))
        return
    if args.audit_only:
        print(json.dumps(audit(args.output), indent=2))
        return
    bank = prepare_bank(args.bank)
    if args.prepare:
        print(
            json.dumps(
                {
                    "bank": str(args.bank.resolve()),
                    "selected": len(bank["selected"]),
                    "screened": len(bank["screened"]),
                    "domains": dict(Counter(i["domain"] for i in bank["selected"])),
                },
                indent=2,
            )
        )
        return
    manifest, contexts = prepare(args.bank, mock=args.mock)
    print(
        canonical({"event": "prepared", "counts": manifest["planned_counts"], "run": str(args.output.resolve())}),
        flush=True,
    )
    factory = (
        None
        if args.mock
        else lambda: LiveProviders(
            judge=False, mixed=True, debate_effort="medium", closed_provider="databricks", databricks_profile="un"
        )
    )
    report = run_journaled(
        manifest,
        contexts,
        args.output,
        question_limit=args.questions,
        request_limit=args.max_inflight,
        workers=8,
        provider_factory=factory,
        token_count=lambda *_: 0,
        graph_type=SuperGPQAGraph,
        mock_provider_type=SuperGPQAMockProvider,
        event_namespace="supergpqa",
    )
    checked = audit(args.output)
    publish(args.output, args.html)
    print(canonical({"event": "published", "html": str(args.html.resolve()), "audit": checked}), flush=True)
    if report["status"] not in ("completed", "preflight_completed", "completed_with_failures"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
