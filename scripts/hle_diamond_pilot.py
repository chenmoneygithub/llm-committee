"""Run and privately report the authorized 20-question HLE-Diamond pilot."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from llm_committee.pivot.models import ROSTERS, canonical, digest
from llm_committee.pivot.providers import LiveProviders
from llm_committee.pivot.strong_run import run
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.supergpqa import SuperGPQAMockProvider
from scripts.hle_diamond_study import (
    BANK,
    RUN,
    VERSION,
    PairedGraph,
    download_bank,
    implementation_hash,
    prepare,
    read,
    sha,
)
from scripts.supergpqa_pilot import LABELS, NAMES, summaries

PAGE = RUN.parent / "hle-diamond-20-pilot.html"


def attempts(directory):
    with closing(sqlite3.connect(f"file:{directory}/requests.sqlite3?mode=ro", uri=True)) as db:
        return [
            {
                "request": json.loads(r),
                "response": json.loads(s) if s else None,
                "parsed": json.loads(v) if v else None,
                "status": status,
                "charge": charge,
            }
            for r, s, v, status, charge in db.execute("SELECT request,response,parsed,status,charge FROM calls")
        ]


def audit(directory=RUN, *, native=True):
    directory = Path(directory).resolve()
    manifest, contexts = read(directory / "manifest.json"), read(directory / "source-contexts.json")
    assert manifest["protocol_version"] == VERSION
    assert manifest["implementation_sha256"] == implementation_hash()
    assert digest(contexts) == manifest["source"]["contexts_sha256"]
    assert sha(manifest["source"]["bank_file"]) == manifest["source"]["bank_sha256"]
    assert sha(manifest["source"]["tone_template"]) == manifest["source"]["tone_template_sha256"]
    saved = {a["request"]["key"]: a for a in attempts(directory) if a["status"] == "completed"}
    checked = Counter()
    renderers = {}
    for a in saved.values():
        request, response, parsed = a["request"], a["response"], a["parsed"]
        payload = canonical(request["messages"])
        assert '"answer_letter"' not in payload and '"rationale"' not in payload
        assert "your_current_position" not in payload and "survey question" not in payload
        if request["candidate_labels"]:
            assert request["effort"] == "none" and response["reasoning_tokens"] == 0
            if native and manifest["kind"] != "offline_mock":
                from llm_committee.pivot.tinker_provider import make_renderer
                from scripts.audit_stateful_dyadic import check_native

                model = request["model"]
                if model not in renderers:
                    renderers[model] = make_renderer(model, "none")
                check_native(request, response, parsed, *renderers[model])
                checked[model] += 1
    questions = requests = 0
    for plan in manifest["plans"]:
        record_file = directory / "questions" / f"{plan['question_id']}.json"
        if not record_file.exists():
            continue
        record = read(record_file)
        graph = PairedGraph(contexts[plan["question_id"]], plan, manifest)
        graph.failed, graph.blocked = record["failed"], record["blocked"]
        graph.submitted = set(graph.failed)
        while not graph.complete:
            task = graph.ready()
            assert task is not None
            request = task.build(graph.values, lambda *_: 0)
            item = saved[task.key]
            assert canonical(request.document()) == canonical(item["request"]), task.key
            graph.accept(task, request, item["parsed"])
            requests += 1
        assert canonical(graph.report(None)) == canonical(record)
        for arm in ("original", "alternate"):
            for node in graph.route.nodes:
                f = saved.get(graph.key(arm, f"debate/{node.id}"))
                d = saved.get(graph.key(arm, f"D1/{node.id}"))
                if f and d:
                    assert f["request"]["messages"][:-1] == d["request"]["messages"][:-1]
        questions += 1
    result = {
        "questions_reconstructed": questions,
        "requests_reconstructed": requests,
        "successful_requests": len(saved),
        "native_probability_positions_checked": dict(checked),
        "no_answer_key_or_private_position_feedback": True,
        "D1_input_matches_formal_context": True,
        "mock": manifest["kind"] == "offline_mock",
    }
    atomic_json(directory / "audit.json", result)
    return result


def publish(directory=RUN, page=PAGE):
    directory, page = Path(directory).resolve(), Path(page).resolve()
    manifest, progress = read(directory / "manifest.json"), read(directory / "progress.json")
    records = [
        read(directory / "questions" / f"{p['question_id']}.json")
        for p in manifest["plans"]
        if (directory / "questions" / f"{p['question_id']}.json").exists()
    ]
    rows = attempts(directory)
    saved = {a["request"]["key"]: a for a in rows if a["status"] == "completed"}
    result = summaries(records)
    result.update(
        protocol_version=VERSION,
        source=str(directory),
        progress=progress,
        received_response_estimate_usd=sum(
            a["charge"] for a in rows if a["response"] and a["status"] != "billing_unknown"
        ),
        unresolved_reservations_usd=sum(
            a["charge"] for a in rows if not a["response"] or a["status"] == "billing_unknown"
        ),
        attempt_statuses=dict(Counter(a["status"] for a in rows)),
    )
    tones = defaultdict(Counter)
    for r in records:
        for e in r["events"]:
            if e["after"]:
                tones[e["current_tone"]][e["current_label"]] += 1
    result["labels_by_current_tone"] = {k: dict(v) for k, v in tones.items()}
    atomic_json(page.with_suffix(".json"), result)

    def esc(x):
        return html.escape(str(x))

    def ratio(n, d):
        return f"{n}/{d} ({100*n/d:.1f}%)" if d else "—"

    def table(headers, body):
        return (
            '<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{esc(c)}</td>" for c in row) + "</tr>" for row in body)
            + "</tbody></table></div>"
        )

    def prompt_details(key, label):
        a = saved.get(key)
        if not a:
            return ""
        body = "".join(f'<h4>{esc(m["role"])}</h4><pre>{esc(m["text"])}</pre>' for m in a["request"]["messages"])
        output = {"visible_output": a["response"]["text"], "reasoning_tokens": a["response"]["reasoning_tokens"]}
        if a["parsed"].get("_readout"):
            output["readout"] = a["parsed"]["_readout"]
        return f"<details><summary>{esc(label)}</summary>{body}<pre>{esc(json.dumps(output,ensure_ascii=False,indent=2))}</pre></details>"

    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        "<title>HLE-Diamond · paired turn-tone pilot</title><style>body{font:16px/1.6 system-ui;color:#23343e;background:#f5f7f8;margin:0}main{max-width:1160px;margin:auto;padding:30px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}th{background:#e8eff3}.scroll{overflow:auto}details{background:white;border:1px solid #dce3e8;border-radius:8px;margin:12px 0;padding:14px}summary{cursor:pointer;font-weight:650}pre,.text{white-space:pre-wrap;overflow-wrap:anywhere}pre{font-size:13px}.note{background:#fff4d3;padding:14px}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#126577}</style><main>",
        "<h1>HLE-Diamond · 20 题 pilot</h1>",
        '<p class="note">私有研究报告：包含 gated benchmark 原文。请勿发布、上传或提交到 Git。</p>',
        f'<p>状态：{esc(progress["status"])}；已处理 {len(records)}/20 题；完整路径 {result["coverage"]["complete_trajectories"]}/120。</p>',
        '<nav><a href="#overview">结果概览</a><a href="#a">A 自报态度</a><a href="#c">C 改对 / 改错</a><a href="#d">D1 概率移动</a><a href="#cases">完整案例</a><a href="#setup">设置</a></nav>',
        "<p>从官方 75 道纯文本 reasoning 选择题中固定随机抽 20 道，不按能否解出来或模型表现筛题。不是完整 HLE/HLE-Diamond 得分。</p>",
        "<p>A = Terra，B = Qwen3.8-27B，C = Inkling。每题 AB、CA、BC 三条四轮双人路径；原样复用 GlobalOpinionQA 的逐轮 friendly/neutral/hostile 指令和 20 份排程。两种版本共享 T1/T2，只在 T3/T4 改 tone，合计每题 18 条独立生成的回复。不是三组 global tone。</p>",
        '<section id="overview"><h2>成员初答与路径末次答案</h2>',
        "<p>正确 = 选项命中数据集答案键，不用偏好 judge。初答最多 20 份/模型；每模型每题参与两对搭配、各两个分支，因此有最多 80 个末次答案。重复端点不是独立题；变化与对应初答配对。</p>",
        table(
            ["模型", "独立初答正确", "路径末次正确", "初错→末对", "初对→末错"],
            [
                [
                    NAMES[r["model"]],
                    ratio(r["initial_correct"], r["initial_n"]),
                    ratio(r["endpoint_correct"], r["endpoint_n"]),
                    r["transitions"].get("wrong_to_right", 0),
                    r["transitions"].get("right_to_wrong", 0),
                ]
                for r in result["models"]
            ],
        ),
        "<h2>E：相同 chairman，只有初答 vs 加入讨论</h2>",
        "<p>同一对成员的初答只生成一次，无讨论 synthesis 也只生成一次，与两个讨论版本分别配对。下表合并两个 tone 版本，不把它们当作独立题。讨论侧用了更多计算，不能隔离互动与额外推理的作用。</p>",
        table(
            ["搭配", "配对路径数", "无讨论正确", "讨论后正确", "改对", "改错"],
            [
                [
                    r["pair"],
                    r["n"],
                    ratio(r["baseline_correct"], r["n"]),
                    ratio(r["debate_correct"], r["n"]),
                    r["transitions"].get("wrong_to_right", 0),
                    r["transitions"].get("right_to_wrong", 0),
                ]
                for r in result["chairman"]
            ],
        ),
        '</section><section id="a"><h2>A：当前 turn 的 tone 与自报态度</h2>',
        "<p>每次正式回复只数一次，共享前缀不重复计数。自报态度针对收到的消息，不代表自己的选项是否改变。</p>",
        table(
            ["本轮 tone", "回复数", *[LABELS[k] for k in LABELS]],
            [
                [tone, sum(v.values()), *[ratio(v.get(k, 0), sum(v.values())) for k in LABELS]]
                for tone, v in sorted(tones.items())
            ],
        ),
        '</section><section id="c"><h2>C：收到某类回应后，自己的答案怎么变？</h2>',
        "<p>按 [上条回复 label，本条回复 label] 分组，合并 T2–T4。比较同一成员本次与自己上次参与时的选择；首次参与参照其初答。T1 没有上条 debate label，只在案例和 A 中显示。“改对”的分母是原先答错的次数；“改错”的分母是原先答对的次数。这里没有文本变化 judge。</p>",
        table(
            ["上条 label", "本条 label", "次数", "换选项", "改对 / 原错", "改错 / 原对"],
            [
                [
                    LABELS[r["previous_label"]],
                    LABELS[r["current_label"]],
                    r["n"],
                    ratio(r["option_changes"], r["n"]),
                    ratio(
                        r["transitions"].get("wrong_to_right", 0),
                        sum(r["transitions"].get(k, 0) for k in ("wrong_to_right", "wrong_to_wrong")),
                    ),
                    ratio(
                        r["transitions"].get("right_to_wrong", 0),
                        sum(r["transitions"].get(k, 0) for k in ("right_to_wrong", "right_to_right")),
                    ),
                ]
                for r in result["C_conditional"]
            ],
        ),
        '</section><section id="d"><h2>D1：选项概率分布移动了多少？</h2>',
        "<p>对正式解题的同一输入另做 neutral、reasoning-off 单字母读取，不看本次生成的答案。所有原生选项归一化，top-20 未返回的选项按零处理。</p>",
        "<p><b>TV = ½Σ|p后−p前|</b>：0% 表示分布不动，100% 表示支持完全不重叠；只表示移动量，不表示对错。<b>ΔP(正确答案)</b> 用百分点表示，例如 40%→50% 是 +10 pp。负数表示正确选项的概率下降。</p>",
        '<p class="note">首次参与的前测只有题目，后测还包含自己的 reasoning-on 初答和收到的消息，因此首次跳变不能单独归因于 peer persuasion。T3 的两版 D1 输入完全相同，只读一次；按各正式回复 label 展示时可能出现在不同条件组，不是独立的两次测量，也不是当前 tone 的直接效应。</p>',
    ]
    for model in ROSTERS["mixed_family"][1:]:
        parts += [
            f"<h3>{esc(NAMES[model])}</h3>",
            table(
                ["上条 label", "本条 label", "n", "平均 TV", "中位 TV", "平均 ΔP(正确答案)"],
                [
                    [
                        LABELS[r["previous_label"]],
                        LABELS[r["current_label"]],
                        r["n"],
                        f'{100*r["mean_tv"]:.1f}%',
                        f'{100*r["median_tv"]:.1f}%',
                        f'{r["mean_correct_probability_change_pp"]:+.1f} pp',
                    ]
                    for r in result["D1_conditional"]
                    if r["model"] == model
                ],
            ),
        ]
    parts.append('</section><section id="cases"><h2>20 道原题与完整轨迹（仅本地）</h2>')
    for i, r in enumerate(records, 1):
        plan = next(p for p in manifest["plans"] if p["question_id"] == r["question"]["id"])
        graph = PairedGraph(r, plan, manifest)
        parts += [
            f'<details><summary>Q{i:02d} · {esc(r["domain"])} · {esc(r["question"]["text"][:95])}</summary>',
            f'<div class="text">{esc(r["original_question"])}</div><p>答案键（仅评分使用）：{esc(r["answer_letter"])}</p>',
        ]
        for m, v in r["initial_positions"].items():
            parts.append(
                f'<details><summary>初答 · {esc(NAMES[ROSTERS["mixed_family"][int(m)]])} · {esc(v["choice"])}</summary><div class="text">{esc(v["position"])}</div>{prompt_details(graph.key("original",f"initial/{m}"),"完整初答 prompt / output")}</details>'
            )
        for pair in ("AB", "CA", "BC"):
            parts.append(f"<details><summary>{pair} · 共享 T1/T2 与两个后续分支</summary>")
            for e in (e for e in r["events"] if e["pair"] == pair):
                owner = "shared" if e["T"] <= 2 else e["arm"]
                v = e["after"]
                parts.append(f'<h3>{owner} T{e["T"]} · {esc(NAMES[e["model"]])} · {esc(e["current_tone"])}</h3>')
                if not v:
                    parts.append("<p>缺失，不当作错误或 agreement。</p>")
                    continue
                before = e["before"]["choice"] if e["before"] else "—"
                parts += [
                    f'<p>{esc(LABELS[v["agreement"]])}；自己的选项 {before} → {esc(v["choice"])}</p><div class="text">{esc(v["reply"])}</div>',
                    f'<details><summary>同次生成的 position</summary><div class="text">{esc(v["position"])}</div></details>',
                    prompt_details(e["request_key"], "完整正式 prompt / output"),
                ]
                if e["D1"]:
                    parts.append("<details><summary>D1 前后全分布与完整 prompts</summary>")
                    parts.append(
                        table(
                            ["选项", "前测概率", "后测概率"],
                            [
                                [
                                    k,
                                    f'{100*e["D1_before"]["probabilities"][k]:.5f}%',
                                    f'{100*e["D1_after"]["probabilities"][k]:.5f}%',
                                ]
                                for k in graph.question.labels
                            ],
                        )
                    )
                    for side in ("before", "after"):
                        parts.append(
                            prompt_details(e[f"D1_{side}_request"], f"{side}: prompt / visible output / token logprobs")
                        )
                    parts.append("</details>")
            for arm in ("original", "alternate"):
                q = next(q for q in r["quality"] if q["pair"] == pair and q["arm"] == arm)
                for which in (("baseline", "debate") if arm == "original" else ("debate",)):
                    v = q[which]
                    if v:
                        parts.append(
                            f'<details><summary>Chairman {arm} / {which} · {esc(v["choice"])}</summary><div class="text">{esc(v["position"])}</div></details>'
                        )
            parts.append("</details>")
        parts.append("</details>")
    parts += [
        '</section><section id="setup"><h2>执行与限制</h2>',
        "<p>正式解题与 chairman：medium reasoning，16,384-token 输出上限（含 reasoning）。D1：reasoning off，128-token 上限。无工具或联网；无 B/C 文本 judge、无 D2、无额外独立重复。</p>",
        "<p>计划 60 初答 + 360 回复 + 240 D1 + 180 synthesis = 840 请求。每次失败最多两次同请求重试；保留原始尝试，失败只阻断真正依赖项。20 道题才是独立单位，不做显著性或全 benchmark 泛化主张。</p>",
        f'<p>收到响应的 token 费用估算 US${result["received_response_estimate_usd"]:.3f}；未知计费预留 US${result["unresolved_reservations_usd"]:.3f}，不是供应商账单。</p>',
        f'<p>覆盖：{esc(result["coverage"])}。调用状态：{esc(result["attempt_statuses"])}。</p>',
        f'<details><summary>失败记录</summary><pre>{esc(json.dumps(progress.get("task_failures",{}),ensure_ascii=False,indent=2))}</pre></details>',
        f'<p>冻结 dataset revision：{manifest["source"]["revision"]}。原始数据：{esc(directory)}</p>',
        "</section></main></html>",
    ]
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("".join(parts))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("prepare", "mock", "live", "report-only", "audit-only"):
        actions.add_argument("--" + action, action="store_true")
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--output", type=Path, default=RUN)
    parser.add_argument("--html", type=Path, default=PAGE)
    args = parser.parse_args()
    if args.report_only:
        r = publish(args.output, args.html)
        print(
            canonical(
                {
                    "html": str(args.html.resolve()),
                    "coverage": r["coverage"],
                    "cost_usd": r["received_response_estimate_usd"],
                }
            )
        )
        return
    if args.audit_only:
        print(canonical(audit(args.output)))
        return
    download_bank()
    manifest, contexts = prepare(mock=args.mock)
    print(canonical({"prepared": manifest["planned_counts"], "bank": str(BANK)}), flush=True)
    if args.prepare:
        return
    output = args.output.resolve()
    if args.mock and output == RUN:
        raise ValueError("Use a separate --output for mock runs")
    output.mkdir(parents=True, exist_ok=True)
    protocol = output / "hle_diamond_study_snapshot.py"
    source = Path(__file__).with_name("hle_diamond_study.py")
    if protocol.exists():
        assert sha(protocol) == sha(source)
    else:
        shutil.copyfile(source, protocol)
    result = run(
        manifest,
        contexts,
        output,
        question_limit=args.questions,
        request_limit=32,
        workers=8,
        provider_factory=(
            None
            if args.mock
            else lambda: LiveProviders(
                judge=False, mixed=True, debate_effort="medium", closed_provider="databricks", databricks_profile="un"
            )
        ),
        token_count=lambda *_: 0,
        graph_type=PairedGraph,
        mock_provider_type=SuperGPQAMockProvider,
        event_namespace="hle_diamond",
    )
    checked = audit(output)
    summary = publish(output, args.html)
    print(
        canonical(
            {
                "html": str(args.html.resolve()),
                "audit": checked,
                "coverage": summary["coverage"],
                "cost_usd": summary["received_response_estimate_usd"],
            }
        ),
        flush=True,
    )
    if result["status"] not in ("completed", "completed_with_failures", "preflight_completed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
