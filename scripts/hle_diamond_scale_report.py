"""Size-aware private report for the HLE continuation, including fallback provenance."""

from __future__ import annotations

import html
import json
from collections import Counter, defaultdict
from pathlib import Path

from llm_committee.pivot.models import ROSTERS
from llm_committee.pivot.study import atomic_json
from scripts.hle_diamond_study import PairedGraph, read
from scripts.supergpqa_pilot import LABELS, NAMES, summaries


def publish(directory, page):
    from scripts.hle_diamond_scale import LOW_SUFFIX, all_rows, fallback_summary

    directory, page = Path(directory).resolve(), Path(page).resolve()
    manifest, progress = read(directory / "manifest.json"), read(directory / "progress.json")
    records = [
        read(p) for plan in manifest["plans"] if (p := directory / "questions" / f"{plan['question_id']}.json").exists()
    ]
    rows = all_rows(directory)
    saved = {
        json.loads(r["request"])["key"].removesuffix(LOW_SUFFIX): {
            **r,
            "request": json.loads(r["request"]),
            "response": json.loads(r["response"]),
            "parsed": json.loads(r["parsed"]),
        }
        for r in rows
        if r["status"] == "completed"
    }
    result = summaries(records)
    n = manifest["planned_counts"]["questions"]
    result.update(
        source=str(directory),
        progress=progress,
        target_questions=n,
        reasoning_policy=manifest["execution"]["qwen_low_fallback"],
        qwen_low_fallback=fallback_summary(rows),
        new_received_response_estimate_usd=sum(
            r["charge"] for r in rows if r["response"] and r["status"] != "billing_unknown"
        ),
        unresolved_reservations_usd=sum(
            r["charge"] for r in rows if not r["response"] or r["status"] == "billing_unknown"
        ),
        original_pilot_cost_estimate_usd=manifest["continuation"]["source_cost_estimate_usd"],
    )
    tones = defaultdict(Counter)
    for r in records:
        for event in r["events"]:
            if event["after"]:
                tones[event["current_tone"]][event["current_label"]] += 1
    result["labels_by_current_tone"] = {k: dict(v) for k, v in tones.items()}
    atomic_json(page.with_suffix(".json"), result)

    def esc(value):
        return html.escape(str(value))

    def ratio(a, b):
        return f"{a}/{b} ({100 * a / b:.1f}%)" if b else "—"

    def table(headers, rows):
        return (
            '<div class="scroll"><table><thead><tr>'
            + "".join(f"<th>{esc(v)}</th>" for v in headers)
            + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
            + "</tbody></table></div>"
        )

    def prompt(key):
        item = saved.get(key)
        if not item:
            return "<p>未完成，不用默认答案代替。</p>"
        req, response = item["request"], item["response"]
        prefix = "<details><summary>完整 prompt / output / 概率读取</summary>"
        prefix += f"<p>effort = {esc(req['effort'])}；输出 {response['output_tokens']} tokens（含推理）。"
        if req["key"].endswith(LOW_SUFFIX):
            prefix += "这是 medium 长度失败后的 low 回退，非原始 medium 成功结果。"
        prefix += "</p>"
        for msg in req["messages"]:
            prefix += f"<h4>{esc(msg['role'])}</h4><pre>{esc(msg['text'])}</pre>"
        prefix += f"<h4>可见输出</h4><pre>{esc(response['text'])}</pre>"
        if response.get("readout"):
            prefix += f"<h4>Token / logprob</h4><pre>{esc(json.dumps(response['readout'], ensure_ascii=False, indent=2))}</pre>"
        return prefix + "</details>"

    fb = result["qwen_low_fallback"]
    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>HLE-Diamond · {n} questions</title>",
        "<style>body{font:16px/1.65 system-ui;color:#23343e;background:#f5f7f8;margin:0}main{max-width:1160px;margin:auto;padding:28px}h2{margin-top:34px}table{border-collapse:collapse;width:100%;font-size:14px;background:white}th,td{padding:10px;border-bottom:1px solid #dce3e8;text-align:left}th{background:#e8eff3}.scroll{overflow:auto}details{background:white;border:1px solid #dce3e8;border-radius:8px;margin:12px 0;padding:14px}summary{cursor:pointer;font-weight:650}pre,.text{white-space:pre-wrap;overflow-wrap:anywhere}pre{font-size:13px}.note{background:#fff4d3;padding:14px}nav{display:flex;gap:18px;flex-wrap:wrap}a{color:#126577}</style><main>",
        f"<h1>HLE-Diamond · {n} 题</h1>",
        '<p class="note">私有研究报告，包含 gated benchmark 原文。请勿公开、上传或提交到 Git。</p>',
        f"<p>状态：{esc(progress['status'])}；报告已汇入 {len(records)}/{n} 题；完整路径 {result['coverage']['complete_trajectories']}/{n * 6}。未完成题目尚不进入下列表格。</p>",
        '<nav><a href="#e">E 答案正确率</a><a href="#a">A 自报态度</a><a href="#c">C 选项变化</a><a href="#d">D1 概率移动</a><a href="#cases">完整案例</a><a href="#setup">执行设置</a></nav>',
        "<p>保留原 20 题，从剩余 55 道合格题中固定随机增补 30 题；不按模型表现筛选。范围是 HLE-Diamond 的纯文本 reasoning 选择题子集，不是完整 HLE 得分。</p>",
        "<p>A = Terra，B = Qwen3.8-27B，C = Inkling。每题 AB、CA、BC 三组双人讨论，每条路径四轮。复用 GlobalOpinionQA 的 friendly / neutral / hostile 逐轮指令；两版续写共享 T1/T2，在 T3/T4 分别更换 tone。不是两种 tone，也不是 global-tone 分组。</p>",
        f'<p class="note">正式解题默认 medium；Qwen 耗尽输出长度且常规重试失败后，改为 low 重试。已有 {fb["triggered_requests"]} 个请求触发回退，{fb["recovered_requests"]} 个恢复。报告评价的是这套 medium-first 策略，不能标成所有请求均为 medium。D1 始终 reasoning-off。</p>',
        "<h2>独立初答与讨论后的成员答案</h2>",
        f"<p>正确 = 所选字母命中官方答案键，不用偏好 judge。每模型最多 {n} 份独立初答；每题参与两组搭配、各两版续写，最多 {4 * n} 个路径末次答案。同题的多个末次答案不是独立题。</p>",
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
        '<section id="e"><h2>E：同一 chairman，只有初答 vs 加入讨论</h2>',
        "<p>同一对成员的初答和无讨论 synthesis 各生成一次。无讨论答案与两版讨论答案分别配对；下表合并两版。分母是配对路径数，不是独立题数。“改对”指无讨论答错、讨论后答对；“改错”相反。更多计算与互动的作用未被单独隔离。</p>",
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
        '</section><section id="a"><h2>A：本轮 tone 与自报态度</h2>',
        "<p>共享前缀只数一次。label 表示对收到消息的态度，不等于自己的选项是否改变。</p>",
        table(
            ["本轮 tone", "回复数", *[LABELS[k] for k in LABELS if k]],
            [
                [tone, sum(counts.values()), *[ratio(counts.get(k, 0), sum(counts.values())) for k in LABELS if k]]
                for tone, counts in sorted(tones.items())
            ],
        ),
        '</section><section id="c"><h2>C：按前后 label 看自己的选项变化</h2>',
        "<p>按 [上条回复 label，本条回复 label] 分组，合并 T2–T4。比较同一成员本次与自己上次参与时的选择；首次参与参照其初答。T1 没有前一条 debate label，不放进此表。“改对”的分母是先前答错的次数，“改错”的分母是先前答对的次数。这里不使用文本 judge。</p>",
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
        '</section><section id="d"><h2>D1：选项概率分布移动</h2>',
        "<p>对正式回复的同一上下文另做 neutral、reasoning-off 选项读取，不提供本次生成的答案。只在原生选项上归一化，top-20 未出现的选项记为零。</p>",
        "<p><b>TV = ½Σ|p后−p前|</b>：0% 表示完全不动，100% 表示支持完全不重叠；不是准确率。<b>ΔP(正确答案)</b> 是正确选项概率的百分点变化，例如 40%→50% = +10 pp；正数表示更支持正确答案。</p>",
        '<p class="note">首次参与的前测只有题目，后测还含自己的 reasoning-on 初答和收到的消息，因此不能把首次跳变全归因于 peer persuasion。T3 的两版输入相同，D1 只读一次；按两版正式回复 label 分类时可能重复出现，不是独立测量。</p>',
    ]
    for model in ROSTERS["mixed_family"][1:]:
        parts += [
            f"<h3>{esc(NAMES[model])}</h3>",
            table(
                ["上条 label", "本条 label", "次数", "平均 TV", "中位 TV", "平均 ΔP(正确答案)"],
                [
                    [
                        LABELS[r["previous_label"]],
                        LABELS[r["current_label"]],
                        r["n"],
                        f"{100 * r['mean_tv']:.1f}%",
                        f"{100 * r['median_tv']:.1f}%",
                        f"{r['mean_correct_probability_change_pp']:+.1f} pp",
                    ]
                    for r in result["D1_conditional"]
                    if r["model"] == model
                ],
            ),
        ]
    parts.append('</section><section id="cases"><h2>原题、完整轨迹与请求（仅本地）</h2>')
    for i, record in enumerate(records, 1):
        plan = next(p for p in manifest["plans"] if p["question_id"] == record["question"]["id"])
        graph = PairedGraph(record, plan, manifest)
        parts += [
            f"<details><summary>Q{i:02d} · {esc(record['domain'])} · {esc(record['question']['text'][:100])}</summary>",
            f"<pre>{esc(record['original_question'])}</pre><p>官方答案（只用于评分）：{esc(record['answer_letter'])}</p>",
        ]
        for member, value in record["initial_positions"].items():
            key = graph.key("original", f"initial/{member}")
            effort = saved[key]["request"]["effort"]
            parts.append(
                f"<details><summary>初答 · {esc(NAMES[ROSTERS['mixed_family'][int(member)]])} · {esc(value['choice'])} · {effort}</summary><pre>{esc(value['position'])}</pre>{prompt(key)}</details>"
            )
        for pair in ("AB", "CA", "BC"):
            parts.append(f"<details><summary>{pair} · 共享 T1/T2 与两版后续</summary>")
            for event in (e for e in record["events"] if e["pair"] == pair):
                arm = "shared" if event["T"] <= 2 else event["arm"]
                parts.append(
                    f"<h3>{arm} T{event['T']} · {esc(NAMES[event['model']])} · {esc(event['current_tone'])}</h3>"
                )
                value = event["after"]
                if value is None:
                    parts.append("<p>缺失，不作为错误或 agreement。</p>")
                    continue
                before = event["before"]["choice"] if event["before"] else "—"
                parts += [
                    f"<p>{esc(LABELS[value['agreement']])}；自己的选择 {esc(before)} → {esc(value['choice'])}</p>",
                    f"<pre>{esc(value['reply'])}</pre><details><summary>同次生成的 position</summary><pre>{esc(value['position'])}</pre></details>",
                    prompt(event["request_key"]),
                ]
                if event["D1"]:
                    parts.append("<details><summary>D1 前后全分布</summary>")
                    parts.append(
                        table(
                            ["选项", "前测", "后测"],
                            [
                                [
                                    label,
                                    f"{100 * event['D1_before']['probabilities'][label]:.5f}%",
                                    f"{100 * event['D1_after']['probabilities'][label]:.5f}%",
                                ]
                                for label in graph.question.labels
                            ],
                        )
                    )
                    for side in ("before", "after"):
                        parts += [f"<h4>{side}</h4>", prompt(event[f"D1_{side}_request"])]
                    parts.append("</details>")
            for arm in ("original", "alternate"):
                quality = next(q for q in record["quality"] if q["pair"] == pair and q["arm"] == arm)
                for which in ("baseline", "debate") if arm == "original" else ("debate",):
                    if value := quality[which]:
                        parts.append(
                            f"<details><summary>Chairman · {arm} / {which} · {esc(value['choice'])}</summary><pre>{esc(value['position'])}</pre></details>"
                        )
            parts.append("</details>")
        parts.append("</details>")
    counts = manifest["planned_counts"]["by_purpose"]
    parts += [
        '</section><section id="setup"><h2>执行设置与成本</h2>',
        f"<p>计划 {counts['initial']} 初答 + {counts['debate']} 回复 + {counts['d_choice']} D1 + {counts['synthesis']} synthesis = {n * 42} 个逻辑请求，另计失败重试。原 pilot 的 {manifest['continuation']['successful_requests_reused']} 个成功请求原样复用，未付费重跑。</p>",
        "<p>初答、debate、chairman：medium，16,384-token 输出上限（含 reasoning）。Qwen 常规三次尝试仍以长度耗尽失败时，最多再尝试三次 low；不是因答错而重试，也不选择最好的答案。其他模型不改 effort。D1 始终 reasoning-off，128-token 上限。未导入之前独立 low 诊断的结果。</p>",
        "<p>无工具、无网络搜索，无 B/C 文本 judge，无 D2，无额外独立重复。题目才是独立抽样单位，不能把多条轨迹当作独立题。</p>",
        f"<p>本次新增收到响应的费用估算：US${result['new_received_response_estimate_usd']:.3f}；未确认计费预留：US${result['unresolved_reservations_usd']:.3f}。原 pilot 另计 US${result['original_pilot_cost_estimate_usd']:.3f}。均为 token 估算，不是供应商账单。</p>",
        f"<details><summary>Low 回退记录</summary><pre>{esc(json.dumps(fb, ensure_ascii=False, indent=2))}</pre></details>",
        f"<details><summary>未恢复的失败</summary><pre>{esc(json.dumps(progress.get('task_failures', {}), ensure_ascii=False, indent=2))}</pre></details>",
        f"<p>原始数据：{esc(directory)}</p></section></main></html>",
    ]
    page.parent.mkdir(parents=True, exist_ok=True)
    page.write_text("".join(parts))
    return result
