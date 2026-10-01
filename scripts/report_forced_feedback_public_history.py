"""Append the completed supplement, without changing any main-study numbers."""

from __future__ import annotations

import argparse
import html
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from llm_committee.pivot.models import digest
from llm_committee.pivot.prompts import FILLER_SENTENCE
from llm_committee.pivot.results_report import question_estimate
from llm_committee.pivot.strong_agreement import AGREEMENT
from llm_committee.pivot.study import atomic_json
from scripts.forced_feedback_public_history import FORCE, OUTPUT, VERSION
from scripts.scale_public_history import ROOT, read, sha

PAGE = ROOT / "docs/turn-tone-dyadic-mixed-family-50-public-history.html"
SECTION = "forced-disagreement-v1"
MODEL_NAMES = {"gpt-5.6-terra": "Terra", "Qwen/Qwen3.8-27B": "Qwen3.8-27B", "thinkingmachines/Inkling": "Inkling"}
MEMBER_NAMES = {0: "Terra", 1: "Qwen3.8-27B", 2: "Inkling"}
LABELS = {x: x.replace("_", " ").capitalize() for x in AGREEMENT}
ARM_NAMES = {"natural": "自然对照", "forced": "强制反驳"}


def esc(x):
    return html.escape(str(x))


def rate(n, d):
    return f"{n}/{d} ({100*n/d:.1f}%)" if d else "—"


def table(headers, rows, *, raw_columns=()):
    return (
        '<div class="ff-scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(h)}</th>" for h in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>" + "".join(f"<td>{v if i in raw_columns else esc(v)}</td>" for i, v in enumerate(row)) + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def cells(rows):
    grouped = defaultdict(list)
    for row in rows:
        if row["challenger_label"] in AGREEMENT and row["recipient_label"] in AGREEMENT:
            grouped[row["challenger_label"], row["recipient_label"]].append(row)
    return sorted(grouped.items(), key=lambda item: tuple(AGREEMENT.index(k) for k in item[0]))


def qmean(rows, field):
    values = defaultdict(list)
    for row in rows:
        if row.get(field) is not None:
            values[row["question_id"]].append(row[field])
    return mean(mean(v) for v in values.values()) if values else None


def number(x, signed=False):
    return "—" if x is None else format(x, "+.2f" if signed else ".2f")


def flatten(cases):
    rows = []
    for case in cases:
        for arm, data in case["arms"].items():
            row = {
                "question_id": case["question_id"],
                "case_id": case["id"],
                "pair": case["pair"],
                "cut_T": case["cut_T"],
                "model": case["model"],
                "arm": arm,
                **data,
            }
            prior_choice = data["position_before"]["choice"]
            row["D1_before_pp"] = (
                100 * data["D1_before"]["probabilities"][prior_choice] if data["D1_before"] and prior_choice else None
            )
            row["D1_after_pp"] = (
                100 * data["D1_after"]["probabilities"][prior_choice] if data["D1_after"] and prior_choice else None
            )
            if data["D2"]:
                row.update(
                    {
                        k: data["D2"][k]
                        for k in (
                            "modal_rating_change",
                            "reference_probability_change_pp",
                            "mean_rating_control",
                            "mean_rating_argument",
                            "mean_own_agreement_argument_minus_control",
                        )
                    }
                )
            rows.append(row)
    return rows


def analyze(output=OUTPUT):
    output = Path(output)
    manifest, run, audit = (read(output / p) for p in ("manifest.json", "report.json", "audit.json"))
    if (
        manifest["protocol_version"] != VERSION
        or manifest["kind"] == "offline_mock"
        or run["status"] not in ("completed", "completed_with_failures")
        or run["completed_questions"] != 50
    ):
        raise ValueError("Only the completed real fifty-question supplement may be published")
    records = [read(output / "questions" / f"{p['question_id']}.json") for p in manifest["plans"]]
    cases = [c for r in records for c in r["cases"]]
    if (
        audit["status"] != "passed"
        or audit.get("mock", True)
        or audit["cases_reconstructed"] != len(cases)
        or len(cases) != 300
        or audit.get("manifest_sha256") != sha(output / "manifest.json")
        or audit.get("report_sha256") != sha(output / "report.json")
        or audit.get("question_records_sha256") != digest(records)
    ):
        raise ValueError("A full matching audit is required")
    rows, summary = flatten(cases), []
    for cut in (1, 3):
        selected = [c for c in cases if c["cut_T"] == cut]
        entry = {
            "cut_T": cut,
            "cases": len(selected),
            "arms": {},
            "E_forced_wins": question_estimate(selected, "E_forced_wins"),
            "E_order_inconsistent": sum(c["E_order_inconsistent"] is True for c in selected),
            "E_complete_pairs": sum(c["E_forced_wins"] is not None for c in selected),
        }
        for arm in ("natural", "forced"):
            chosen = [r for r in rows if r["cut_T"] == cut and r["arm"] == arm]
            entry["arms"][arm] = {
                role: dict(
                    Counter(
                        r[f"{role}_label"] or "unavailable"
                        for r in chosen
                        if r["feedback" if role == "challenger" else "return"]
                    )
                )
                for role in ("challenger", "recipient")
            }
            entry["arms"][arm]["C_by_labels"] = [
                {
                    "challenger_label": labels[0],
                    "recipient_label": labels[1],
                    "n": len(group),
                    "questions": len({r["question_id"] for r in group}),
                    "option_changed": sum(r["C_choice_changed"] is True for r in group),
                    "option_comparisons": sum(r["C_choice_changed"] is not None for r in group),
                    "text_counts": dict(Counter((r["C_text"] or {}).get("label", "unavailable") for r in group)),
                }
                for labels, group in cells(chosen)
            ]
        summary.append(entry)
    result = {
        "version": VERSION,
        "run": run,
        "audit": audit,
        "summary": summary,
        "records": records,
        "cases": cases,
        "source_manifest_sha256": sha(output / "manifest.json"),
        "interpretation": "Assigned opposition with endpoint-matched archived controls; not natural strong-disagreement prevalence or isolated argument quality.",
    }
    atomic_json(output / "results.json", result)
    return result


def a_table(rows):
    values = []
    for role, title in (("challenger", "反驳者"), ("recipient", "接收者")):
        for arm in ("natural", "forced"):
            selected = [r for r in rows if r["arm"] == arm and r["feedback" if role == "challenger" else "return"]]
            counts = Counter(r[f"{role}_label"] for r in selected)
            values.append(
                [
                    f"{ARM_NAMES[arm]} · {title}",
                    len(selected),
                    *[rate(counts[label], len(selected)) for label in AGREEMENT],
                ]
            )
    return table(["分支／角色", "回复数", *LABELS.values()], values)


def b_tables(rows):
    out = []
    for arm in ("forced", "natural"):
        for role, title, key in (("challenger", "反驳者", "B_feedback"), ("recipient", "接收者", "B_return")):
            selected = [r for r in rows if r["arm"] == arm]
            values = []
            for label in AGREEMENT:
                group = [
                    r for r in selected if r[f"{role}_label"] == label and (r[key] or {}).get("label") in AGREEMENT
                ]
                counts = Counter(r[key]["label"] for r in group)
                values.append([LABELS[label], len(group), *[rate(counts[judged], len(group)) for judged in AGREEMENT]])
            missing = sum(
                (r[key] or {}).get("label") not in AGREEMENT or r[f"{role}_label"] not in AGREEMENT for r in selected
            )
            out.append(
                f"<details><summary>{ARM_NAMES[arm]} · {title}：自报与文本评审对应表</summary>"
                + table(["自报标签", "可比较回复", *["Judge: " + LABELS[label] for label in AGREEMENT]], values)
                + f"<p>另有 {missing} 条未同时取得四档内自报和文本判定，不当作一致。</p></details>"
            )
    return "".join(out)


def c_table(rows):
    result = []
    for labels, group in cells(rows):
        choices = [r["C_choice_changed"] for r in group if r["C_choice_changed"] is not None]
        text = [(r["C_text"] or {}).get("label") for r in group]
        judged = sum(t in ("unchanged", "adjusted", "conclusion_changed") for t in text)
        result.append(
            [
                LABELS[labels[0]],
                LABELS[labels[1]],
                f'{len(group)} / {len({r["question_id"] for r in group})}',
                rate(sum(choices), len(choices)),
                rate(text.count("adjusted"), judged),
                rate(text.count("conclusion_changed"), judged),
            ]
        )
    omitted = sum(r["challenger_label"] not in AGREEMENT or r["recipient_label"] not in AGREEMENT for r in rows)
    return (
        table(["反驳者标签", "接收者标签", "样本 / 题目", "选项改变", "理由／限定调整", "主要结论改变"], result)
        + f'<p class="ff-note">各项分母独立，无法比较不计为“未改变”；另有 {omitted} 条缺少完整标签组合。文本两类互斥，但不能与选项变化相加。</p>'
    )


def dots(group):
    values = [r["D1_prior_choice_change_pp"] for r in group if r["D1_prior_choice_change_pp"] is not None]
    points = "".join(
        f'<circle cx="{100+v*.86:.3f}" cy="{15+(i%5)*4}" r="2.4" fill="{"#216d87" if v<0 else "#ad6013"}"><title>{v:+.3f} pp</title></circle>'
        for i, v in enumerate(values)
    )
    return f'<svg viewBox="0 0 200 55" width="200" height="55" role="img" aria-label="Individual probability changes from minus 100 to plus 100 percentage points"><path d="M14 40H186 M100 5V40" stroke="#bac5c0"/>{points}<text x="14" y="53" font-size="10">−100</text><text x="97" y="53" font-size="10">0</text><text x="164" y="53" font-size="10">+100</text></svg>'


def d_tables(rows):
    out = []
    for model in ("Qwen/Qwen3.8-27B", "thinkingmachines/Inkling"):
        for arm in ("forced", "natural"):
            selected = [r for r in rows if r["arm"] == arm and r["model"] == model]
            d1rows, d2rows = [], []
            for labels, group in cells(selected):
                label = f"{LABELS[labels[0]]} / {LABELS[labels[1]]}"
                d1 = [r for r in group if r["D1_prior_choice_change_pp"] is not None]
                d1rows.append(
                    [
                        label,
                        f'{len(d1)} / {len({r["question_id"] for r in d1})}',
                        f'{number(qmean(d1,"D1_before_pp"))}% → {number(qmean(d1,"D1_after_pp"))}%' if d1 else "—",
                        number(qmean(d1, "D1_prior_choice_change_pp"), True),
                        dots(d1),
                    ]
                )
                d2 = [r for r in group if r.get("D2")]
                mode = [r["modal_rating_change"] for r in d2 if r.get("modal_rating_change") is not None]
                categories = " / ".join(
                    str(sum(test(v) for v in mode)) for test in (lambda v: v < 0, lambda v: v == 0, lambda v: v > 0)
                )
                prob = [r for r in d2 if r.get("reference_probability_change_pp") is not None]
                d2rows.append(
                    [
                        label,
                        len(d2),
                        f"{categories}（n={len(mode)}）",
                        f'{number(qmean(prob,"reference_probability_change_pp"),True)}（n={len(prob)}）',
                        (
                            f'{number(qmean(d2,"mean_rating_control"))} → {number(qmean(d2,"mean_rating_argument"))}'
                            if d2
                            else "—"
                        ),
                        number(qmean(d2, "mean_own_agreement_argument_minus_control"), True),
                    ]
                )
            out.append(
                f"<details><summary>{MODEL_NAMES[model]} · {ARM_NAMES[arm]} · D1 / D2</summary><h5>D1：此前选项的概率</h5>"
                + table(
                    ["反驳者 / 接收者标签", "读数 / 题目", "此前 → 本次", "变化 (pp)", "逐例分布"],
                    d1rows,
                    raw_columns=(4,),
                )
                + "<h5>D2：对自己固定旧观点的认同</h5>"
                + table(
                    [
                        "反驳者 / 接收者标签",
                        "配对读数",
                        "最可能档位：下降 / 不变 / 上升",
                        "filler 原最可能档位 Δ概率 (pp)",
                        "平均认同：filler → argument",
                        "平均认同差 (档位分)",
                    ],
                    d2rows,
                )
                + "</details>"
            )
    return "".join(out)


def probability_tables(case):
    natural, forced = (case["arms"][a] for a in ("natural", "forced"))
    source = natural.get("D1_before") or forced.get("D1_before")
    if not source:
        return "<p>该样本无可展示的 D 读数；Terra 不测 D。</p>"

    def p(dist, label):
        return f'{100*dist["probabilities"][label]:.4f}%' if dist else "—"

    d1 = table(
        ["原题选项", "此前", "自然回应输入", "强制反驳后的回应输入"],
        [
            [label, p(source, label), p(natural.get("D1_after"), label), p(forced.get("D1_after"), label)]
            for label in source["probabilities"]
        ],
    )
    groups = [(case["arms"][a].get("D2") or {}).get(t) for a in ("natural", "forced") for t in ("control", "argument")]
    d2 = table(
        ["认同档位", "自然：filler", "自然：argument", "强制：filler", "强制：argument"],
        [[f"{label} = {i}", *[p(g, label) for g in groups]] for i, label in enumerate("ABCDEFG", 1)],
    )
    return d1 + d2


def case_html(case, record, index):
    before = case["arms"]["natural"]["position_before"]
    content = (
        f'<p><b>原题：</b>{esc(record["question"]["text"])}</p><p>'
        + "；".join(f"{chr(65+i)}: {esc(t)}" for i, t in enumerate(record["question"]["options"]))
        + "</p>"
    )
    content += f'<p><b>接收者此前选项：</b>{esc(before["choice"])}</p><p class="ff-prose">{esc(before["position"])}</p>'
    tones = case.get("tones", {})
    challenger_tone = tones.get(f'{case["pair"]}-{case["cut_T"]}', "—")
    recipient_tone = tones.get(f'{case["pair"]}-{case["return_T"]}', "—")
    content += (
        f'<p>反驳者：{MEMBER_NAMES[case["challenger"]]}，T{case["cut_T"]} tone = '
        f"{esc(challenger_tone)}；"
        f'接收者：{MEMBER_NAMES[case["recipient"]]}，T{case["return_T"]} tone = '
        f"{esc(recipient_tone)}。自然与干预分支使用相同 tone。</p>"
    )
    shared = []
    for member, value in record["initial_positions"].items():
        if int(member) in (case["recipient"], case["challenger"]):
            shared.append(
                f'<h5>{MEMBER_NAMES[int(member)]} · 初始观点</h5><p class="ff-prose">{esc(value["position"])}</p>'
            )
    if case["cut_T"] == 3:
        # Natural T2 position is not the T2 public reply. Recover only the exact
        # two public prefix messages from the frozen question snapshot.
        prefix = record.get("shared_prefixes", {}).get(case["id"], [])
        for n in prefix:
            shared.append(
                f'<h5>{esc(n["node"])} · {esc(n.get("speaker", ""))} · 公开回复</h5><p class="ff-prose">{esc(n["reply"])}</p>'
            )
    content += "<details><summary>共同的初始回答与公开前缀</summary>" + "".join(shared) + "</details>"
    content += '<div class="ff-pair">'
    for arm in ("natural", "forced"):
        data = case["arms"][arm]
        content += f"<article><h4>{ARM_NAMES[arm]}</h4>"
        for role, name in (("feedback", "反驳者的消息"), ("return", "接收者的回应")):
            value = data[role]
            content += f'<h5>{name} · {LABELS.get((value or {}).get("agreement"),"未取得标签")}</h5><p class="ff-prose">{esc((value or {}).get("reply","未取得回复"))}</p>'
        ret = data["return"] or {}
        content += f'<h5>接收者的新观点 · 选项 {esc(ret.get("choice","—"))}</h5><p class="ff-prose">{esc(ret.get("position","—"))}</p>'
        content += f'<p>C 文本判定：{esc((data["C_text"] or {}).get("label","—"))}</p><p>{esc((data["C_text"] or {}).get("evidence",""))}</p></article>'
    content += "</div><details><summary>D：完整选项与认同概率</summary>" + probability_tables(case) + "</details>"
    content += "<details><summary>E：同一截止点的两份综合答案与双顺序评审</summary>"
    for arm in ("natural", "forced"):
        content += f'<h5>{ARM_NAMES[arm]}</h5><p class="ff-prose">{esc((case["arms"][arm]["synthesis"] or {}).get("answer","—"))}</p>'
    for o in case["E_orders"]:
        preferred = (
            ("强制反驳" if o["judgment"]["preference"] == o["target_side"] else "自然对照")
            if o["judgment"]
            else "未取得判断"
        )
        content += f'<p>强制答案展示在 {esc(o["target_side"])}：选择 {preferred}。{esc((o["judgment"] or {}).get("evidence",""))}</p>'
    content += "</details>"
    forced = case["arms"]["forced"]
    title = f'{index}. T{case["cut_T"]}→T{case["return_T"]} · {MODEL_NAMES[case["model"]]} · {LABELS.get(forced["challenger_label"],"—")} / {LABELS.get(forced["recipient_label"],"—")}'
    return f'<details class="ff-case" data-t="{case["cut_T"]}" data-model="{esc(MODEL_NAMES[case["model"]])}"><summary>{esc(title)} · {esc(record["question"]["text"])}</summary>{content}</details>'


def findings(result):
    rows = flatten(result["cases"])
    parts = ['<div class="ff-findings"><h3>先看这几件事</h3>']
    for summary in result["summary"]:
        cut = summary["cut_T"]
        forced = [r for r in rows if r["cut_T"] == cut and r["arm"] == "forced"]
        strong = sum(r["challenger_label"] == "strongly_disagree" for r in forced)
        leaning = [r for r in forced if r["challenger_label"] == "leaning_disagree"]
        judged = [r for r in leaning if (r["B_feedback"] or {}).get("label") in AGREEMENT]
        judge_strong = sum(r["B_feedback"]["label"] == "strongly_disagree" for r in judged)
        parts.append(
            f"<p><b>T{cut}→T{cut+1}：</b>反驳者自报 strongly disagree 为 {rate(strong,len(forced))}。"
            f"自报 leaning disagree 的回复中，Gemini 将文字判成 strongly disagree 的有 {rate(judge_strong,len(judged))}。"
            "所以“要求明确反对”、“自报强烈反对”和“judge 判为强烈反对”不能当作同一件事。</p>"
        )
        groups = cells(forced)
        if groups:
            labels, group = max(groups, key=lambda item: len(item[1]))
            choices = [r["C_choice_changed"] for r in group if r["C_choice_changed"] is not None]
            texts = [(r["C_text"] or {}).get("label") for r in group]
            ntext = sum(t in ("unchanged", "adjusted", "conclusion_changed") for t in texts)
            parts.append(
                f"<p>该时间点最常见的组合是 <b>{LABELS[labels[0]]} / {LABELS[labels[1]]}</b>（{len(group)} 例）："
                f'接收者改变选项 {rate(sum(choices),len(choices))}，调整理由或限定 {rate(texts.count("adjusted"),ntext)}，'
                f'改变主要结论 {rate(texts.count("conclusion_changed"),ntext)}。这是该标签组合的描述，不是所有情形的平均，也不是独立的因果分组。</p>'
            )
    parts.append(
        "<p><b>D 要分开读：</b>选项不变，不代表概率不变；D1 测旧选项的概率，D2 测对完整旧观点的认同。两者可能方向不同。"
        "特别是早期 D1 的“此前”尚未见到公开讨论，“本次”已见到讨论，不能把这段变化全部归因于强制反驳；自然分支提供同截止点参照。</p>"
    )
    for summary in result["summary"]:
        estimate = summary["E_forced_wins"]
        if estimate["ci"] is None:
            continue
        low, high = estimate["ci"]
        interpretation = (
            "区间包含 50%，偏好方向尚不明确"
            if low <= 0.5 <= high
            else "这是该 judge 在本样本中的偏好，不等于普遍质量提升"
        )
        parts.append(
            f'<p><b>E · T{summary["cut_T"]}→T{summary["cut_T"]+1}：</b>强制反驳答案相对自然讨论答案的 wins 为 '
            f'{100*estimate["mean"]:.1f}%（95% 区间 {100*low:.1f}–{100*high:.1f}%）；{interpretation}。</p>'
        )
    return "".join(parts) + "</div>"


def render(result):
    rows = flatten(result["cases"])
    run = result["run"]
    parts = [
        f'<section id="{SECTION}"><h2>Supplementary experiment · Forced disagreement</h2>',
        "<p>如果给成员明确的“实质性反对”任务，而不只是 hostile tone，收到反驳的另一成员会怎样回应、是否改变立场？</p>",
        "<p>50 题，300 个干预样本（不是 300 个独立题目），不同分支的讨论不互相混入。三个定向配对：Terra → Qwen → Terra、Inkling → Terra → Inkling、Qwen → Inkling → Qwen。每题、每个配对各有早期和后期干预。T1 是首次 debate，初始观点记为 T0。早期在 T1 反驳、T2 回应后停止；后期复用原 T1–T2，在 T3 反驳、T4 回应后停止。自然对照取相同截止点。</p>",
        f'<p>已完成 {run["completed_cases"]}/300 个案例，其中 {run["successful_cases"]} 个取得全部计划测量。反驳者有私有任务，接收者没有；既有逐轮 tone 保持不变。strongly disagree 不是强制填写的标签，未达到它的回复也全部保留。</p>',
        "<p><b>阅读顺序：</b>A 看反驳是否被表达出来；B 看文字是否支持标签；C/D 按“反驳者标签 / 接收者标签”看立场变化；E 看这种干预后的答案是否更受偏好。下列结果与上方主实验分开。</p>",
        f"<details><summary>精确干预指令、费用与边界</summary><blockquote>{esc(FORCE)}</blockquote>"
        f"<p>D2 的 control 重复以下原文，近似匹配 argument 的 token 长度：</p><blockquote>{esc(FILLER_SENTENCE)}</blockquote>"
        f'<p>新增成功调用 {sum(run["completed_new_calls_by_purpose"].values())} 次，复用原测量 {run["reused_measurements"]} 份。收到响应的费用估算 ${run["cost_accounting"]["received_response_estimate_usd"]:.2f}；未知计费预留 ${run["cost_accounting"]["unresolved_reservations_usd"]:.2f}，不是确认费用或供应商账单。</p>'
        "<p>这是指定反对立场后的干预，不是自然强烈反对的发生率。它同时可能改变论点、措辞、长度和论据，不能归因为某一个孤立因素。C/D 按事后标签分组：两分支同一个标签格子可能包含不同案例，不是配对因果效应。缺失和无法判断保留在数据中，不填成零。</p></details>",
        findings(result),
    ]
    for cut in (1, 3):
        selected = [r for r in rows if r["cut_T"] == cut]
        timing = "初始观点 → T2" if cut == 1 else "自己的 T2 观点 → T4"
        parts.append(
            f'<details class="ff-timing" open><summary>T{cut}→T{cut+1} · {"早期" if cut==1 else "后期"}干预 · 150 个样本</summary><h3>A · 两个角色各自如何描述态度？</h3>'
            + a_table(selected)
        )
        parts.append(
            "<h3>B · 自报与文本是否对应？</h3><p>一个外部 judge：Gemini 3.8 Flash。只看原题、被回复的消息和回复，不看自报、tone 或私有干预指令。每行以自报标签分组，不用一个全局一致率替代。本补充未做人评；judge 与自报不一致，不能单独确定哪一方判错。</p>"
            + b_tables(selected)
        )
        parts.append(
            f"<h3>C · 接收者自己的观点是否变化？</h3><p>比较 {timing}，不是比较两位成员的观点。选项改变由程序直接检查；理由／限定调整和主要结论改变由 Gemini 比较完整观点文本判定。下表每个格子都保留实际标签组合。</p>"
        )
        for arm in ("forced", "natural"):
            parts.append(
                f'<details {"open" if arm=="forced" else ""}><summary>{ARM_NAMES[arm]} · C</summary>'
                + c_table([r for r in selected if r["arm"] == arm])
                + "</details>"
            )
        parts.append(
            "<h3>D · 相同节点的概率补充读数</h3><p>只测 Qwen、Inkling。正式讨论开启 reasoning；D 是独立、neutral、关闭 reasoning 的读取，不是正式回答的生成概率。D1 比较接收者此前选项的概率：正值表示该选项更可能，负值表示更不可能。点图保留逐例变化，避免正负相抵被均值掩盖。</p>"
            "<p>D2 固定接收者之前的完整观点，只把 incoming argument 换成近似等 token 长度、不谈题目内容的程序性 filler。A–G 对应 1–7 档认同，1 最不同意、7 最同意。依次报告：① 最可能档位下降／不变／上升的例数；② filler 下最可能档位在 argument 下的概率变化；③ 所有档位加权的平均认同变化。比如 7→6 是档位下降，70%→40% 是 −30 pp，6.50→6.35 是平均认同下降 0.15 档，不是 15% 的样本改变。② 负值只表示原最可能档位变得不那么可能，不一定是认同下降；③ 负值才表示平均认同减少。它们都不能自动解读成朝对方立场移动。并列最高档不随意打破。</p>"
            "<p>数值均值先在每题、每个标签组内平均，再对题目等权；概率差单位为百分点（pp）。下列小样本格子只作描述。</p>"
            + d_tables(selected)
            + "</details>"
        )
    e_rows = []
    for s in result["summary"]:
        estimate = s["E_forced_wins"]
        ci = estimate["ci"]
        win = f'{100*estimate["mean"]:.1f}% [{100*ci[0]:.1f}, {100*ci[1]:.1f}]' if ci else "—"
        selected = [c for c in result["cases"] if c["cut_T"] == s["cut_T"]]
        lengths = []
        for arm in ("natural", "forced"):
            words = [
                len(c["arms"][arm]["synthesis"]["answer"].split()) for c in selected if c["arms"][arm]["synthesis"]
            ]
            lengths.append(f"{mean(words):.1f}" if words else "—")
        e_rows.append(
            [
                f'T{s["cut_T"]}→T{s["cut_T"]+1}',
                f'{s["E_complete_pairs"]}/{s["cases"]}',
                win,
                f'{s["E_order_inconsistent"]}/{s["E_complete_pairs"]}',
                " / ".join(lengths),
            ]
        )
    parts.append(
        "<h3>E · 强制反驳后的综合答案是否更受偏好？</h3><p>同一 Terra chairman，在各组相同截止点综合完整公开讨论，均限制 190–210 词。Gemini 用主实验原 rubric 盲评两个展示顺序。Forced wins 是选择干预答案的比例，不是质量提升幅度：每对在两种展示顺序下的 0/1 选择先平均，再按题目平均。两次选强制反驳得 1，改选得 0.5，两次选自然得 0；0.5 不是真实平局。对照是自然讨论，不是 no-debate。区间按 50 个题目 bootstrap，同题模型配对先平均。</p>"
        + table(["干预位置", "答案对", "Forced wins [95% 区间]", "顺序不一致", "平均词数：自然 / 强制"], e_rows)
        + "<p>没有为本补充另跑 damage battery 或人评；原诊断只提供背景，不当作新增验证。</p>"
    )
    parts.append(
        '<details><summary>逐例查看：全部 300 个自然／干预配对</summary><p>标签顺序始终是反驳者 / 接收者；每条都显示原题、共同上下文、回复、观点和概率分布。</p><label>时间 <select class="ff-filter-time"><option value="">全部</option><option value="1">T1→T2</option><option value="3">T3→T4</option></select></label> <label>接收者 <select class="ff-filter-model"><option value="">全部</option>'
        + "".join(f"<option>{esc(m)}</option>" for m in MODEL_NAMES.values())
        + "</select></label>"
    )
    i = 0
    for record in result["records"]:
        for case in record["cases"]:
            i += 1
            parts.append(case_html(case, record, i))
    parts.append("</details></section>")
    return "".join(parts)


CSS = f"""#{SECTION}{{border-top:3px solid #267366;margin-top:36px;padding-top:22px}}#{SECTION} .ff-scroll{{overflow-x:auto}}#{SECTION} table{{width:100%;border-collapse:collapse;margin:14px 0}}#{SECTION} th,#{SECTION} td{{padding:9px;border-bottom:1px solid #d5ddd8;text-align:left;vertical-align:top}}#{SECTION} details{{margin:14px 0;padding:13px;border:1px solid #d5ddd8;border-radius:7px}}#{SECTION} summary{{cursor:pointer;font-weight:650}}#{SECTION} .ff-pair{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}#{SECTION} .ff-prose{{white-space:pre-wrap}}#{SECTION} .ff-note{{font-size:13px;color:#596d62}}#{SECTION} select{{font:inherit;padding:6px}}#{SECTION} [hidden]{{display:none!important}}#{SECTION} article{{min-width:0;overflow-wrap:anywhere}}@media(max-width:720px){{#{SECTION} .ff-pair{{grid-template-columns:1fr}}#{SECTION} th,#{SECTION} td{{padding:6px;font-size:13px}}}}"""
JS = f"""(() => {{const root=document.getElementById('{SECTION}');if(!root)return;const time=root.querySelector('.ff-filter-time'),model=root.querySelector('.ff-filter-model');function filter(){{root.querySelectorAll('.ff-case').forEach(c=>c.hidden=!!((time.value&&time.value!==c.dataset.t)||(model.value&&model.value!==c.dataset.model)));}}time.addEventListener('change',filter);model.addEventListener('change',filter);}})();"""


def attach(document, section):
    """Only insert removable blocks; keep every byte of the main HTML intact."""
    document = re.sub(r"<!-- ff-supplement:(\w+):start -->.*?<!-- ff-supplement:\1:end -->", "", document, flags=re.S)
    if f'id="{SECTION}"' in document:
        raise ValueError("Unmarked existing supplement; refusing to overwrite it")
    additions = [
        ("style", "</head>", f'<style id="ff-supplement-style">{CSS}</style>'),
        ("nav", "</nav>", f'<a href="#{SECTION}" id="ff-supplement-link"> · Supplement: forced disagreement</a>'),
        ("section", '<section id="paired">' if '<section id="paired">' in document else "</main>", section),
        ("script", "</body>", f'<script id="ff-supplement-script">{JS}</script>'),
    ]
    for name, anchor, content in additions:
        if document.count(anchor) != 1:
            raise ValueError(f"Ambiguous report insertion point: {anchor}")
        block = f"<!-- ff-supplement:{name}:start -->{content}<!-- ff-supplement:{name}:end -->"
        document = document.replace(anchor, block + anchor, 1)
    return document


def publish(output=OUTPUT, page=PAGE, *, refresh_dashboard=True):
    output, page = Path(output), Path(page)
    result = analyze(output)
    contexts = read(output / "source-contexts.json")
    # Add only public shared prefixes for qualitative display, not private state.
    for record in result["records"]:
        source = contexts[record["question_id"]]["record"]
        record["shared_prefixes"] = {
            c["id"]: [
                {
                    "node": f'{c["pair"]}-{t}',
                    "speaker": MEMBER_NAMES[c["challenger"] if t % 2 else c["recipient"]],
                    "reply": source["branches"][c["source_assignment"]]["formal_replies"]["turn_level"][
                        f'{c["pair"]}-{t}'
                    ]["reply"],
                }
                for t in range(1, c["cut_T"])
            ]
            for c in record["cases"]
        }
    for suffix in (".html", ".json"):
        backup = output / f"base-report-before-supplement{suffix}"
        if not backup.exists():
            shutil.copyfile(page.with_suffix(suffix), backup)
    page.write_text(attach(page.read_text(), render(result)))
    data = read(page.with_suffix(".json"))
    data.setdefault("supplements", {})["forced_disagreement"] = {
        "version": VERSION,
        "summary": result["summary"],
        "run": result["run"],
        "audit": result["audit"],
        "results_sha256": sha(output / "results.json"),
    }
    atomic_json(page.with_suffix(".json"), data)
    # Base observations must remain byte-equivalent as structured data.
    before = read(output / "base-report-before-supplement.json")
    assert digest({k: v for k, v in before.items() if k != "supplements"}) == digest(
        {k: v for k, v in data.items() if k != "supplements"}
    )
    if refresh_dashboard:
        from scripts.report_experiment_dashboard import publish as dashboard

        dashboard()
    return page.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--page", type=Path, default=PAGE)
    args = parser.parse_args()
    print(publish(args.output, args.page))
