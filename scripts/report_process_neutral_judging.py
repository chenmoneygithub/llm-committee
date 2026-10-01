"""Compare frozen v1/v2 judgments, pooled within questions; no model dispatch."""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from bs4 import BeautifulSoup

from llm_committee.pivot import prompts
from llm_committee.pivot.databricks_provider import ENDPOINTS, databricks_payload
from llm_committee.pivot.models import digest
from llm_committee.pivot.results_report import question_estimate
from llm_committee.pivot.study import atomic_json
from scripts.grok_quality_check import request_from
from scripts.human_answer_review import OUTPUT as HUMAN_PAGE
from scripts.human_answer_review import PRIVATE
from scripts.prepare_process_neutral_judging import ADDENDUM, OUTPUT, VERSION
from scripts.run_process_neutral_judging import execution_report
from scripts.scale_public_history import ROOT, freeze, read, sha

ANNOTATION = Path("/Users/Chen/Downloads/answer-review-money-2855f1a7.json")
PAGE = ROOT / "docs/E2-judge-rubric-comparison-2026-09-28.html"
SETTING_PAGE = ROOT / "docs/turn-tone-triadic-mixed-family-50-public-history.html"
MODELS = {"gemini-3.8-flash": "Gemini", "grok-4-6": "Grok"}
RESPONSE_MODELS = {"gemini-3.8-flash": "gemini-3.8-flash", "grok-4-6": "global.xai.grok-4.6"}


def summarize_rows(rows):
    summaries = []
    for model in MODELS:
        selected = [r for r in rows if r["model"] == model]
        summaries.append(
            {
                "model": model,
                "pairs": len(selected),
                "decisions": 2 * len(selected),
                **{name: question_estimate(selected, name) for name in ("old_score", "new_score", "delta")},
                "old_debate_votes": sum(sum(r["old_votes"]) for r in selected),
                "new_debate_votes": sum(sum(r["new_votes"]) for r in selected),
                "changed_decisions": sum(
                    old != new for r in selected for old, new in zip(r["old_votes"], r["new_votes"], strict=True)
                ),
                "debate_to_baseline": sum(
                    old and not new for r in selected for old, new in zip(r["old_votes"], r["new_votes"], strict=True)
                ),
                "baseline_to_debate": sum(
                    not old and new for r in selected for old, new in zip(r["old_votes"], r["new_votes"], strict=True)
                ),
                "changed_pair_verdicts": sum(r["old_score"] != r["new_score"] for r in selected),
                "old_order_inconsistent": sum(r["old_score"] == 0.5 for r in selected),
                "new_order_inconsistent": sum(r["new_score"] == 0.5 for r in selected),
            }
        )
    return summaries


def human_comparison(annotation, key, rows, questions, translations):
    if annotation["dataset_id"] != key["dataset_id"] or annotation["version"] != "blind-answer-review-v1":
        raise ValueError("Wrong human review dataset")
    responses = {r["item_id"]: r for r in annotation["responses"]}
    expected = {k["item_id"] for k in key["items"]}
    if len(responses) != len(annotation["responses"]) or not set(responses) <= expected:
        raise ValueError("Duplicate or unknown annotation")
    model_rows = {(r["question_id"], r["assignment"], r["model"]): r for r in rows}
    cases, metrics = [], defaultdict(list)
    for index, k in enumerate(key["items"], 1):
        h = responses.get(k["item_id"])
        if h is None:
            continue
        if h["preference"] not in ("A", "B", "tie", "unjudgeable", None):
            raise ValueError("Invalid human preference")
        q = questions[k["question_id"]]
        pair = next(r for r in q["E2"] if r["assignment"] == k["assignment"])
        for label in ("baseline", "debate"):
            text = pair["baseline" if label == "baseline" else "debated"]["answer"]
            if digest(text) != k[f"{label}_sha256"]:
                raise ValueError("Human annotation refers to different answers")
        baseline_side = "B" if k["debate_side"] == "A" else "A"
        case = {
            "number": index,
            "question_id": k["question_id"],
            "assignment": k["assignment"],
            "question": q["question"]["text"],
            "answers": {k["debate_side"]: pair["debated"]["answer"], baseline_side: pair["baseline"]["answer"]},
            "chinese": translations[k["item_id"]],
            "human_preference": h["preference"],
            "rating_language": h.get("rating_language", "en"),
            "human_debate": h["preference"] == k["debate_side"] if h["preference"] in ("A", "B") else None,
            "judges": {},
        }
        displayed_target = "left" if k["debate_side"] == "A" else "right"
        for model in MODELS:
            row = model_rows[k["question_id"], k["assignment"], model]
            orders = []
            for order in row["orders"]:
                orders.append(
                    {
                        "order": order["order"],
                        "same_order_as_human": order["target_side"] == displayed_target,
                        **{
                            version: {
                                "choice": (
                                    k["debate_side"]
                                    if order[version]["preference"] == order["target_side"]
                                    else baseline_side
                                ),
                                "evidence": order[version]["evidence"],
                            }
                            for version in ("old", "new")
                        },
                    }
                )
            case["judges"][model] = orders
            if h["preference"] not in ("A", "B"):
                continue
            matched = next(o for o in orders if o["same_order_as_human"])
            metric = {"question_id": k["question_id"]}
            for version in ("old", "new"):
                metric[f"{version}_match"] = matched[version]["choice"] == h["preference"]
                metric[f"{version}_agree_orders"] = sum(o[version]["choice"] == h["preference"] for o in orders)
            metric["delta"] = int(metric["new_match"]) - int(metric["old_match"])
            metrics[model].append(metric)
        cases.append(case)
    summary = []
    for model in MODELS:
        sample = metrics[model]
        row = {"model": model, "binary_ratings": len(sample), "paired_change": question_estimate(sample, "delta")}
        for version in ("old", "new"):
            row[version] = {
                "matched_order_agreement": sum(r[f"{version}_match"] for r in sample),
                "both_orders_agree": sum(r[f"{version}_agree_orders"] == 2 for r in sample),
                "both_orders_disagree": sum(r[f"{version}_agree_orders"] == 0 for r in sample),
                "order_inconsistent": sum(r[f"{version}_agree_orders"] == 1 for r in sample),
            }
        summary.append(row)
    return {
        "dataset_id": key["dataset_id"],
        "ratings": len(cases),
        "excluded_nonbinary": sum(c["human_preference"] not in ("A", "B") for c in cases),
        "language_counts": dict(Counter(c["rating_language"] for c in cases)),
        "human_debate_wins": sum(c["human_debate"] is True for c in cases),
        "summary": summary,
        "cases": cases,
    }


def analyze(output=OUTPUT, annotation_path=ANNOTATION):
    output = Path(output)
    execution = execution_report(output)
    if execution["status"] != "completed":
        raise ValueError("Do not publish an incomplete judging comparison")
    prepared, tasks = read(output / "manifest.json"), read(output / "tasks.json")
    old = read(Path(prepared["previous_judge_directory"]) / "report.json")
    prior = {(r["question_id"], r["assignment"]): r for r in old["rows"]}
    values = {}
    with sqlite3.connect(f"file:{output}/requests.sqlite3?mode=ro", uri=True) as db:
        for raw, response, parsed in db.execute("SELECT request,response,parsed FROM calls WHERE status='completed'"):
            request, completion, judgment = map(json.loads, (raw, response, parsed))
            if completion["raw"]["request_payload"] != databricks_payload(request_from(request)):
                raise ValueError("Actual provider payload differs from the frozen request")
            if (
                completion["raw"]["http_status"] != 200
                or completion["status"] != "completed"
                or completion["raw"]["provider"] != "databricks"
                or completion["raw"]["endpoint"] != ENDPOINTS[request["model"]]
                or completion["raw"]["response"].get("model") != RESPONSE_MODELS[request["model"]]
                or prompts.parse_json(completion["text"], prompts.E_SCHEMA) != judgment
            ):
                raise ValueError("Judgment does not match the successful raw response")
            values[request["key"]] = judgment
    grouped = {}
    for task in tasks:
        model = task["request"]["model"]
        key = task["question_id"], task["assignment"], model
        row = grouped.setdefault(key, {"question_id": key[0], "assignment": key[1], "model": model, "orders": []})
        old_order = next(o for o in prior[key[0], key[1]]["orders"] if o["order"] == task["order"])
        if old_order["target_side"] != task["target_side"]:
            raise ValueError("Answer order mapping changed")
        row["orders"].append(
            {
                "order": task["order"],
                "target_side": task["target_side"],
                "old": old_order["gemini" if model.startswith("gemini") else "grok"],
                "new": values[task["request"]["key"]],
            }
        )
    rows = list(grouped.values())
    for row in rows:
        row["orders"].sort(key=lambda o: o["order"])
        if [o["order"] for o in row["orders"]] != [0, 1] or {o["target_side"] for o in row["orders"]} != {
            "left",
            "right",
        }:
            raise ValueError("Incomplete or invalid two-order pair")
        for version in ("old", "new"):
            row[f"{version}_votes"] = [o[version]["preference"] == o["target_side"] for o in row["orders"]]
            row[f"{version}_score"] = sum(row[f"{version}_votes"]) / 2
        row["delta"] = row["new_score"] - row["old_score"]
    annotation = read(annotation_path)
    freeze(output / "human-annotations.json", annotation)
    key = read(PRIVATE / "researcher-key.json")
    public = json.loads(BeautifulSoup(HUMAN_PAGE.read_text(), "html.parser").select_one("#blind-data").string)
    if public["dataset_id"] != key["dataset_id"]:
        raise ValueError("Human page and private mapping differ")
    for r in annotation["responses"]:
        if r.get("rating_language") == "zh" and r.get("rating_translation_id") != public["translation_ids"]["zh"]:
            raise ValueError("Different annotation translation version")
    translations = {r["id"]: r for r in public["translations"]["zh"]}
    source = Path(prepared["source_directory"])
    questions = {qid: read(source / "questions" / f"{qid}.json") for qid in {r["question_id"] for r in rows}}
    result = {
        "version": VERSION,
        "status": "completed",
        "questions": prepared["questions"],
        "pairs": prepared["answer_pairs"],
        "execution": execution,
        "summary": summarize_rows(rows),
        "human": human_comparison(annotation, key, rows, questions, translations),
        "rows": rows,
        "provenance": {
            "prepared_manifest_sha256": sha(output / "manifest.json"),
            "human_file_sha256": sha(annotation_path),
        },
        "limitations": [
            "Old and new judge calls are different stochastic samples; no unchanged-prompt rerun control was collected, so changes cannot be attributed solely to the added instruction.",
            "The 20-question single-reviewer sample motivated the revision; it is exploratory, not held-out validation. Sixteen ratings were selected with Chinese displayed while model judges read English.",
            "No new judge diagnostic battery, debate, or chairman synthesis was run. Forced binary wins do not measure the size of a quality improvement.",
        ],
    }
    atomic_json(output / "comparison.json", result)
    return result


def percent(estimate):
    if estimate["mean"] is None:
        return "—"
    if estimate["ci"] is None:
        return f'{100*estimate["mean"]:.1f}%'
    low, high = estimate["ci"]
    return f'{100*estimate["mean"]:.1f}% [{100*low:.1f}, {100*high:.1f}]'


def change_pp(estimate):
    text = f'{100*estimate["mean"]:+.1f}'
    if estimate["ci"] is not None:
        low, high = estimate["ci"]
        text += f" [{100*low:+.1f}, {100*high:+.1f}]"
    return text + " pp"


def summary_html(result):
    rows = "".join(
        f'<tr><th>{MODELS[s["model"]]}</th><td>{percent(s["old_score"])}</td><td>{percent(s["new_score"])}</td>'
        f'<td>{change_pp(s["delta"])}</td><td>{s["changed_decisions"]}/{s["decisions"]}</td></tr>'
        for s in result["summary"]
    )
    human_rows = "".join(
        f'<tr><th>{MODELS[s["model"]]}</th><td>{s["old"]["matched_order_agreement"]}/{s["binary_ratings"]}</td>'
        f'<td>{s["new"]["matched_order_agreement"]}/{s["binary_ratings"]}</td></tr>'
        for s in result["human"]["summary"]
    )
    details = "".join(
        f'<p>{MODELS[s["model"]]}：原选 debate → 新选 baseline 有 {s["debate_to_baseline"]} 次，'
        f'反向有 {s["baseline_to_debate"]} 次。交换答案顺序后选择不一致的答案对：'
        f'旧 {s["old_order_inconsistent"]}/{s["pairs"]}，新 {s["new_order_inconsistent"]}/{s["pairs"]}。</p>'
        for s in result["summary"]
    )
    return f"""<section id="process-neutral-judge-v2"><h2>Judge 规则修订：旧评分 vs 新评分</h2>
<p>仅限 mixed-family 三人组：同样的 50 题、100 对既有答案。两位 judge 各评两个答案顺序，共 400 次新评审。
没有重新生成讨论或最终答案。两套 turn-level tone assignment 合并汇总，每题等权；不拆成两个实验组。</p>
<p>新增规则只要求：不能仅因提到 committee、成员或讨论过程而加分或扣分；真正的证据和逻辑缺口仍可评价。</p>
<div class="table-wrap"><table><thead><tr><th>Judge</th><th>旧 Debate wins</th><th>新 Debate wins</th><th>变化</th><th>改变选择</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Debate wins = 选择讨论后答案的评审比例，不是质量提升百分比。每位 judge 有 200 次判断，但只有 50 个独立题目。
方括号为按题 bootstrap 的 95% 区间；变化单位 pp 是百分点。“改变选择”按同一答案对、同一展示顺序比较新旧判断。
每题先平均两套 assignment 与两个顺序，不把它们当作四个独立样本。交换顺序后选择不同不等于“平局”。</p>
<details><summary>选择往哪个方向变化？答案顺序还会影响判断吗？</summary>{details}</details>
<h3>与你已完成的 20 题标注对照</h3><p>以下只比较与你相同的答案展示顺序；分母是你选择 A 或 B 的题目数。</p>
<table><thead><tr><th>Judge</th><th>旧规则：与你选同一答案</th><th>新规则：与你选同一答案</th></tr></thead><tbody>{human_rows}</tbody></table>
<p>这是看过人工分歧后做的探索性修订，不是独立验证。16 题是在中文界面下评分，judge 读英文。
新旧请求还包含采样波动；没有额外做“原 prompt 再跑一次”的控制，不能把变化全部归因于新增指令。</p></section>"""


def case_html(case):
    escape = html.escape
    labels = {"A": "A", "B": "B", "tie": "实质相当", "unjudgeable": "无法判断", None: "未评分"}
    contents = [f'<h4>{escape(case["question"])}</h4><p>你的选择：{labels[case["human_preference"]]}</p>']
    contents.append(
        '<div class="answers">'
        + "".join(
            f'<article><h4>回答 {side} · 中文辅助译文</h4><p>{escape(case["chinese"]["answers"][side])}</p>'
            f'<details><summary>英文原文</summary><p>{escape(case["answers"][side])}</p></details></article>'
            for side in ("A", "B")
        )
        + "</div>"
    )
    for model, orders in case["judges"].items():
        contents.append(f"<h4>{MODELS[model]}</h4>")
        for order in orders:
            name = "与你相同的展示顺序" if order["same_order_as_human"] else "交换后的顺序"
            contents.append(
                f'<details><summary>{name}：旧选 {order["old"]["choice"]} → 新选 {order["new"]["choice"]}</summary>'
                f'<p>旧理由：{escape(order["old"]["evidence"])}</p><p>新理由：{escape(order["new"]["evidence"])}</p></details>'
            )
    return f'<details id="human-case-{case["number"]}"><summary>第 {case["number"]} 题 · {escape(case["chinese"]["question"])}</summary>{"".join(contents)}</details>'


def publish(result, output=OUTPUT, page=PAGE, attach=False):
    page, output = Path(page), Path(output)
    css = """body{margin:0;background:#f3f5f1;color:#20382e;font:16px/1.65 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:32px 24px}h1{font-size:28px}h2{font-size:23px}table{border-collapse:collapse;width:100%;background:white;margin:18px 0}th,td{border-bottom:1px solid #d6dfd8;text-align:left;padding:12px}thead{background:#e7eee6}.table-wrap{overflow-x:auto}details{background:#fff;border:1px solid #d6dfd8;border-radius:8px;margin:14px 0;padding:14px}summary{cursor:pointer;font-weight:650}.answers{display:grid;grid-template-columns:1fr 1fr;gap:22px}article p{white-space:pre-wrap}blockquote{border-left:3px solid #78a387;padding-left:15px;color:#465f51}a{color:#246e4b}@media(max-width:700px){main{padding:18px 12px}.answers{grid-template-columns:1fr}th,td{padding:8px;font-size:14px}}"""
    details = "".join(case_html(c) for c in result["human"]["cases"])
    cost = result["execution"]["cost_accounting"]
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>E2 judge rubric comparison</title><style>{css}</style><main><h1>三人 cross-model · E2 judge 重评</h1>
{summary_html(result)}<p><a href="#human-case-18">查看第 18 题的新旧判断</a></p>
<details><summary>精确新增指令与费用</summary><blockquote>{html.escape(ADDENDUM)}</blockquote>
<p>已完成 {result["execution"]["completed_calls"]}/{result["execution"]["planned_calls"]} 个评审，含技术重试共 {result["execution"]["attempts"]} 次请求尝试；收到响应的费用估算 ${cost["received_response_estimate_usd"]:.4f}，未知计费预留 ${cost["unresolved_reservations_usd"]:.4f}。
预留不是确认产生的费用，估算也不是供应商账单。成功评审没有重复请求，失败记录保留。</p></details>
<h2>20 题逐项对照</h2><p>A/B 始终对应你标注页面的两份答案，不随 judge 的展示顺序改变。评审理由保留英文原文。</p>{details}</main></html>"""
    page.write_text(document)
    atomic_json(page.with_suffix(".json"), result)
    if attach:
        for suffix in (".html", ".json"):
            backup = output / f"setting-report-before-v2{suffix}"
            if not backup.exists():
                shutil.copyfile(SETTING_PAGE.with_suffix(suffix), backup)
        soup = BeautifulSoup(SETTING_PAGE.read_text(), "html.parser")
        for old in soup.select("#process-neutral-judge-v2"):
            old.decompose()
        for paragraph in soup.select("#independent-grok-note p"):
            if "no human results have been collected yet" in paragraph.get_text():
                paragraph.string = (
                    "The constructed diagnostics below belong to Gemini's original rubric only; Grok has not "
                    "been evaluated on that battery. One human reviewer has now annotated the separate "
                    "outcome-blind 20-question sample. See the process-reference-neutral v2 section below "
                    "for old/new agreement with those ratings. Humans may report a tie or inability to judge, "
                    "unlike the forced-choice model judges."
                )
        section = BeautifulSoup(summary_html(result), "html.parser").select_one("section")
        link = soup.new_tag("a", href=page.name)
        link.string = "打开新旧评分与 20 题逐项对照"
        section.append(link)
        soup.select_one("#e2").insert_after(section)
        SETTING_PAGE.write_text(str(soup))
        data = read(SETTING_PAGE.with_suffix(".json"))
        data["process_neutral_judge_v2"] = {
            "version": VERSION,
            "summary": result["summary"],
            "human_summary": result["human"]["summary"],
            "detail_report": page.name,
        }
        atomic_json(SETTING_PAGE.with_suffix(".json"), data)
        from scripts.report_experiment_dashboard import publish as publish_dashboard

        publish_dashboard()
    return page.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=OUTPUT)
    parser.add_argument("--annotations", type=Path, default=ANNOTATION)
    parser.add_argument("--page", type=Path, default=PAGE)
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args()
    result = analyze(args.run, args.annotations)
    print(publish(result, args.run, args.page, args.attach))
