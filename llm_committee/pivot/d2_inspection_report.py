"""Offline, token-level inspection of archived D2 argument/filler pairs. No model calls."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import median

from .models import OPEN_MODELS, digest
from .strong_study import ARMS, VERSION, StrongGraph
from .tinker_provider import make_renderer, message_content
from .turn_tone import STREAM

LABELS = tuple("ABCDEFG")
MEANINGS = (
    "Completely disagree",
    "Mostly disagree",
    "Somewhat disagree",
    "Neither agree nor disagree",
    "Somewhat agree",
    "Mostly agree",
    "Completely agree",
)
MODEL_NAMES = {"Qwen/Qwen3.8-27B": "Qwen3.8-27B", "thinkingmachines/Inkling": "Inkling"}


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pct(value):
    return "unavailable" if value is None else f"{100 * value:.8g}%"


def number(value):
    return "unavailable" if value is None else f"{value:.10g}"


def esc(value):
    return html.escape(str(value), quote=True)


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def inspect_request(entry, tokenizer, renderer):
    """Check the saved position against native decoding and original top-k rows.

    Deliberately does not call capture_readout(), locate_visible_start() or
    distribution(): the stored readout is evidence to check, not the reconstruction.
    """
    request, response, parsed = entry["request"], entry["response"], entry["parsed"]
    raw, saved = response["raw"], parsed["_readout"]
    tokens = raw["sampled_token_ids"]
    decoded = tokenizer.decode(tokens)
    pieces = [tokenizer.decode([token]) for token in tokens]
    visible = response["text"]
    positions = [i for i, piece in enumerate(pieces) if piece == visible]
    independent_index = positions[0] if len(positions) == 1 else None
    index = saved["sampled_output_index"]
    in_bounds = 0 <= index < len(tokens)
    expected_ids = {letter: tokenizer.encode(letter, add_special_tokens=False) for letter in LABELS}
    checks = {
        "one_visible_rating_letter": visible in LABELS and parsed["rating"] == visible,
        "rating_token_position": independent_index is not None and independent_index == index,
        "candidate_token_ids": all(
            len(expected_ids[k]) == 1
            and saved["candidate_ids"][k] == expected_ids[k][0]
            and tokenizer.decode(expected_ids[k]) == k
            for k in LABELS
        ),
        "raw_arrays_aligned": len(tokens) == len(raw["sampled_logprobs"]) == len(raw["topk_logprobs"]),
        "sampled_token_id": in_bounds and tokens[index] == saved["sampled_token_id"],
    }
    parsed_message, termination = renderer.parse_response(tokens)
    text, reasoning = message_content(parsed_message)
    checks["native_response_parse"] = text == visible and termination.is_clean and raw["stop_reason"] == "stop"
    checks["reasoning_disabled"] = (
        request["effort"] == "none" and not reasoning.strip() and response["reasoning_tokens"] == 0
    )
    messages = [
        {"role": "system" if m["role"] == "developer" else m["role"], "content": m["text"]} for m in request["messages"]
    ]
    kwargs = {"effort": 0.0} if request["model"] == "thinkingmachines/Inkling" else {}
    reproduced_prompt = renderer.build_generation_prompt(messages, **kwargs).to_ints()
    checks["native_prompt_token_ids"] = reproduced_prompt == raw["prompt_token_ids"]
    prefix = raw["prompt_token_ids"] + tokens[:index] if in_bounds else []
    checks["saved_prefix"] = prefix == saved["prefix_token_ids"] and digest(prefix) == saved["prefix_sha256"]
    checks["readout_copies"] = saved == response["readout"] == raw["probability_readout"]
    checks["unscaled_sampling"] = (
        all(
            saved[k] == raw[k] == expected
            for k, expected in (("sampling_temperature", 1.0), ("top_p", 1.0), ("top_k", -1))
        )
        and saved["read_temperature"] == 1.0
    )
    top = dict(raw["topk_logprobs"][index] or []) if in_bounds else {}
    scores = {k: top[ids[0]] for k, ids in expected_ids.items() if len(ids) == 1 and ids[0] in top}
    checks["candidate_scores_from_saved_row"] = scores == saved["candidate_logprobs"]
    logprob = raw["sampled_logprobs"][index] if in_bounds else None
    checks["sampled_vs_topk_logprob"] = (
        in_bounds
        and tokens[index] in top
        and logprob is not None
        and abs(top[tokens[index]] - logprob) < 1e-7
        and saved["sampled_token_logprob"] == logprob
    )
    masses = {k: math.exp(v) for k, v in scores.items()}
    mass = sum(masses.values())
    probabilities = {k: masses.get(k, 0.0) / mass for k in LABELS} if mass else {}
    rating = sum((i + 1) * probabilities[k] for i, k in enumerate(LABELS)) if mass else None
    token_rows = []
    for i, token in enumerate(tokens):
        lp = raw["sampled_logprobs"][i]
        alternatives = [
            {
                "token_id": token_id,
                "text": tokenizer.decode([token_id]),
                "logprob": value,
                "probability": math.exp(value),
                "is_sampled": token_id == token,
                "is_rating_letter": tokenizer.decode([token_id]) in LABELS,
            }
            for token_id, value in (raw["topk_logprobs"][i] or [])
        ]
        token_rows.append(
            {
                "index": i,
                "token_id": token,
                "text": pieces[i],
                "logprob": lp,
                "probability": math.exp(lp) if lp is not None else None,
                "used_for_D2": i == index,
                "independently_located_rating": i == independent_index,
                "top20": alternatives,
            }
        )
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "read_index": index,
        "independent_rating_index": independent_index,
        "visible_output": visible,
        "decoded_output": decoded,
        "reasoning_text": reasoning,
        "tokens": token_rows,
        "candidate_logprobs": scores,
        "candidate_raw_probabilities": masses,
        "candidate_mass": mass,
        "missing_letters": [k for k in LABELS if k not in scores],
        "normalized_probabilities": probabilities,
        "recomputed_rating": rating,
        "decoded_prompt": tokenizer.decode(raw["prompt_token_ids"]),
        "decoded_scoring_prefix": tokenizer.decode(prefix),
        "scoring_prefix_suffix": tokenizer.decode(tokens[:index]) if in_bounds else "unavailable",
        "raw": entry,
    }


def select_cases(cases):
    """Purposeful, outcome-aware diagnostic coverage; not a population sample."""
    selected = []
    for model in MODEL_NAMES:
        pool = [c for c in cases if c["model"] == model]
        if len(pool) < 5:
            raise ValueError(f"Need at least five archived D2 pairs for {model}")
        midpoint = median(c["reported_delta"] for c in pool)
        rules = (
            (
                "Largest decrease with an agreeing peer",
                lambda c: c["peer_label"] in ("strongly_agree", "leaning_agree"),
                lambda c: c["reported_delta"],
            ),
            (
                "Largest decrease with a disagreeing peer",
                lambda c: c["peer_label"] in ("strongly_disagree", "leaning_disagree"),
                lambda c: c["reported_delta"],
            ),
            (
                "Closest to zero with mutual strong agreement",
                lambda c: c["peer_label"] == c["receiver_label"] == "strongly_agree",
                lambda c: abs(c["reported_delta"]),
            ),
            (
                "Closest to the model's median case-level difference",
                lambda c: True,
                lambda c, midpoint=midpoint: abs(c["reported_delta"] - midpoint),
            ),
            ("Highest signed difference (may still be near zero)", lambda c: True, lambda c: -c["reported_delta"]),
        )
        used = set()
        for rank, (reason, predicate, score) in enumerate(rules, 1):
            available = [c for c in pool if c["id"] not in used]
            eligible = [c for c in available if predicate(c)]
            fallback = not eligible
            chosen = min(eligible or available, key=lambda c: (score(c), c["id"]))
            used.add(chosen["id"])
            selected.append({**chosen, "selection_reason": reason, "selection_fallback": fallback, "rank": rank})
    return selected


def load_inspection(source, *, renderer_factory=make_renderer):
    source = Path(source).resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["protocol_version"] != VERSION or manifest["kind"] != "paid_strong_dyadic_pilot":
        raise ValueError("This inspection requires the archived, paid strong-label run")
    contexts = json.loads((source / "source-contexts.json").read_text())
    source_files = [
        source / name for name in ("manifest.json", "source-contexts.json", "report.json", "requests.sqlite3")
    ]
    source_files.extend(source / "questions" / f"{plan['question_id']}.json" for plan in manifest["plans"])
    source_hashes = {str(p.relative_to(source)): file_hash(p) for p in source_files}
    entries = {}
    with sqlite3.connect(f"file:{source}/requests.sqlite3?mode=ro", uri=True) as db:
        for rq, rs, parsed in db.execute(
            "SELECT request,response,parsed FROM calls WHERE status='completed' AND json_extract(request,'$.purpose')='d_text'"
        ):
            request = json.loads(rq)
            entries[request["key"]] = {"request": request, "response": json.loads(rs), "parsed": json.loads(parsed)}
    renderers = {model: renderer_factory(model, "none") for model in sorted(OPEN_MODELS)}
    inspected = {key: inspect_request(entry, *renderers[entry["request"]["model"]]) for key, entry in entries.items()}
    cases, seen = [], set()
    for plan in manifest["plans"]:
        path = source / "questions" / f"{plan['question_id']}.json"
        record = json.loads(path.read_text())
        graph = StrongGraph(contexts[plan["question_id"]], plan, manifest)
        for arm in ARMS:
            for event in record["branches"][arm]["events"]:
                d = event["D_text"]
                if not d or event["member"] not in (1, 2) or (arm == "alternate" and event["T"] <= 3):
                    continue
                keys = {t: graph.key(arm, f"Dtext/{event['id']}/{t}") for t in ("argument", "control")}
                identity = (keys["argument"], keys["control"])
                if identity in seen:
                    raise ValueError("Duplicated D2 request pair")
                seen.add(identity)
                pair = {t: inspected[key] for t, key in keys.items()}
                messages = {t: pair[t]["raw"]["request"]["messages"] for t in pair}
                a, b = messages["argument"], messages["control"]
                differences = [i for i in range(min(len(a), len(b))) if a[i] != b[i]]
                payloads = {t: json.loads(v[-2]["text"]) for t, v in messages.items()}
                checks = {
                    "only_incoming_message_differs": len(a) == len(b) and differences == [len(a) - 2],
                    "same_sender": payloads["argument"]["member"] == payloads["control"]["member"],
                    "fixed_full_position": all(
                        json.loads(v[-3]["text"])["your_previously_recorded_full_position"] == d["fixed_full_position"]
                        for v in messages.values()
                    ),
                }
                for treatment in pair:
                    actual, reported = pair[treatment], d[treatment]
                    checks[f"{treatment}_probabilities_reproduced"] = all(
                        k in actual["normalized_probabilities"]
                        and abs(actual["normalized_probabilities"][k] - reported["probabilities"][k]) < 1e-10
                        for k in LABELS
                    )
                recomputed = (
                    pair["argument"]["recomputed_rating"] - pair["control"]["recomputed_rating"]
                    if all(v["recomputed_rating"] is not None for v in pair.values())
                    else None
                )
                checks["reported_difference_reproduced"] = (
                    recomputed is not None and abs(recomputed - d["mean_own_agreement_argument_minus_control"]) < 1e-10
                )
                previous_own = graph.route.previous_own(event["node_id"])
                public_own_text = (
                    record["branches"][arm]["formal_replies"][STREAM][previous_own.id]["reply"]
                    if previous_own
                    else record["initial_answers"][str(event["member"])]
                )
                cases.append(
                    {
                        "id": f"{record['question_id']}:{event['node_id']}:{arm}",
                        "question": record["question"],
                        "node_id": event["node_id"],
                        "T": event["T"],
                        "assignment": "Shared T1/T2" if event["T"] == 2 else arm.title(),
                        "model": entries[keys["argument"]]["request"]["model"],
                        "peer_label": event["previous_peer_self_label"],
                        "receiver_label": event["current_self_label"],
                        "peer_tone": event["previous_tone"],
                        "receiver_tone": event["current_tone"],
                        "fixed_position": d["fixed_full_position"],
                        "position_reading_id": event["previous_reading"],
                        "prior_public_contribution": public_own_text,
                        "position_verbatim_matches_public_contribution": public_own_text == d["fixed_full_position"],
                        "incoming_argument": payloads["argument"]["contribution"],
                        "filler": payloads["control"]["contribution"],
                        "argument_tokens": d["argument_tokens"],
                        "filler_tokens": d["filler_tokens"],
                        "filler_repetitions": d["filler_repetitions"],
                        "reported_delta": d["mean_own_agreement_argument_minus_control"],
                        "recomputed_delta": recomputed,
                        "pair_checks": checks,
                        "arms": pair,
                    }
                )
    audit = []
    for model in MODEL_NAMES:
        requests = [v for v in inspected.values() if v["raw"]["request"]["model"] == model]
        pairs = [c for c in cases if c["model"] == model]
        audit.append(
            {
                "model": model,
                "requests": len(requests),
                "pairs": len(pairs),
                "request_failures": sum(not v["passed"] for v in requests),
                "pair_failures": sum(not all(c["pair_checks"].values()) for c in pairs),
                "indices": dict(Counter(v["read_index"] for v in requests)),
                "candidate_mass_min": min(v["candidate_mass"] for v in requests),
                "candidate_mass_max": max(v["candidate_mass"] for v in requests),
                "requests_with_missing_letters": sum(bool(v["missing_letters"]) for v in requests),
                "missing_letter_count": sum(len(v["missing_letters"]) for v in requests),
                "failed_requests": [
                    {"key": k, "checks": v["checks"]}
                    for k, v in inspected.items()
                    if v["raw"]["request"]["model"] == model and not v["passed"]
                ],
            }
        )
    return {
        "schema": "d2_token_inspection_v1",
        "source": str(source),
        "protocol": VERSION,
        "no_new_model_calls": True,
        "audit": audit,
        "cases": select_cases(cases),
        "request_checks": {k: v["checks"] for k, v in inspected.items()},
        "pair_checks": {c["id"]: c["pair_checks"] for c in cases},
        "source_hashes": source_hashes,
    }


def table(headers, rows):
    return (
        '<div class="scroll"><table><thead><tr>'
        + "".join(f"<th>{esc(v)}</th>" for v in headers)
        + "</tr></thead><tbody>"
        + "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
        + "</tbody></table></div>"
    )


def arm_html(arm, name):
    request, response = arm["raw"]["request"], arm["raw"]["response"]
    raw = response["raw"]
    stream = " ".join(
        f'<span class="token {"read" if r["used_for_D2"] else ""}" title="output index {r["index"]}; token ID {r["token_id"]}">{esc(r["text"])}<small>#{r["index"]} · id {r["token_id"]}</small></span>'
        for r in arm["tokens"]
    )
    tokens = [
        [
            r["index"],
            r["token_id"],
            repr(r["text"]),
            number(r["logprob"]),
            pct(r["probability"]),
            "READ HERE" if r["used_for_D2"] else "",
        ]
        for r in arm["tokens"]
    ]
    top20 = "".join(
        f"<details {'open' if r['used_for_D2'] else ''}><summary>Top-20 at output index {r['index']}{' — D2 reads this row' if r['used_for_D2'] else ' — not used'}</summary>"
        + table(
            ["Token ID", "Token text", "ln P(token)", "Raw P(token)", "Role"],
            [
                [
                    v["token_id"],
                    repr(v["text"]),
                    number(v["logprob"]),
                    pct(v["probability"]),
                    "sampled" if v["is_sampled"] else "A–G candidate" if v["is_rating_letter"] else "other token",
                ]
                for v in r["top20"]
            ],
        )
        + "</details>"
        for r in arm["tokens"]
    )
    messages = "".join(
        f'<article class="message"><h5>Message {i} · {esc(m["role"])}{" → native system" if m["role"] == "developer" else ""}</h5><pre>{esc(m["text"])}</pre></article>'
        for i, m in enumerate(request["messages"])
    )
    return f"""<article class="arm"><h3>{esc(name)}</h3>
<p>Visible output: <strong class="output">{esc(arm["visible_output"])}</strong> · expected rating: <strong>{number(arm["recomputed_rating"])}</strong></p>
<p>Saved read index: <strong>{arm["read_index"]}</strong>; independently decoded rating index: {arm["independent_rating_index"]}.
Reasoning text: {"empty" if not arm["reasoning_text"] else esc(arm["reasoning_text"])}.</p>
<div class="tokens">{stream}</div>
<p class="note">Indices are zero-based within the sampled output, not the input prompt. Green marks the row used for D2.</p>
{table(["Index", "Token ID", "Decoded token", "ln P(token)", "Raw P(token)", "Used"], tokens)}
<p>Recorded A–G raw probability mass: <strong>{pct(arm["candidate_mass"])}</strong>.
Missing letters: {esc(", ".join(arm["missing_letters"]) or "none")}. Missing is approximated as zero only after extraction.</p>
<details class="full-prompt"><summary>Full request — all {len(request["messages"])} messages, no truncation</summary>{messages}</details>
<details><summary>Exact native prompt decoded from saved token IDs</summary><pre>{esc(arm["decoded_prompt"])}</pre></details>
<details><summary>Exact prefix immediately before the measured rating token</summary>
<p>This is the saved prompt plus generated framing tokens before index {arm["read_index"]}. The generated suffix is:</p>
<pre>{esc(arm["scoring_prefix_suffix"] or "(empty)")}</pre><pre>{esc(arm["decoded_scoring_prefix"])}</pre></details>
<details><summary>Raw decoded output including format/end tokens</summary><pre>{esc(arm["decoded_output"])}</pre></details>
<details><summary>Original top-20 logprobs at every generated position</summary>{top20}</details>
<details><summary>Request, response, token IDs and saved readout — exact archived JSON</summary><pre>{esc(pretty(arm["raw"]))}</pre></details>
<details><summary>Alignment checks: {"all passed" if arm["passed"] else "FAILURE — inspect"}</summary>
{table(["Check", "Result"], [[k, "PASS" if v else "FAIL"] for k, v in arm["checks"].items()])}
<p>Renderer: {esc(raw["renderer"])}; effort: {esc(request["effort"])}; sampling/read temperatures: 1/1.</p></details></article>"""


def case_html(case, anchor):
    control, argument = case["arms"]["control"], case["arms"]["argument"]
    matrix = []
    for i, k in enumerate(LABELS, 1):
        a = argument["normalized_probabilities"].get(k)
        c = control["normalized_probabilities"].get(k)
        matrix.append(
            [
                f"{k} = {i}",
                MEANINGS[i - 1],
                pct(c),
                pct(a),
                number(i * (a - c)) if a is not None and c is not None else "unavailable",
            ]
        )
    return f'''<details class="case" id="{anchor}"><summary>{esc(MODEL_NAMES[case["model"]])} #{case["rank"]} · {esc(case["selection_reason"])} · Δ {number(case["reported_delta"])}</summary>
<p><strong>Question:</strong> {esc(case["question"]["text"])}</p>
{table(["Original question option", "Text"], [[chr(65 + i), v] for i, v in enumerate(case["question"]["options"])])}
<p>{esc(case["question"]["id"])} · {esc(case["node_id"])} · T={case["T"]} · {esc(case["assignment"])}.
Peer self-label: <strong>{esc(case["peer_label"])}</strong>; receiver's later self-label: <strong>{esc(case["receiver_label"])}</strong>.
Peer/current reply tones: {esc(case["peer_tone"])}/{esc(case["receiver_tone"])}; both D2 probes themselves are neutral.</p>
<p class="note">The receiver label comes from its subsequent debate reply. That reply is not in these pre-reply D2 prompts.
Selection is deliberate diagnostic coverage, not a random or representative sample. {"Fallback to the remaining cases was needed." if case["selection_fallback"] else ""}</p>
<h3>What is being endorsed? The member's exact prior position</h3><pre class="position">{esc(case["fixed_position"])}</pre>
<p class="note">This paragraph comes from the separate C position reading {esc(case["position_reading_id"])}.
It {"is" if case["position_verbatim_matches_public_contribution"] else "is not"} verbatim identical to the preceding public contribution the peer answered.
That distinction may matter when the peer criticizes particular reasons or factual claims.</p>
<details><summary>Compare: the member's preceding public contribution, before the separate position reading</summary><pre>{esc(case["prior_public_contribution"])}</pre></details>
<div class="pair"><article><h3>Filler replacing the peer message</h3><p>{case["filler_tokens"]} tokens; {case["filler_repetitions"]} repetitions.</p><pre>{esc(case["filler"])}</pre></article>
<article><h3>Actual incoming peer argument</h3><p>{case["argument_tokens"]} tokens.</p><pre>{esc(case["incoming_argument"])}</pre></article></div>
<h3>Seven rating probabilities at the saved read position</h3>
<p>These A–G letters are endorsement ratings, not the question's answer options above. Probabilities here are normalized over A–G;
raw vocabulary probabilities are shown in the token tables below. “Contribution” is k × [p(peer,k) − p(filler,k)]; its sum is the rating difference.</p>
{table(["Rating", "Meaning", "Filler probability", "Peer probability", "Contribution to Δ"], matrix)}
<p class="score">Filler {number(control["recomputed_rating"])}; peer {number(argument["recomputed_rating"])};
recomputed difference <strong>{number(case["recomputed_delta"])}</strong>; archived difference {number(case["reported_delta"])}.</p>
<details><summary>Pair verification — same preceding messages, fixed own text and rating request</summary>
{table(["Check", "Result"], [[k, "PASS" if v else "FAIL"] for k, v in case["pair_checks"].items()])}</details>
<div class="pair">{arm_html(control, "Control · filler")}{arm_html(argument, "Treatment · peer argument")}</div></details>'''


CSS = """
:root { color-scheme:light; --ink:#24323b; --line:#dce4e8; --accent:#14775f; }
* { box-sizing:border-box; } body { margin:0; background:#f2f5f6; color:var(--ink); font:16px/1.6 system-ui,sans-serif; }
main { max-width:1560px; margin:24px auto; padding:36px; background:white; }
h1 { margin-top:0; font-size:30px; } h2 { margin-top:36px; } h3 { font-size:18px; } h5 { margin:0 0 8px; font-size:14px; }
a { color:var(--accent); } .note { color:#52636e; font-size:14px; } .status,.score { background:#edf6f1; padding:14px; border-left:4px solid var(--accent); }
.warning { background:#fff4df; border-left-color:#aa7300; } .pair { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; align-items:start; }
.arm { border:1px solid var(--line); padding:18px; min-width:0; } pre { white-space:pre-wrap; overflow-wrap:anywhere; font:13px/1.6 ui-monospace,monospace; background:#f6f8fa; padding:14px; }
.position { background:#fff9eb; } .scroll { overflow-x:auto; } table { border-collapse:collapse; width:100%; font-size:13px; margin:14px 0; }
th,td { text-align:left; vertical-align:top; padding:9px; border-bottom:1px solid var(--line); overflow-wrap:anywhere; }
.arm th, .arm td:nth-child(4), .arm td:nth-child(5) { white-space:nowrap; }
thead { border-top:2px solid var(--ink); border-bottom:1px solid var(--ink); } tbody tr:nth-child(even) { background:#f6f8fa; }
details { border:1px solid var(--line); padding:12px; margin:14px 0; } summary { cursor:pointer; font-weight:650; overflow-wrap:anywhere; }
.case { padding:18px; } .case>summary { font-size:19px; } .token { display:inline-block; border:1px solid #ccd6dc; background:#f5f7f9; padding:7px; margin:4px 0; font:13px ui-monospace,monospace; }
.token small { display:block; margin-top:6px; color:#586771; } .token.read { background:#d9f2e5; border:2px solid #14775f; } .output { font:700 24px ui-monospace,monospace; }
.message { border-left:3px solid #bacbd3; padding-left:12px; margin:18px 0; } button { cursor:pointer; padding:8px 14px; margin:4px; background:white; border:1px solid #a6b8c2; border-radius:5px; }
@media(max-width:1100px) { .pair { grid-template-columns:1fr; } main { padding:18px; margin:0; } }
"""


def render(data):
    audit_rows = [
        [
            MODEL_NAMES[a["model"]],
            a["pairs"],
            a["requests"],
            str(a["indices"]),
            a["request_failures"],
            a["pair_failures"],
            f"{pct(a['candidate_mass_min'])}–{pct(a['candidate_mass_max'])}",
            a["requests_with_missing_letters"],
        ]
        for a in data["audit"]
    ]
    failures = sum(a["request_failures"] + a["pair_failures"] for a in data["audit"])
    overview, bodies = [], []
    for i, case in enumerate(data["cases"], 1):
        anchor = f"case-{i}"
        overview.append(
            f'<tr><td><a href="#{anchor}">{esc(MODEL_NAMES[case["model"]])} #{case["rank"]}</a></td><td>{esc(case["selection_reason"])}</td><td>{esc(case["question"]["id"])} · {esc(case["node_id"])}</td><td>{esc(case["assignment"])}</td><td>{number(case["reported_delta"])}</td></tr>'
        )
        bodies.append(case_html(case, anchor))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>D2 · full-prompt and token-position inspection</title><style>{CSS}</style></head><body><main>
<h1>D2 · Full prompts, raw outputs and the exact probability-reading position</h1>
<p>Five archived argument/filler pairs per model; ten cases and twenty original model outputs. No new model calls, resampling or changes to the main results.</p>
<p class="status {"warning" if failures else ""}">{"Alignment/pair failures detected: " + str(failures) if failures else "No token-position mismatch found in the archived D2 checks."}
The audit covers every archived D2 pair, not just the ten displayed cases.</p>
<h2>What was checked?</h2>
<p>Decode every sampled token, independently locate the visible A–G rating, reproduce the native prompt token IDs and response parsing,
compare the sampled token's logprob with its entry in the original top-20 row, and recompute all seven normalized probabilities and the saved rating difference.
For every pair, check that only the incoming peer/filler message changes. This is an internal consistency check of archived responses,
not independent verification of the provider's backend probabilities or validation of the endorsement measure.</p>
{table(["Model", "Unique pairs", "Outputs audited", "Read indices (0-based): counts", "Request failures", "Pair failures", "Recorded A–G raw mass range", "Outputs missing ≥1 rating letter"], audit_rows)}
<p>Inkling emits two formatting tokens before its rating; Qwen starts directly with the rating letter. These formatting tokens are visible below.
Raw logprobs are natural logs; raw probability = exp(logprob). Displayed normalized rating probabilities divide each returned A–G probability by their total mass.
Absent top-20 letters are zero-filled, not proven impossible; masses a tiny fraction above 100% reflect floating-point logprob precision.</p>
<h2>How the ten cases were selected</h2>
<p>Per model: the largest decrease with an agreeing peer, the largest decrease with a disagreeing peer, a near-zero mutual-strong-agreement case,
a case nearest the model's median individual difference, and the highest signed difference. Cases are distinct, but questions need not be:
two branches of the same question can provide useful contrasts. This is deliberate, outcome-aware troubleshooting, not prevalence estimation.</p>
<p>Scores here are individual comparisons, not the question-weighted group averages in the main report. Open a case, then
“Full request” for both complete message lists; native serialized prompts and original top-20 rows are also available without truncation.</p>
<p class="status warning">A measurement issue to inspect separately from token alignment: the peer responds to a public debate contribution,
whereas D2 rates a paragraph generated by a separate C position read. Those texts need not be verbatim identical.
Each case includes both. Also, agreeing with a conclusion is compatible with criticizing its reasons; a G → F endorsement shift
does not necessarily mean the survey-option position changed. The case texts, not just the label pair, are needed to assess that interpretation.</p>
<button onclick="document.querySelectorAll('.case').forEach(e=>e.open=true)">Open all ten cases</button>
<button onclick="document.querySelectorAll('.case,.full-prompt').forEach(e=>e.open=true)">Expand all full prompts</button>
<button onclick="document.querySelectorAll('.case').forEach(e=>e.open=false)">Collapse cases</button>
<div class="scroll"><table><thead><tr><th>Open case</th><th>Selection reason</th><th>Question / node</th><th>Branch</th><th>Peer − filler rating</th></tr></thead><tbody>{"".join(overview)}</tbody></table></div>
{"".join(bodies)}
<details><summary>Reproducibility: source hashes and audit scope</summary><pre>{esc(pretty({k: v for k, v in data.items() if k != "cases"}))}</pre></details>
</main><script>function openTarget(){{const e=document.getElementById(location.hash.slice(1));if(e&&e.matches('.case')){{e.open=true;requestAnimationFrame(()=>e.scrollIntoView());}}}}addEventListener('hashchange',openTarget);openTarget();</script></body></html>"""


def publish(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.suffix != ".html" or source == output.parent or source in output.parents:
        raise ValueError("Write a separate HTML outside the immutable run")
    if output.exists():
        sidecar = output.with_suffix(".json")
        old = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if old.get("schema") != "d2_token_inspection_v1" or old.get("source") != str(source):
            raise ValueError("Refuse to overwrite an unrelated report")
    data = load_inspection(source)
    document = render(data)
    for relative, expected in data["source_hashes"].items():
        if file_hash(source / relative) != expected:
            raise ValueError("Source changed during inspection")
    output.write_text(document, encoding="utf-8")
    output.with_suffix(".json").write_text(pretty(data) + "\n", encoding="utf-8")
    return {
        "html": str(output),
        "json": str(output.with_suffix(".json")),
        "cases": len(data["cases"]),
        "audit": data["audit"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(pretty(publish(args.source, args.output)))
