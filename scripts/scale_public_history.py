"""Freeze an outcome-blind, question-disjoint extension of all six settings.

Preparation is offline. Live execution is explicitly requested with --live.
Each extension has its own journal; existing twenty-question batches are never
resumed, rebalanced, or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_committee.pivot.models import canonical, digest
from llm_committee.pivot.planning import rng_for
from llm_committee.pivot.quality_study import plan_questions as quality_plans
from llm_committee.pivot.stateful_study import prepare as prepare_dyadic
from llm_committee.pivot.study import atomic_json
from llm_committee.pivot.triadic_study import make_design
from llm_committee.pivot.triadic_study import prepare as prepare_triadic
from llm_committee.pivot.turn_tone import plan_questions
from llm_committee.pivot.turn_tone_fork import alternate_plans

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runs/public-history-extension30-20260928"
BASE_DYADIC = ROOT / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"
BASE_TRIADIC = ROOT / "docs/turn-tone-triadic-shared-plan-2026-09-28.json"
BANK = ROOT / "docs/phase-1-question-bank.json"
ROSTERS = ("mixed_family", "same_family", "same_model")
SEED = 20260928


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen file differs: {path}")
    else:
        atomic_json(path, value)


def execution_specs(output):
    """Resolve documented recovery journals without modifying the frozen plan."""
    output = Path(output)
    specs = copy.deepcopy(read(output / "execution-plan.json")["settings"])
    override = output / "recovery-overrides.json"
    if not override.exists():
        return specs
    seen = set()
    for replacement in read(override)["replacements"]:
        key = replacement["roster"], replacement["setting"]
        matches = [s for s in specs if (s["roster"], s["setting"]) == key]
        if key in seen or len(matches) != 1:
            raise ValueError("Invalid or duplicate recovery replacement")
        seen.add(key)
        spec = matches[0]
        target = Path(replacement["replacement_output_30"])
        manifest = read(target / "manifest.json")
        if (
            replacement["original_output_30"] != spec["output_30"]
            or manifest["recovery"]["source_directory"] != spec["output_30"]
            or manifest["config"]["roster"] != spec["roster"]
            or sha(target / "manifest.json") != replacement["replacement_manifest_sha256"]
        ):
            raise ValueError("Recovery replacement provenance mismatch")
        spec["output_30"] = str(target)
    return specs


def previous_runs():
    from scripts.report_experiment_dashboard import PILOT_REPORTS

    result = {}
    for (roster, members), (name, _) in PILOT_REPORTS.items():
        data = read((ROOT / "docs" / name).with_suffix(".json"))
        result[roster, "dyadic" if members == 2 else "triadic"] = Path(data["source_directory"])
        if members == 2:
            result[roster, "quality"] = Path(data["length_controlled_E"]["source_directory"])
    return result


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    template, triadic = read(BASE_DYADIC), read(BASE_TRIADIC)
    archive_path = Path(triadic["sources"]["route_manifest"])
    archive = read(archive_path)
    questions = {q["id"]: q for q in archive["questions"]}
    bank = read(BANK)
    assert bank["status"] == "user_approved_question_set" and len(bank["questions"]) == 60
    assert {(q["text"], tuple(q["options"])) for q in questions.values()} == {
        (q["question"], tuple(q["options"])) for q in bank["questions"]
    }
    old_ids = sorted(p["question"]["id"] for p in template["plans"])
    assert len(old_ids) == 20 and set(old_ids) == {p["question_id"] for p in triadic["plans"]}
    remaining = sorted(questions.keys() - set(old_ids))
    assert len(remaining) == 40
    selected = sorted(rng_for(SEED, "public-history-extension-30", "question-selection").sample(remaining, 30))
    selection = {
        "target_questions_per_setting": 50,
        "existing_question_ids": old_ids,
        "new_question_ids": selected,
        "unused_question_ids": sorted(set(remaining) - set(selected)),
        "seed": SEED,
        "method": "Uniform 30 of the remaining approved 40; IDs and fixed seed only, no outcome-based selection or redraw",
        "bank_file": str(BANK),
        "bank_sha256": sha(BANK),
        "old_dyadic_design_sha256": template["design_sha256"],
        "old_triadic_design_sha256": triadic["design_sha256"],
        "same_extension_for_all_six_settings": True,
        "old_generations_and_failures_unchanged": True,
    }
    freeze(output / "selection.json", selection)
    original = plan_questions(selected)
    alternate = {p["question_id"]: p for p in alternate_plans(original)}
    extension = copy.deepcopy(template)
    extension.pop("design_sha256")
    extension["design_id"] = "dyadic-shared-prefix-tone-extension30-2026-09-28-v1"
    extension["execution_authorization"] = (
        "User approved thirty additional questions for each of six settings, total fifty per setting"
    )
    extension["sources"] = {
        "base_plan": str(BASE_DYADIC),
        "base_plan_sha256": sha(BASE_DYADIC),
        "selection": str(output / "selection.json"),
        "selection_sha256": sha(output / "selection.json"),
    }
    extension["scope"].update(question_count=30, paired_paths_per_roster=90, unique_debate_replies_per_roster=540)
    extension["plans"] = [
        {
            "question": questions[p["question_id"]],
            "original_tones": p["tone_schedule"],
            "alternate_tones": alternate[p["question_id"]]["tone_schedule"],
            "sampled_node_ids": [e["node_id"] for e in p["events"]],
        }
        for p in original
    ]
    shared = sum(e["T"] == 2 for p in original for e in p["events"])
    extension["sampling_contract"].update(
        shared_T2_event_count=shared,
        paired_T3_T4_event_locations=240 - shared,
        original_events=240,
        alternate_new_events=240 - shared,
        independent_questions=30,
    )
    for mapping in extension["existing_roster_slot_mappings"].values():
        mapping["status"] = "extension_authorized"
    freeze(output / "plans/dyadic-plan.json", {**extension, "design_sha256": digest(extension)})
    freeze(output / "plans/triadic-plan.json", make_design(output / "plans/dyadic-plan.json", archive_path))
    freeze(output / "quality-plan.json", quality_plans(selected))

    old_runs = previous_runs()
    preservation = {}
    for source in old_runs.values():
        for path in [
            source / "manifest.json",
            source / "source-contexts.json",
            source / "report.json",
            source / "requests.sqlite3",
            *sorted((source / "questions").glob("*.json")),
        ]:
            preservation[str(path)] = sha(path)
    for path in [
        BASE_DYADIC,
        BASE_TRIADIC,
        BANK,
        *ROOT.glob("docs/turn-tone-*public-history-2026-09-2*.html"),
        *ROOT.glob("docs/turn-tone-*public-history-2026-09-2*.json"),
    ]:
        preservation[str(path)] = sha(path)
    freeze(output / "preserved-source-hashes.json", preservation)
    specs = []
    for roster in ROSTERS:
        for setting, fn in (("dyadic", prepare_dyadic), ("triadic", prepare_triadic)):
            kwargs = {"roster": roster, **({"public_history": True} if setting == "dyadic" else {})}
            manifest, contexts = fn(output / "plans" / f"{setting}-plan.json", **kwargs)
            old = read(old_runs[roster, setting] / "manifest.json")
            assert manifest["config"] == old["config"]
            assert manifest["protocol_version"] == old["protocol_version"]
            assert manifest["prompt_version"] == old["prompt_version"]
            assert set(contexts) == set(selected)
            counts = manifest["planned_counts"]
            specs.append(
                {
                    "roster": roster,
                    "setting": setting,
                    "planned_counts": counts,
                    "source_20": str(old_runs[roster, setting]),
                    "output_30": str(output / roster / setting),
                    "baseline_cost_usd": read(old_runs[roster, setting] / "report.json")["charged_or_reserved_usd"],
                }
            )
        specs.append(
            {
                "roster": roster,
                "setting": "quality",
                "planned_counts": {"questions": 30, "primary_answer_pairs": 180, "logical_calls": 30 * 21 + 12 * 10},
                "source_20": str(old_runs[roster, "quality"]),
                "output_30": str(output / roster / "quality"),
                "baseline_cost_usd": read(old_runs[roster, "quality"] / "report.json")["charged_or_reserved_usd"],
            }
        )
    freeze(
        output / "execution-plan.json",
        {
            "settings": specs,
            "rough_linear_estimate_usd": sum(s["baseline_cost_usd"] for s in specs) * 1.5,
            "estimate_note": "Linear 30/20 extrapolation including previous retries; diagnostic counts stay twelve per added batch, not an invoice or cap",
            "max_total_inflight": 64,
            "concurrent_jobs": 2,
            "max_inflight_per_job": 32,
        },
    )
    return selection, specs


def launch_one(output, roster, setting, *, gate=False):
    destination = output / roster / setting
    log = output / roster / f"{setting}{'-gate' if gate else ''}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    module = {"dyadic": "stateful_run", "triadic": "triadic_run", "quality": "quality_run"}[setting]
    cmd = [sys.executable, "-m", f"llm_committee.pivot.{module}"]
    if setting == "quality":
        cmd += ["--source", str(output / roster / "dyadic")]
    else:
        cmd += ["--plan", str(output / "plans" / f"{setting}-plan.json"), "--roster", roster]
        if setting == "dyadic":
            cmd += ["--public-history"]
    cmd += [
        "--output",
        str(destination),
        "--live",
        "--questions",
        "1" if gate else "30",
        "--max-inflight",
        "32",
        "--workers",
        "8",
    ]
    print(canonical({"starting": roster, "setting": setting, "gate": gate, "log": str(log)}), flush=True)
    with log.open("a") as stream:
        result = subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{roster}/{setting} exited {result.returncode}; inspect {log}")
    print(canonical({"finished": roster, "setting": setting, "gate": gate}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--gate", action="store_true", help="Paid mixed-family one-question gates only")
    args = parser.parse_args()
    output = args.output.resolve()
    selection, specs = prepare(output)
    if not args.live and not args.gate:
        print(json.dumps({"selection": selection, "execution": specs}, indent=2))
        return
    if args.gate:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(
                pool.map(lambda setting: launch_one(output, "mixed_family", setting, gate=True), ("dyadic", "triadic"))
            )
        return
    for setting in ("dyadic", "triadic"):
        audit = read(output / "mixed_family" / f"{setting}-gate-audit.json")
        if (
            audit["questions_reconstructed_exactly"] != 1
            or audit["completed_requests"] != audit["requests_reconstructed_exactly"]
        ):
            raise ValueError("The one-question live gate must pass its exact request audit before scaling")

    # Six debate jobs and three dyadic-E jobs, with no more than two pools active.
    # Each dyadic-E phase waits for its own completed debate; other chains continue.
    def chain(roster, setting):
        launch_one(output, roster, setting)
        if setting == "dyadic":
            from llm_committee.pivot.quality_study import prepare as prepare_quality

            manifest, _ = prepare_quality(output / roster / setting)
            assert manifest["plans"] == read(output / "quality-plan.json")
            launch_one(output, roster, "quality")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(chain, roster, setting) for roster in ROSTERS for setting in ("dyadic", "triadic")]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
