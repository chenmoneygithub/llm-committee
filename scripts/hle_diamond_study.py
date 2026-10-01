"""Private HLE-Diamond pilot: objective scoring and the archived GOQA tone design."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from llm_committee.pivot import supergpqa
from llm_committee.pivot.models import Message, PilotConfig, canonical, digest
from llm_committee.pivot.prompts import TONE_TEXT
from llm_committee.pivot.strong_agreement import agreement_manifest
from llm_committee.pivot.supergpqa import BenchmarkQuestion, SuperGPQAGraph, freeze
from llm_committee.pivot.taskgraph import QuestionGraph, Task

ROOT = Path(__file__).resolve().parents[1]
VERSION = "hle-diamond-paired-turn-tone-2026-09-28-v1"
REVISION = "04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23"
SEED = 20260928
PLAN = ROOT / "runs/hle-diamond-20-20260928-plan"
BANK = PLAN / "question-bank.json"
RUN = ROOT / "runs/hle-diamond-20-20260928/live"
TEMPLATE = ROOT / "docs/turn-tone-dyadic-shared-plan-2026-09-26.json"
ARMS = ("original", "alternate")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def implementation_hash():
    return digest(
        {
            "hle_diamond_study.py": sha(__file__),
            **{p.name: sha(p) for p in sorted(Path(supergpqa.__file__).parent.glob("*.py"))},
        }
    )


def parse_options(raw):
    """Split a native lettered option block, never invent or reorder options."""
    markers = list(re.finditer(r"(?m)^\s*([A-Z])[.)]\s+", raw))
    for start in reversed([i for i, m in enumerate(markers) if m.group(1) == "A"]):
        block = markers[start:]
        labels = [m.group(1) for m in block]
        if len(labels) < 2 or labels != [chr(65 + i) for i in range(len(labels))]:
            continue
        stem = raw[: block[0].start()].strip()
        options = tuple(
            raw[m.end() : block[i + 1].start() if i + 1 < len(block) else len(raw)].strip() for i, m in enumerate(block)
        )
        if stem and all(options):
            return stem, options
    raise ValueError("Native answer options require inspection; do not redraw this question")


def download_bank(path=BANK):
    """Authorized HF read; gated text stays under the git-ignored runs directory."""
    path = Path(path)
    if path.exists():
        return read(path)
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem, get_token

    fs = HfFileSystem(token=get_token())
    source = f"datasets/cais/hle-diamond@{REVISION}/data/test-00000-of-00001.parquet"
    with fs.open(source, "rb") as stream:
        rows = (
            pq.ParquetFile(stream)
            .read(columns=["id", "question", "answer", "answer_type", "partition", "image", "category"])
            .to_pylist()
        )
    pool = sorted(
        (r for r in rows if r["partition"] == "reasoning" and r["answer_type"] == "multipleChoice" and not r["image"]),
        key=lambda r: r["id"],
    )
    if len(pool) != 75:
        raise ValueError("Pinned eligible pool changed")
    selected = random.Random(SEED).sample(pool, 20)
    bank = {
        "dataset": "cais/hle-diamond",
        "revision": REVISION,
        "seed": SEED,
        "selection": "Uniform 20 of 75 original text-only reasoning multiple-choice items; no outcome or solver-based screening",
        "pool_ids": [r["id"] for r in pool],
        "pool_content_sha256": digest([{k: v for k, v in r.items() if k != "image"} for r in pool]),
        "selected": [{k: v for k, v in r.items() if k != "image"} for r in selected],
        "privacy": "Local research only: do not publish, commit, re-upload or distribute question plaintext",
    }
    # Freeze before parsing: a formatting issue must not cause an outcome-dependent redraw.
    freeze(path, bank)
    return bank


def prepare(bank_file=BANK, *, mock=False, template_file=TEMPLATE):
    bank_file, template_file = Path(bank_file).resolve(), Path(template_file).resolve()
    bank, template = read(bank_file), read(template_file)
    if digest({k: v for k, v in template.items() if k != "design_sha256"}) != template["design_sha256"]:
        raise ValueError("GOQA tone template changed")
    selected = sorted(bank["selected"], key=lambda r: r["id"])
    originals = sorted(template["plans"], key=lambda p: p["question"]["id"])
    if len(selected) != len(originals) or len({r["id"] for r in selected}) != len(selected):
        raise ValueError("Need one frozen tone template per selected question")
    config = PilotConfig(seed=SEED, judge_model=None, initial_tokens=16384, debate_tokens=16384, chairman_tokens=16384)
    config.validate()
    contexts, plans = {}, []
    for row, tones in zip(selected, originals, strict=True):
        stem, options = parse_options(row["question"])
        q = BenchmarkQuestion.from_dict({"id": "hle-diamond-" + row["id"], "text": stem, "options": options})
        answer = row["answer"].strip()
        if answer not in q.labels:
            raise ValueError(f"Reference answer is not an original option letter: {q.id}")
        contexts[q.id] = {
            "question": asdict(q),
            "answer_letter": answer,
            "domain": row["category"],
            "source_id": row["id"],
            "original_question": row["question"],
        }
        for node, tone in tones["original_tones"].items():
            if (tone == tones["alternate_tones"][node]) != (int(node.rsplit("-", 1)[1]) <= 2):
                raise ValueError("Shared-prefix tone contract changed")
        plans.append(
            {
                "question_id": q.id,
                "tone_template_question_id": tones["question"]["id"],
                "tones": {arm: tones[f"{arm}_tones"] for arm in ARMS},
            }
        )
    manifest = {
        "kind": "offline_mock" if mock else "paid_hle_diamond_pilot",
        "protocol_version": VERSION,
        "implementation_sha256": implementation_hash(),
        "config": asdict(config),
        "agreement_rubric": agreement_manifest(),
        "probability_read_policy": supergpqa.TOPK_ZERO_FILL_POLICY,
        "source": {
            "directory": str(bank_file.parent),
            "bank_file": str(bank_file),
            "bank_sha256": sha(bank_file),
            "dataset": bank["dataset"],
            "revision": bank["revision"],
            "contexts_sha256": digest(contexts),
            "tone_template": str(template_file),
            "tone_template_sha256": sha(template_file),
        },
        "design": {
            "tone_unit": "turn",
            "global_tone_arms": False,
            "same_GOQA_tone_instructions_and_20_schedules": True,
            "shared_T1_T2": True,
            "T3_T4_changed_each_turn": True,
            "pairs": supergpqa.PAIRS,
            "turns_per_path": 4,
            "initial_answers_shared": True,
            "private_position_feedback": False,
            "B_text_judge": False,
            "C": "Programmatic choice changes and correct/incorrect transitions, grouped by previous/current peer labels",
            "D1": "Neutral reasoning-off input replay; identical inputs through T3 reused; TV and delta P(reference answer)",
            "D2": False,
            "E": "Same Terra chairman and same dyad initials, no debate versus each paired four-turn path; exact reference scoring",
            "scope": "20-question pilot on HLE-Diamond text-only reasoning MCQ subset, not a full HLE score",
            "independent_repetitions": 0,
            "tools_and_web": False,
        },
        "execution": {
            "budget_policy": "no_limit_user_requested",
            "failure_policy": supergpqa.FORMAT_RETRY_POLICY,
            "output_limit_retry_policy": supergpqa.OUTPUT_LIMIT_RETRY_POLICY,
            "max_inflight_requests": 32,
            "question_workers": 8,
        },
        "plans": plans,
    }
    counts = Counter(
        t.purpose for p in plans for t in PairedGraph(contexts[p["question_id"]], p, manifest).tasks.values()
    )
    manifest["planned_counts"] = {
        "questions": len(plans),
        "trajectories": 6 * len(plans),
        "logical_calls": sum(counts.values()),
        "by_purpose": dict(counts),
    }
    return json.loads(canonical(manifest)), json.loads(canonical(contexts))


class PairedGraph(QuestionGraph):
    """Compose tested closed-book graphs; shared nodes have one journal key and one owner."""

    def __init__(self, stored, plan, manifest):
        if manifest["protocol_version"] != VERSION:
            raise ValueError("Wrong HLE protocol")
        self.question = BenchmarkQuestion.from_dict(stored["question"])
        self.config = PilotConfig(**manifest["config"])
        self.context, self.plan, self.manifest = stored, plan, manifest
        self.tasks, self.values, self.failed, self.blocked = {}, {}, {}, {}
        self.submitted = set()
        self.bases, self.maps = {}, {}
        self.route = supergpqa.make_route()
        base_manifest = {**manifest, "protocol_version": supergpqa.VERSION}
        for arm in ARMS:
            base = SuperGPQAGraph(stored, {"question_id": self.question.id, "tone": "neutral"}, base_manifest)
            self.bases[arm] = base
            mapping = {key: self.key(arm, key.split(f"/{supergpqa.VERSION}/", 1)[1]) for key in base.tasks}
            self.maps[arm] = mapping
            for old_key, task in base.tasks.items():
                key = mapping[old_key]
                deps = frozenset(mapping[d] for d in task.dependencies)

                def build(values, count, task=task, mapping=mapping, key=key, arm=arm):
                    local = {old: values[new] for old, new in mapping.items() if new in values}
                    request = task.build(local, count)
                    if request.purpose == "debate":
                        node = request.key.rsplit("/", 1)[1]
                        tone = self.plan["tones"][arm][node]
                        instruction = (
                            "For THIS turn, use the following debate instruction:\n"
                            + (TONE_TEXT[tone] or "No additional friendly or hostile tone instruction.")
                            + "\n"
                        )
                        old = "For THIS turn: no additional friendly or hostile tone instruction. "
                        assert request.messages[-1].text.startswith(old)
                        final = Message("user", instruction + request.messages[-1].text[len(old) :])
                        request = replace(request, messages=(*request.messages[:-1], final))
                    return replace(request, key=key)

                if key in self.tasks:
                    if self.tasks[key].dependencies != deps:
                        raise ValueError("Shared request dependency conflict")
                    continue
                self.tasks[key] = Task(key, deps, build, task.parse, task.purpose, task.model)
        reached = set()
        while more := {t.key for t in self.tasks.values() if t.dependencies <= reached} - reached:
            reached.update(more)
        if reached != self.tasks.keys():
            raise ValueError("Invalid composed dependency graph")

    def key(self, arm, suffix):
        shared = suffix.startswith(("initial/", "D1/initial/")) or suffix.endswith("/baseline")
        if suffix.startswith(("debate/", "D1/")) and not suffix.startswith("D1/initial/"):
            depth = int(suffix.rsplit("-", 1)[1])
            shared = depth <= (3 if suffix.startswith("D1/") else 2)
        return f"{self.question.id}/{VERSION}/{'shared' if shared else arm}/{suffix}"

    def report(self, count):
        branches = {}
        for arm, base in self.bases.items():
            mapping = self.maps[arm]
            base.values = {old: self.values[new] for old, new in mapping.items() if new in self.values}
            base.failed = {old: self.failed[new] for old, new in mapping.items() if new in self.failed}
            base.blocked = {old: self.blocked[new] for old, new in mapping.items() if new in self.blocked}
            branch = base.report(count)
            for e in branch["events"]:
                e.update(arm=arm, current_tone=self.plan["tones"][arm][e["node"]])
                node = self.route.get(e["node"])
                e["previous_tone"] = self.plan["tones"][arm][node.parent] if node.parent else None
                e["request_key"] = self.key(arm, f"debate/{e['node']}")
                for side in ("before", "after"):
                    e[f"D1_{side}_request"] = mapping.get(e[f"D1_{side}_request"])
                e["D1_shared_between_arms"] = e["T"] <= 3
            for kind in ("endpoints", "quality", "trajectories"):
                for item in branch[kind]:
                    item["arm"] = arm
            branches[arm] = branch
        return {
            **self.context,
            "protocol_version": VERSION,
            "status": "completed" if self.successful else "completed_with_failures",
            "tone_schedule": self.plan["tones"],
            "initial_positions": branches["original"]["initial_positions"],
            "events": [e for arm, b in branches.items() for e in b["events"] if arm == "original" or e["T"] >= 3],
            **{k: [e for b in branches.values() for e in b[k]] for k in ("endpoints", "quality", "trajectories")},
            "failed": self.failed,
            "blocked": self.blocked,
        }
