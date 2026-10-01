"""Same scientific requests as runner.py, expressed as immutable dependency nodes.

The serial runner remains the canonical report builder and offline request-hash oracle.
Only execution order changes; no measurement output enters debate or synthesis context.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import prompts
from .failures import LocalTaskFailure
from .models import OPEN_MODELS, TONES, Request
from .planning import rng_for, route_from_plan
from .probabilities import RATING_LABELS, distribution, uses_topk_zero_fill
from .storage import RunBlocked, RunStopped


@dataclass(frozen=True)
class Task:
    key: str
    dependencies: frozenset[str]
    build: Callable
    parse: Callable
    purpose: str = ""
    model: str = ""


class QuestionGraph:
    """All mutable state is owned by the coordinator, never by API worker threads."""

    def __init__(self, question, config, plan, manifest):
        if not uses_topk_zero_fill(manifest):
            raise ValueError("Ready-node scheduling requires the original-top-k zero-fill policy")
        self.question = question
        self.tasks = {}
        self.values = {}
        self.submitted = set()
        self.failed, self.blocked = {}, {}
        self.config, self.plan, self.manifest = config, plan, manifest
        isolate = bool(manifest.get("execution", {}).get("failure_policy"))
        route = route_from_plan(plan)
        prefix = f"{question.id}/{config.roster}"

        def key(suffix):
            return f"{prefix}/{suffix}"

        def initial(values):
            return {m: values[key(f"initial/{m}")]["position"] for m in range(3) if key(f"initial/{m}") in values}

        def branch(values, tone):
            # Include only completed nodes; prompt builders themselves select ancestors.
            return {
                n.id: values[key(f"{tone}/debate/{n.id}")]
                for n in route.nodes
                if key(f"{tone}/debate/{n.id}") in values
            }

        def add(
            suffix,
            dependencies,
            purpose,
            model,
            messages,
            effort,
            tokens,
            schema=None,
            *,
            labels=(),
            parse=None,
            cache=False,
        ):
            full_key = key(suffix)
            if full_key in self.tasks:
                raise ValueError("Duplicate task key in question graph")

            def build(values, token_count):
                return Request(
                    full_key,
                    purpose,
                    model,
                    messages(values, token_count),
                    effort,
                    tokens,
                    schema,
                    cache,
                    candidate_labels=labels,
                )

            self.tasks[full_key] = Task(
                full_key,
                frozenset(key(d) for d in dependencies),
                build,
                parse if parse is not None else lambda text: prompts.parse_json(text, schema),
                purpose,
                model,
            )

        initials = [f"initial/{m}" for m in range(3)]
        for member, model in enumerate(config.members):
            add(
                f"initial/{member}",
                (),
                "initial",
                model,
                lambda values, count, m=member: prompts.initial_messages(question, m),
                config.debate_effort,
                config.initial_tokens,
                prompts.INITIAL_SCHEMA,
            )

        baseline_reads = [f"C/{r['id']}" for r in plan["readings"] if r["node_id"] is None]
        for reading in plan["readings"]:
            model = config.members[reading["member"]]
            dependencies = (
                initials if reading["node_id"] is None else [f"{reading['tone']}/debate/{reading['node_id']}"]
            )
            if isolate and reading["node_id"] is None:
                dependencies = [f"initial/{reading['member']}"]
            add(
                f"C/{reading['id']}",
                dependencies,
                "position",
                model,
                lambda values, count, r=reading: prompts.position_messages(
                    question, r, initial(values), route, branch(values, r["tone"])
                ),
                "none",
                config.position_tokens,
                labels=question.labels if model in OPEN_MODELS else (),
                parse=lambda text: prompts.parse_position(text, question),
                cache=config.cache,
            )

        for tone in TONES:
            for node in route.nodes:
                dependencies = [f"{tone}/debate/{node.parent}"] if node.parent else [*initials, *baseline_reads]
                if isolate:
                    root_sender = route.path(node.id)[0].sender
                    dependencies = [f"initial/{node.receiver}", f"initial/{root_sender}"]
                    if node.parent:
                        dependencies.append(f"{tone}/debate/{node.parent}")
                add(
                    f"{tone}/debate/{node.id}",
                    dependencies,
                    "debate",
                    config.members[node.receiver],
                    lambda values, count, t=tone, n=node: prompts.debate_messages(
                        question, t, initial(values), route, branch(values, t), n.id
                    ),
                    config.debate_effort,
                    config.debate_tokens,
                    prompts.REPLY_SCHEMA,
                    cache=config.cache,
                )

        for event in plan["events"]:
            node = route.get(event["node_id"])
            model = config.members[event["member"]]
            if model in OPEN_MODELS:
                dependencies = [f"C/{event['previous_reading']}", *initials]
                if isolate:
                    dependencies = [
                        f"C/{event['previous_reading']}",
                        f"initial/{node.receiver}",
                        f"initial/{route.path(node.id)[0].sender}",
                    ]
                if node.parent:
                    dependencies.append(f"{event['tone']}/debate/{node.parent}")
                for arm in ("argument", "control"):

                    def d_messages(values, count, e=event, n=node, a=arm, m=model):
                        history = branch(values, e["tone"])
                        own = initial(values)
                        peer = history[n.parent]["reply"] if n.parent else own[n.sender]
                        incoming = peer
                        if a == "control":
                            repeats = max(1, round(count(m, peer) / count(m, prompts.FILLER_SENTENCE)))
                            incoming = " ".join([prompts.FILLER_SENTENCE] * repeats)
                        fixed = values[key(f"C/{e['previous_reading']}")]["position"]
                        return prompts.d_text_messages(question, e, own, route, history, fixed, incoming)

                    add(
                        f"Dtext/{event['id']}/{arm}",
                        dependencies,
                        "d_text",
                        model,
                        d_messages,
                        "none",
                        config.probability_tokens,
                        labels=RATING_LABELS,
                        parse=prompts.parse_rating,
                    )
            if config.judge_model:

                def b_messages(values, count, e=event, n=node):
                    history = branch(values, e["tone"])
                    peer = history[n.parent]["reply"] if n.parent else initial(values)[n.sender]
                    return prompts.b_messages(question, peer, history[n.id]["reply"])

                add(
                    f"B/{event['id']}",
                    [f"{event['tone']}/debate/{node.id}"],
                    "judge_b",
                    config.judge_model,
                    b_messages,
                    "low",
                    config.judge_tokens,
                    prompts.B_SCHEMA,
                )

        if config.judge_model:
            for pair in plan["text_pairs"]:
                add(
                    f"Cjudge/{pair['id']}",
                    [f"C/{pair['before']}", f"C/{pair['after']}"],
                    "judge_c",
                    config.judge_model,
                    lambda values, count, p=pair: prompts.c_messages(
                        question,
                        values[key(f"C/{p['before']}")]["position"],
                        values[key(f"C/{p['after']}")]["position"],
                    ),
                    "low",
                    config.judge_tokens,
                    prompts.C_SCHEMA,
                )

        add(
            "E/baseline",
            initials,
            "synthesis",
            config.chairman_model,
            lambda values, count: prompts.synthesis_messages(question, initial(values), route, None),
            config.chairman_effort,
            config.chairman_tokens,
            prompts.SYNTHESIS_SCHEMA,
        )
        for tone in TONES:
            add(
                f"E/{tone}/synthesis",
                [*initials, *(f"{tone}/debate/{n.id}" for n in route.nodes)],
                "synthesis",
                config.chairman_model,
                lambda values, count, t=tone: prompts.synthesis_messages(
                    question, initial(values), route, branch(values, t)
                ),
                config.chairman_effort,
                config.chairman_tokens,
                prompts.SYNTHESIS_SCHEMA,
            )
            if config.judge_model:
                first_left = bool(
                    rng_for(config.seed, question.fingerprint, config.roster, tone, "answer_order").randrange(2)
                )
                for order in range(2):
                    debate_left = first_left if order == 0 else not first_left

                    def e_messages(values, count, t=tone, left=debate_left):
                        debated = values[key(f"E/{t}/synthesis")]["answer"]
                        baseline = values[key("E/baseline")]["answer"]
                        a, b = (debated, baseline) if left else (baseline, debated)
                        return prompts.e_messages(question, a, b)

                    add(
                        f"E/{tone}/order{order}",
                        ["E/baseline", f"E/{tone}/synthesis"],
                        "judge_e",
                        config.judge_model,
                        e_messages,
                        "low",
                        config.judge_tokens,
                        prompts.E_SCHEMA,
                    )

        # Fail before any API dispatch if the graph is malformed.
        reachable = set()
        while True:
            more = {t.key for t in self.tasks.values() if t.dependencies <= reachable} - reachable
            if not more:
                break
            reachable.update(more)
        if reachable != set(self.tasks):
            raise ValueError("Question graph has missing dependencies or a cycle")

    @property
    def complete(self):
        return len(self.values) + len(getattr(self, "failed", {})) + len(getattr(self, "blocked", {})) == len(
            self.tasks
        )

    @property
    def successful(self):
        return len(self.values) == len(self.tasks)

    def ready(self):
        return next(
            (
                t
                for t in self.tasks.values()
                if t.key not in self.submitted
                and t.key not in getattr(self, "blocked", {})
                and t.dependencies <= self.values.keys()
            ),
            None,
        )

    def accept(self, task, request, value):
        if request.candidate_labels:
            distribution(value["_readout"], request.candidate_labels, missing_as_zero=True)
        self.values[task.key] = value
        self.submitted.add(task.key)

    def reject(self, task, failure):
        self.failed[task.key] = {**failure.document(), "purpose": task.purpose, "model": task.model}
        self.submitted.add(task.key)
        while True:
            changed = False
            for child in self.tasks.values():
                if child.key in self.values or child.key in self.failed or child.key in self.blocked:
                    continue
                causes = set()
                for dep in child.dependencies:
                    if dep in self.failed:
                        causes.add(dep)
                    elif dep in self.blocked:
                        causes.update(self.blocked[dep]["causes"])
                if causes:
                    if child.key in self.submitted:
                        raise RunBlocked("In-flight task depends on a failed prerequisite")
                    self.blocked[child.key] = {
                        "status": "blocked",
                        "causes": sorted(causes),
                        "purpose": child.purpose,
                        "model": child.model,
                    }
                    changed = True
            if not changed:
                break

    def restore(self, journal, offline_provider):
        """Verify ALL saved graph nodes, including non-prefix parallel checkpoints, offline."""
        saved = {key for (key,) in journal.db.execute("SELECT key FROM calls")} & self.tasks.keys()
        remaining = set(saved)
        while remaining:
            ready = [self.tasks[key] for key in sorted(remaining) if self.tasks[key].dependencies <= self.values.keys()]
            if not ready:
                # A malformed saved task can be awaiting its next authorized attempt.
                # Its descendants must not already have responses.
                raise RunBlocked("Saved graph requests lack their prerequisite records")
            for task in ready:
                request = task.build(self.values, offline_provider.token_count)
                try:
                    value = journal.call(request, offline_provider, task.parse)
                    self.accept(task, request, value)
                except RunStopped:
                    pass  # Saved invalid attempt; next identical attempt remains ready, never dispatched offline.
                except LocalTaskFailure as exc:
                    self.reject(task, exc)
                remaining.remove(task.key)
