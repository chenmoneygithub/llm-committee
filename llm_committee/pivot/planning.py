"""Deterministic route and sample planning, independent of generated content."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

from .models import TONES, PilotConfig, Question, digest


def rng_for(seed: int, *parts: str) -> random.Random:
    return random.Random(int(digest([seed, *parts]), 16))


@dataclass(frozen=True)
class Node:
    id: str
    parent: str | None
    sender: int
    receiver: int
    depth: int


@dataclass(frozen=True)
class Route:
    nodes: tuple[Node, ...]

    def get(self, node_id: str) -> Node:
        return next(n for n in self.nodes if n.id == node_id)

    def path(self, node_id: str) -> tuple[Node, ...]:
        result = []
        node = self.get(node_id)
        while True:
            result.append(node)
            if node.parent is None:
                return tuple(reversed(result))
            node = self.get(node.parent)

    @property
    def leaves(self) -> tuple[Node, ...]:
        parents = {n.parent for n in self.nodes}
        return tuple(n for n in self.nodes if n.id not in parents)

    def previous_own(self, node_id: str) -> Node | None:
        path = self.path(node_id)
        return next((n for n in reversed(path[:-1]) if n.receiver == path[-1].receiver), None)

    def participation(self, node_id: str) -> int:
        path = self.path(node_id)
        return sum(n.receiver == path[-1].receiver for n in path)

    def final_trajectories(self, node_id: str) -> tuple[str, ...]:
        member = self.get(node_id).receiver
        return tuple(
            leaf.id
            for leaf in self.leaves
            if next((n.id for n in reversed(self.path(leaf.id)) if n.receiver == member), None) == node_id
        )

    def validate(self) -> None:
        if len({n.id for n in self.nodes}) != len(self.nodes):
            raise ValueError("Duplicate node IDs")
        roots = [n for n in self.nodes if n.parent is None]
        if len(roots) != 2 or len({(n.sender, n.receiver) for n in roots}) != 2:
            raise ValueError("Need two distinct initial sender/receiver pairs")
        seen: dict[str, Node] = {}
        for n in self.nodes:
            if n.receiver not in range(3) or n.sender not in range(3) or n.sender == n.receiver:
                raise ValueError("Invalid sender/receiver")
            if n.parent is None:
                if n.depth != 1:
                    raise ValueError("Initial replies have depth one")
            elif n.parent not in seen or n.depth != seen[n.parent].depth + 1 or n.sender != seen[n.parent].receiver:
                raise ValueError("Invalid or non-topological parent link")
            children = [child for child in self.nodes if child.parent == n.id]
            if len(children) > 2 or len({child.receiver for child in children}) != len(children):
                raise ValueError("Branches need distinct receivers, at most two")
            seen[n.id] = n
        if len(self.leaves) != 6 or any(n.depth != 5 for n in self.leaves) or not 15 <= len(self.nodes) <= 24:
            raise ValueError("Expected six depth-five paths with 15–24 unique replies")


def make_route(question: Question, seed: int) -> Route:
    rng = rng_for(seed, question.fingerprint, "routing")  # No tone or roster here.
    nodes: list[Node] = []

    def add(parent: Node | None, sender: int, receiver: int) -> Node:
        node = Node(
            f"n{len(nodes):02d}", parent.id if parent else None, sender, receiver, parent.depth + 1 if parent else 1
        )
        nodes.append(node)
        return node

    def extend(node: Node) -> None:
        while node.depth < 5:
            node = add(node, node.receiver, rng.choice([m for m in range(3) if m != node.receiver]))

    pairs = [(a, b) for a in range(3) for b in range(3) if a != b]
    for sender, receiver in rng.sample(pairs, 2):
        extend(add(None, sender, receiver))
    for _ in range(4):
        candidates = [n for n in nodes if n.depth < 5 and sum(c.parent == n.id for c in nodes) == 1]
        parent = rng.choice(candidates)
        existing = next(c.receiver for c in nodes if c.parent == parent.id)
        receiver = next(m for m in range(3) if m not in (parent.receiver, existing))
        extend(add(parent, parent.receiver, receiver))
    route = Route(tuple(nodes))
    route.validate()
    return route


def read_id(tone: str, node: Node | None, member: int) -> str:
    return f"{tone}/{node.id}" if node else f"initial/{member}"


def plan_question(question: Question, config: PilotConfig) -> dict:
    question.validate()
    config.validate()
    route = make_route(question, config.seed)
    rng = rng_for(config.seed, question.fingerprint, config.roster, "sampling")
    pools = {t: [(tone, n) for tone in TONES for n in route.nodes if n.depth == t] for t in range(1, 6)}
    pool_sizes = {t: len(pool) for t, pool in pools.items()}
    selected: list[tuple[str, Node]] = []

    def take(depth: int) -> None:
        selected.append(pools[depth].pop(rng.randrange(len(pools[depth]))))

    for depth in range(1, 6):
        take(depth)
    for depth in rng.choices(list(pools), weights=[1, 1, 1, 1, 1.5], k=3):
        take(depth)
    readings: dict[str, dict] = {}
    pairs: dict[str, dict] = {}
    events = []

    def read(tone: str, node: Node | None, member: int) -> str:
        key = read_id(tone, node, member)
        readings[key] = {
            "id": key,
            "tone": tone if node else "initial",
            "member": member,
            "node_id": node.id if node else None,
            "T": node.depth if node else 0,
            "participation_index": route.participation(node.id) if node else 0,
        }
        return key

    def pair(before: str, after: str, kind: str, event: str) -> str:
        key = f"{before}->{after}"
        item = pairs.setdefault(key, {"id": key, "before": before, "after": after, "kinds": [], "events": []})
        if kind not in item["kinds"]:
            item["kinds"].append(kind)
        if event not in item["events"]:
            item["events"].append(event)
        return key

    for tone, node in selected:
        event_id = f"{tone}/{node.id}"
        previous = read(tone, route.previous_own(node.id), node.receiver)
        initial = read(tone, None, node.receiver)
        current = read(tone, node, node.receiver)
        finals = route.final_trajectories(node.id)
        adjacent = pair(previous, current, "adjacent", event_id)
        final = pair(initial, current, "initial_to_final", event_id) if finals else None
        weight = 1.5 if node.depth == 5 else 1.0
        events.append(
            {
                "id": event_id,
                "tone": tone,
                "node_id": node.id,
                "member": node.receiver,
                "T": node.depth,
                "participation_index": route.participation(node.id),
                "inclusion_probability": (1 + 3 * weight / 5.5) / pool_sizes[node.depth],
                "previous_reading": previous,
                "initial_reading": initial,
                "current_reading": current,
                "adjacent_pair": adjacent,
                "final_pair": final,
                "final_trajectories": list(finals),
            }
        )
    return {
        "question_id": question.id,
        "question_fingerprint": question.fingerprint,
        "route": [asdict(n) for n in route.nodes],
        "events": events,
        "readings": list(readings.values()),
        "text_pairs": list(pairs.values()),
    }


def route_from_plan(plan: dict) -> Route:
    route = Route(tuple(Node(**n) for n in plan["route"]))
    route.validate()
    return route
