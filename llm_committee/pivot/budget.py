"""Offline pilot estimates from the actual preplanned roster/routing/sample, not mock usage."""

from __future__ import annotations

from collections import defaultdict

from .models import OPEN_MODELS, ROSTERS, Completion
from .probabilities import uses_topk_zero_fill
from .providers import PRICES, cost_usd


def estimate_budget(manifest: dict) -> dict:
    config = manifest["config"]
    members = ROSTERS[config["roster"]]
    zero_fill = uses_topk_zero_fill(manifest)
    scenarios = {}
    for heavy in (False, True):
        # Preserve the earlier budget's conservative input assumptions. These are NOT
        # measured sizes; output assumptions now respect the executable token limits.
        t = (
            {
                "initial": 2000,
                "debate": 16000,
                "position": 16500,
                "initial_position": 4000,
                "b_judge": 3500,
                "c_judge": 2200,
                "e_judge": 5000,
                "baseline": 6000,
                "nine_node_chair": 35000,
            }
            if heavy
            else {
                "initial": 1500,
                "debate": 8000,
                "position": 8500,
                "initial_position": 2500,
                "b_judge": 2000,
                "c_judge": 1400,
                "e_judge": 3200,
                "baseline": 3500,
                "nine_node_chair": 18000,
            }
        )
        rows = defaultdict(lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})

        def add(stage, model, inputs, outputs, ledger=rows):
            row = ledger[(stage, model)]
            row["calls"] += 1
            row["input_tokens"] += inputs
            row["output_tokens"] += outputs
            row["usd"] += cost_usd(model, Completion("", inputs, outputs), closed_provider=config["closed_provider"])

        for plan in manifest["plans"]:
            for model in members:
                add(
                    "initial",
                    model,
                    t["initial"],
                    config["initial_tokens"] if heavy else min(1800, config["initial_tokens"]),
                )
            for _tone in range(3):
                for node in plan["route"]:
                    add(
                        "debate",
                        members[node["receiver"]],
                        t["debate"],
                        config["debate_tokens"] if heavy else min(1800, config["debate_tokens"]),
                    )
            for reading in plan["readings"]:
                model = members[reading["member"]]
                inputs = t["position"] if reading["node_id"] else t["initial_position"]
                add("C", model, inputs, config["position_tokens"] if heavy else min(700, config["position_tokens"]))
                if model in OPEN_MODELS and not zero_fill:
                    add("D_choice_scoring_allowance", model, inputs + 64, 1)
            for event in plan["events"]:
                model = members[event["member"]]
                if model in OPEN_MODELS:
                    for _arm in range(2):
                        add("D_text", model, t["position"], config["probability_tokens"] if heavy else 16)
                        if not zero_fill:
                            add("D_text_scoring_allowance", model, t["position"] + 64, 1)
                if config["judge_model"]:
                    add(
                        "B_judge",
                        config["judge_model"],
                        t["b_judge"],
                        config["judge_tokens"] if heavy else min(1000, config["judge_tokens"]),
                    )
            if config["judge_model"]:
                for _pair in plan["text_pairs"]:
                    add(
                        "C_judge",
                        config["judge_model"],
                        t["c_judge"],
                        config["judge_tokens"] if heavy else min(1000, config["judge_tokens"]),
                    )
                for _order in range(6):
                    add(
                        "E_judge",
                        config["judge_model"],
                        t["e_judge"],
                        config["judge_tokens"] if heavy else min(800, config["judge_tokens"]),
                    )
            outputs = config["chairman_tokens"] if heavy else min(2500, config["chairman_tokens"])
            add("E_baseline", config["chairman_model"], t["baseline"], outputs)
            for _tone in range(3):
                inputs = round(t["baseline"] + (t["nine_node_chair"] - t["baseline"]) * len(plan["route"]) / 9)
                add("E_debate", config["chairman_model"], inputs, outputs)
        total = sum(row["usd"] for row in rows.values())
        regular = total + sum(row["usd"] for (_, model), row in rows.items() if model == "thinkingmachines/Inkling")
        scenarios["high_tokens" if heavy else "central_tokens"] = {
            "rows": [{"stage": stage, "model": model, **row} for (stage, model), row in sorted(rows.items())],
            "logical_calls_including_all_scoring_allowances": sum(row["calls"] for row in rows.values()),
            "usd": total,
            "usd_with_30_percent_reserve": total * 1.3,
            "usd_with_30_percent_reserve_and_inkling_regular_price": regular * 1.3,
        }
    return {
        "kind": "offline_estimate_not_invoice_or_authorization",
        "price_date": "2026-09-25",
        "closed_provider": config["closed_provider"],
        "closed_provider_pricing_multiplier": 1.1 if config["closed_provider"] == "databricks" else 1.0,
        "assumptions": [
            "No cache discounts assumed; native Tinker may report actual cache hits",
            "Output includes reasoning where enabled; high scenario uses configured output ceilings",
            "Input counts are conservative assumptions from the prior budget, not measurements",
            (
                "D uses original top-20 values, zero-fills absent candidates and normalizes; no supplemental calls"
                if zero_fill
                else "Every D-choice AND D-text reading includes one supplemental scoring allowance"
            ),
            "No A instruction-ablation, human audit, taxes, provider repricing or further retries included",
            "Databricks uses standard listed rates at assumed USD 0.07/DBU plus a conservative 10% regional uplift; not an account invoice",
        ],
        "prices_usd_per_million": {
            model: {"input": p.input, "cached": p.cached, "output": p.output}
            for model, p in PRICES.items()
            if model in (*members, config["chairman_model"], config["judge_model"])
        },
        "scenarios": scenarios,
    }
