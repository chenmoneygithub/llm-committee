"""Wait for the nine extension journals, audit them, then publish the six tabs.

This process never dispatches model calls. Missing exhausted measurements stay
missing. A blocked/interrupted journal stops publication instead of substituting
an incomplete cohort.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from llm_committee.pivot.study import atomic_json
from scripts.report_scaled_public_history import publish
from scripts.scale_public_history import OUTPUT, ROOT, execution_specs, read


def readiness(output):
    specs = execution_specs(output)
    ready, progress = True, []
    for spec in specs:
        source = Path(spec["output_30"])
        path = source / "progress.json"
        report = read(path) if path.exists() else {}
        status = report.get("status", "queued")
        progress.append(
            {
                "roster": spec["roster"],
                "setting": spec["setting"],
                "status": status,
                "completed": report.get("completed_questions", 0),
                "target": 30,
            }
        )
        if status in ("blocked", "stopped"):
            raise RuntimeError(
                f"Unfinished {spec['roster']}/{spec['setting']}: {status}; do not publish partial expansion"
            )
        final = source / "report.json"
        complete = final.exists() and read(final).get("status") in ("completed", "completed_with_failures")
        ready &= complete and report.get("completed_questions") == 30
    return ready, progress


def finish(output, audit_python=sys.executable):
    ready, _ = readiness(output)
    if not ready:
        raise ValueError("Extension is not complete")
    specs = execution_specs(output)
    audits = []
    for spec in specs:
        module = {"dyadic": "audit_stateful_dyadic", "triadic": "audit_triadic", "quality": "audit_quality_study"}[
            spec["setting"]
        ]
        audit = output / spec["roster"] / f"{spec['setting']}-full-audit.json"
        log = audit.with_suffix(".log")
        with log.open("w") as stream:
            result = subprocess.run(
                [
                    str(audit_python),
                    "-m",
                    "scripts.audit_frozen_run",
                    spec["output_30"],
                    "--kind",
                    module.removeprefix("audit_"),
                    "--output",
                    str(audit),
                ],
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Audit failed; inspect {log}")
        audits.append(str(audit))
        print(json.dumps({"audit_passed": str(audit)}), flush=True)
    dashboard = publish(output)
    atomic_json(output / "finalization.json", {"status": "published", "audits": audits, "dashboard": str(dashboard)})
    print(json.dumps({"published": str(dashboard)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--audit-python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    try:
        while args.wait:
            ready, progress = readiness(args.output)
            print(json.dumps({"extension_progress": progress}), flush=True)
            if ready:
                break
            time.sleep(30)
        finish(args.output, args.audit_python)
    except Exception as exc:
        atomic_json(args.output / "finalization.json", {"status": "blocked", "error": str(exc)})
        raise
