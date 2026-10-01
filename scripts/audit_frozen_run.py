"""Read-only audit using the implementation snapshot saved with a run.

Loading the run's snapshot keeps old results auditable after unrelated runtime
fixes. Existing auditors still verify its implementation hash and reconstruct
every request. No historical source directory is modified.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import sys
import tarfile
import tempfile
from pathlib import Path, PurePosixPath


def audit_snapshot(source, kind, output=None):
    source = Path(source).resolve()
    if any(k.startswith("llm_committee.pivot") for k in sys.modules):
        raise RuntimeError("Snapshot audit must start in a fresh interpreter")
    import llm_committee

    archive = source / "implementation-start.tar.gz"
    with tempfile.TemporaryDirectory(prefix="committee-frozen-audit-") as directory:
        package = Path(directory)
        with tarfile.open(archive) as tar:
            seen = set()
            for member in tar:
                name = PurePosixPath(member.name)
                if (
                    not member.isfile()
                    or len(name.parts) != 3
                    or name.parts[:2] != ("llm_committee", "pivot")
                    or name.suffix != ".py"
                    or name.name in seen
                ):
                    raise ValueError("Unexpected implementation snapshot member")
                seen.add(name.name)
                (package / name.name).write_bytes(tar.extractfile(member).read())
        spec = importlib.util.spec_from_file_location(
            "llm_committee.pivot", package / "__init__.py", submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        llm_committee.pivot = module
        spec.loader.exec_module(module)
        auditor = importlib.import_module(f"scripts.audit_{kind}")
        result = auditor.audit(source)
        result["implementation_snapshot_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        result["used_archived_implementation"] = True
        if output:
            # Import and use this helper before the temporary package is removed.
            from llm_committee.pivot.study import atomic_json

            atomic_json(output, result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--kind", choices=("stateful_dyadic", "triadic", "quality_study"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_snapshot(args.source, args.kind, args.output)
    print(json.dumps(result, indent=2))
