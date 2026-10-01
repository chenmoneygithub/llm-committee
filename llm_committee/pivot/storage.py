"""Transactional request journal: bounded authorized retries, immutable attempt history."""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from .failures import (
    FORMAT_RETRY_POLICY,
    OUTPUT_LIMIT_RETRY_POLICY,
    LocalTaskFailure,
    local_output_limit_record,
    local_transport_record,
    retry_key,
    retryable_format_record,
)
from .models import Request, canonical, digest
from .probabilities import LOGPROB_MISMATCH_MESSAGES, LogprobMismatch
from .providers import Provider, cost_usd, reservation_usd


class RunBlocked(RuntimeError):
    pass


class BudgetExceeded(RunBlocked):
    pass


class RunStopped(RunBlocked):
    """Cooperative stop: an independent worker failed or the operator requested shutdown."""


class ProbabilityReadBlocked(RunBlocked):
    """Only a known supplemental-score mismatch; not a transport or billing failure."""

    def __init__(self, key: str, reason: str):
        super().__init__(f"Probability read {key} failed: {reason}")
        self.key, self.reason = key, reason


class Journal:
    def __init__(
        self,
        path: Path,
        manifest: dict,
        cap_usd: float | None,
        *,
        progress: Callable[[dict], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ):
        if cap_usd is None:
            if manifest.get("execution", {}).get("budget_policy") != "no_limit_user_requested":
                raise ValueError("An uncapped run requires an explicit no-limit authorization in its manifest")
        elif not math.isfinite(cap_usd) or cap_usd <= 0:
            raise ValueError("A positive finite spending cap is required")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS calls (
            key TEXT PRIMARY KEY, request TEXT NOT NULL, request_hash TEXT NOT NULL,
            status TEXT NOT NULL, charge REAL NOT NULL, response TEXT, parsed TEXT, error TEXT
        )""")
        identity = {"manifest": manifest, "cap_usd": cap_usd}
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if old and old[0] != canonical(identity):
                raise ValueError("Existing run has a different manifest or budget; refusing to overwrite/resume")
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('identity', ?)", (canonical(identity),))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            self.db.close()
            raise
        self.cap_usd = cap_usd
        config = manifest.get("config")
        self.closed_provider = config.get("closed_provider", "direct") if isinstance(config, dict) else "direct"
        self.progress = progress
        self.dispatch_enabled = manifest.get("execution", {}).get("dispatch_enabled", True)
        self.should_stop = should_stop
        policy = manifest.get("execution", {}).get("failure_policy")
        if policy is not None and policy != FORMAT_RETRY_POLICY:
            self.db.close()
            raise ValueError("Unsupported failure policy")
        self.format_retries = policy is not None
        output_policy = manifest.get("execution", {}).get("output_limit_retry_policy")
        if output_policy is not None and (output_policy != OUTPUT_LIMIT_RETRY_POLICY or not self.format_retries):
            self.db.close()
            raise ValueError("Unsupported output-limit retry policy")
        self.output_limit_retries = output_policy is not None

    def import_checkpoint(self, descriptor: dict, rows: list[dict]) -> None:
        """Copy a frozen checkpoint once, retaining failures and charging no new API cost.

        Every reused call still goes through the ordinary exact request-hash check.
        Original costs remain in the source and the manifest's continuation ledger.
        """
        if digest(rows) != descriptor["import_rows_sha256"]:
            raise ValueError("Checkpoint rows changed after validation")
        identity = canonical(descriptor)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT value FROM metadata WHERE key='continuation'").fetchone()
            if old:
                if old[0] != identity:
                    raise ValueError("Continuation checkpoint changed")
            else:
                if self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0]:
                    raise ValueError("Cannot import a checkpoint into a nonempty journal")
                self.db.executemany(
                    "INSERT INTO calls VALUES(:key,:request,:request_hash,:status,0,:response,:parsed,:error)", rows
                )
                self.db.execute("INSERT INTO metadata VALUES('continuation',?)", (identity,))
                self.db.execute(
                    "INSERT INTO metadata VALUES('imported_keys',?)", (canonical([r["key"] for r in rows]),)
                )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    @property
    def charged_usd(self) -> float:
        return self.db.execute("SELECT COALESCE(SUM(charge), 0) FROM calls").fetchone()[0]

    def call(self, request: Request, provider: Provider, parse: Callable[[str], dict]) -> dict:
        """Use first valid attempt; each API attempt retains a separate immutable record."""
        if not self.format_retries:
            return self._call_once(request, provider, parse)
        attempts = []
        for attempt in range(3):
            storage_key = retry_key(request.key, attempt)
            attempts.append(storage_key)
            try:
                return self._call_once(request, provider, parse, storage_key=storage_key)
            except RunBlocked:
                row = self.db.execute(
                    "SELECT status,response,error,request_hash,request FROM calls WHERE key=?", (storage_key,)
                ).fetchone()
                if (
                    row is None
                    or row[3] != digest(request.document())
                    or canonical(json.loads(row[4])) != canonical(request.document())
                ):
                    raise
                saved = dict(zip(("status", "response", "error"), row[:3], strict=True))
                transport = local_transport_record(request.document(), saved, self.closed_provider)
                output_limit = self.output_limit_retries and local_output_limit_record(
                    request.document(), saved, self.closed_provider
                )
                if not transport and not output_limit and not retryable_format_record(request.document(), saved):
                    raise
                if self.progress:
                    self.progress(
                        {
                            "event": (
                                "transport_attempt_failed"
                                if transport
                                else "output_limit_attempt_failed" if output_limit else "format_attempt_failed"
                            ),
                            "key": storage_key,
                            "logical_key": request.key,
                            "attempt": attempt,
                            "error": row[2],
                        }
                    )
                if attempt == 2:
                    kind = (
                        "failed_after_transport_retries"
                        if transport
                        else "failed_after_output_limit_retries" if output_limit else "failed_after_format_retries"
                    )
                    raise LocalTaskFailure(request.key, attempts, row[2], kind=kind) from None
                if (
                    transport
                    and self.dispatch_enabled
                    and not (self.should_stop and self.should_stop())
                    and self.db.execute(
                        "SELECT 1 FROM calls WHERE key=?", (retry_key(request.key, attempt + 1),)
                    ).fetchone()
                    is None
                ):
                    time.sleep(FORMAT_RETRY_POLICY["transport_retry_backoff_seconds"][attempt])
        raise AssertionError("Unreachable retry loop")

    def _call_once(
        self, request: Request, provider: Provider, parse: Callable[[str], dict], *, storage_key=None
    ) -> dict:
        key = request.key if storage_key is None else storage_key
        document = request.document()
        request_hash = digest(document)
        hold = reservation_usd(request, closed_provider=self.closed_provider)
        self.db.execute("BEGIN IMMEDIATE")
        try:
            old = self.db.execute("SELECT request_hash,status,parsed,error FROM calls WHERE key=?", (key,)).fetchone()
            if old:
                if old[0] != request_hash:
                    raise RunBlocked("Request changed under an existing key; no silent regeneration")
                if old[1] != "completed":
                    if request.scoring and old[1] == "invalid" and old[3] in LOGPROB_MISMATCH_MESSAGES:
                        raise ProbabilityReadBlocked(request.key, old[3])
                    raise RunBlocked(f"Request {request.key} is {old[1]}; inspect the journal before any paid retry")
                self.db.execute("COMMIT")
                return json.loads(old[2])
            if not self.dispatch_enabled:
                raise RunBlocked(f"Offline reanalysis lacks saved request {request.key}; model dispatch is disabled")
            if self.should_stop is not None and self.should_stop():
                raise RunStopped("Run stopped before dispatch; existing responses preserved")
            if self.cap_usd is not None and self.charged_usd + hold > self.cap_usd:
                raise BudgetExceeded("Spending cap cannot cover the next conservative per-request hold")
            self.db.execute(
                "INSERT INTO calls(key,request,request_hash,status,charge) VALUES(?,?,?,'pending',?)",
                (key, canonical(document), request_hash, hold),
            )
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        # No transaction is held over network I/O. The unique pending row prevents duplicate dispatch.
        if self.progress:
            self.progress(
                {
                    "event": "dispatch",
                    "key": key,
                    **({"logical_key": request.key} if key != request.key else {}),
                    "purpose": request.purpose,
                    "model": request.model,
                    "reserved_total_usd": self.charged_usd,
                }
            )
        try:
            completion = provider.generate(request)
        except BaseException as exc:
            # A timeout can have incurred cost. Keep the full hold; only the explicit
            # bounded outer policy may retry known transport failures in another row.
            self.db.execute("UPDATE calls SET status='uncertain',error=? WHERE key=?", (type(exc).__name__, key))
            raise RunBlocked(f"Provider failed for {request.key}; reserved cost retained, no automatic retry") from exc
        raw = canonical(asdict(completion))
        self.db.execute("UPDATE calls SET response=? WHERE key=?", (raw, key))
        try:
            cost = cost_usd(request.model, completion, closed_provider=self.closed_provider)
            if cost > hold + 1e-9:
                raise RunBlocked("Usage exceeded the documented price/limit reservation; stop and reconcile")
        except (ValueError, RunBlocked) as exc:
            self.db.execute("UPDATE calls SET status='billing_unknown',error=? WHERE key=?", (str(exc), key))
            raise RunBlocked("Cannot safely reconcile provider usage") from exc
        self.db.execute("UPDATE calls SET charge=?,status='received' WHERE key=?", (cost, key))
        try:
            if completion.status != "completed":
                raise ValueError(f"Response status: {completion.status}")
            if request.effort == "none" and completion.reasoning_tokens:
                raise ValueError("Direct measurement unexpectedly generated reasoning tokens")
            parsed = parse(completion.text)
            if request.candidate_labels and request.scoring is None:
                if completion.readout is None:
                    raise ValueError("Missing probability provenance for the actual choice output")
                parsed["_readout"] = completion.readout
        except (ValueError, KeyError, TypeError) as exc:
            self.db.execute("UPDATE calls SET status='invalid',error=? WHERE key=?", (str(exc), key))
            if request.scoring and isinstance(exc, LogprobMismatch):
                raise ProbabilityReadBlocked(request.key, str(exc)) from exc
            raise RunBlocked(
                f"Invalid response for {request.key}; saved raw output, no label defaults or retries"
            ) from exc
        self.db.execute("UPDATE calls SET status='completed',parsed=? WHERE key=?", (canonical(parsed), key))
        if self.progress:
            self.progress(
                {
                    "event": "completed",
                    "key": key,
                    **({"logical_key": request.key} if key != request.key else {}),
                    "cost_usd": cost,
                    "charged_or_reserved_usd": self.charged_usd,
                }
            )
        return parsed

    def audit(self) -> dict:
        rows = self.db.execute("SELECT key,status,charge,request,response,error FROM calls ORDER BY rowid").fetchall()
        source = self.db.execute("SELECT value FROM metadata WHERE key='continuation'").fetchone()
        imported = self.db.execute("SELECT value FROM metadata WHERE key='imported_keys'").fetchone()
        imported_keys = set(json.loads(imported[0])) if imported else set()
        return {
            "charged_or_reserved_usd": self.charged_usd,
            "cap_usd": self.cap_usd,
            "continuation": json.loads(source[0]) if source else None,
            "calls": [
                {
                    "key": key,
                    "status": status,
                    "charge_usd": charge,
                    "reused_without_new_call": key in imported_keys,
                    "request": json.loads(request),
                    "response": json.loads(response) if response else None,
                    "error": error,
                }
                for key, status, charge, request, response, error in rows
            ],
        }

    def close(self) -> None:
        self.db.close()
