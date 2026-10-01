# Main mixed committee: 60 approved questions

Launched 2026-09-26 07:43 UTC (00:43 America/Los_Angeles), after the user explicitly
requested no budget limit. **Completed at 21:24:01 UTC (14:24:01 America/Los_Angeles):
60/60 questions fully complete; 1,080/1,080 paths successful; no technical task remains missing.**
The earlier checkpoints below are historical, not current missingness.

Results: [self-contained HTML summary](main-mixed-results-2026-09-26.html) and
[unrounded JSON tables](main-mixed-results-2026-09-26.json). The report uses the new A–E
definitions, all four A/B agreement categories, sampled C position comparisons, same-event D
probability reads, and order-reversed E preferences. It does not merge old-paper results.
Seven C responses do not select an original option; numeric denominators explicitly reflect
this semantic unavailability, which is separate from request completion.

Offline final audit: the predecessor checkpoint's recorded hash still matches; all 6,916
imported requests, raw responses, parsed values and statuses are unchanged. Imported charge
columns are zero in the new journal because their costs are carried once in the continuation
ledger. The new journal adds exactly 32 completed calls and contains 6,896 valid scientific
outputs. Report regeneration is deterministic and verifies saved metrics against probabilities,
pair references, answer-order verdicts and planned counts. Full local suite after adding the
report tests: **307 passed, 3 optional native-runtime tests skipped**.

To regenerate without any API calls:

```bash
.venv/bin/python -m llm_committee.pivot.results_report \
  runs/main-mixed-60-20260926/live-transport-retry \
  docs/main-mixed-results-2026-09-26.html
```

## Latest: logged retries and failure isolation; 64-request continuation

### Connection retry and missing-data backfill (latest)

The user pointed out that connection failures should be retried before being skipped. The
previous direct-skip policy was too conservative and has been replaced: known Databricks
connection/timeout errors receive at most two additional identical attempts (0.5/1.0-second
backoff), sharing the same per-request attempt budget as output-format failures. Only exhaustion
marks a task missing. Original failure rows and their unresolved reservations remain unchanged;
no claim of zero provider usage is made. Billing/configuration/probability-integrity checks remain.

Latest run: `runs/main-mixed-60-20260926/live-transport-retry/`, launched **21:23:14 UTC**
(runner PID 54088; sleep guard 54089). Console: `live-transport-retry-console.log` in the parent
directory. It imported all 6,916 physical records from `live-inkling-retry/`, reused **6,864 valid
scientific outputs**, and generated exactly **17 retries plus 15 previously blocked tasks**.
All 17 retries succeeded on their first additional attempt; no successful response was regenerated.
Final counts: 6,896 valid scientific calls, 3,513 unique replies, 480 sampled events, 818 C readings,
673 unique C text judgments, 540 D-choice readings, 316 D-text pairs, and 180 final-answer pairs
judged in both orders. All 1,080 paths succeeded. Across the lineage, 49 logical requests recovered
after 50 additional attempts, with zero retry exhaustion (32 format/contract and 17 transport recoveries).
The final segment adds $0.12330134 to the inherited ledger of $67.831798663, totaling
**$67.955100003 in estimated charges plus unresolved reservations**, not a provider invoice.
Preflight verified every saved scientific request offline with zero API calls. Tests cover
connection recovery, two-retry exhaustion, identical requests, immutable failures/reservations,
and filling prior debate/measurement/judge gaps without regenerating successful outputs.

Reports now separate `request_retries` (unduplicated totals), `format_retries`, and
`transport_retries`; a logical request encountering both error kinds can belong to both subsets.
The historical `::format-retry:` storage-key spelling is retained for compatibility only.

The completion figures below describe the **previous checkpoint**, before this backfill.

**Finished at 21:03:54 UTC (14:03:54 America/Los_Angeles).** All 60 questions processed:
53 fully complete, 7 with explicitly recorded missing data. Of 1,080 planned debate trajectories,
1,077 succeeded and 3 failed; none remain pending or upstream-blocked as a separate path status.
The 3 failed paths share 2 failed debate nodes. Seventeen connection-failed tasks (including
measurements/judgments) are preserved as unavailable, with dependent missingness reported separately.
No request remains in flight. Final status is `completed_with_failures`, not a failure-free run.

Final files: `live-inkling-retry/report.json`, `trajectory-outcomes.json`, and `questions/*.json`.
Across the lineage, 32 logical requests recovered after 33 additional format/contract attempts;
none exhausted the retry limit. Final Inkling audit: 1 unexpected-reasoning occurrence among
273 distinct C reads, and none among 322 distinct D-text reads. That single C occurrence was
recovered on its first identical retry with zero reasoning and valid probability provenance.
The figures 250/298 below describe the earlier paused checkpoint, not the final denominator.

**Previous continuation:** `runs/main-mixed-60-20260926/live-inkling-retry/`,
launched **21:02:10 UTC** (PID 52723; sleep guard 52724). Read its `progress.json` for
live status. Console: `runs/main-mixed-60-20260926/live-inkling-retry-console.log`.
All earlier directories below are retained checkpoints, not concurrently running studies.

### Inkling direct-read retry (user authorized)

The preceding segment reached 51 processed questions (49 complete, 2 with missing data),
then stopped at 20:42 UTC on `global-row-684/mixed_family/C/initial/2`: Inkling generated
reasoning despite effort `none`. The response contains 693 retokenized reasoning tokens;
it is not a valid direct probability read and is never reused as one.

Read-only audit of all saved Inkling direct measurements found this **one occurrence among
250 C reads**; the 298 distinct D-text reads had no unexpected reasoning (299 physical D-text
attempts because a separate malformed-output retry was already recovered). This establishes
rarity in the collected sample, not a guarantee of the endpoint's disable-reasoning behavior.

At the user's request, this specific Inkling response-contract failure now uses the existing
maximum of two additional identical attempts. Only failed measurements are retried. Successful
Inkling/other-model outputs are reused, all settings remain unchanged, and invalid reasoning
responses are preserved rather than stripped or accepted. Exhaustion marks dependent measurements
missing without deleting the valid debate. Other probability-integrity failures still stop.
The continuation imports all 6,376 physical records, carrying forward $49.529704766 in estimated
charges plus reservations exactly once. No full question, debate or successful measurement reruns.
At 21:02:34 UTC the affected measurement's **first retry was completed** with zero reasoning
tokens and an intact probability readout (estimated retry cost $0.00174002); the original invalid
record remains unchanged in the source. The continuation was running and had processed 53/60
questions. The updated retry tests cover exact requests, the two-additional-attempt limit,
first valid acceptance, exclusion of reasoning-bearing outputs, scoped failure isolation and
offline replay without resampling. Full ordinary suite: 289 passed, 3 optional skips.
Native-renderer plus failure-policy suite: 31 passed. Read-only verification confirmed all
6,376 imported records' requests, statuses, responses and parsed values are unchanged; the
source checkpoint hash also matches. Imported charges are carried forward once, not billed again.

The authorized policy is implemented as `--retry-format-failures` with the ready-node scheduler:

- Original attempt plus at most **two additional identical attempts**. Keep the first format-valid
  response; never select according to its label or stance. Every attempt has a separate journal row
  and charge; SQL retry keys differ, but the actual model request/payload does not.
- Exhausted debate replies block only their descendants. Failed C/B/D reads or judgments mark
  dependent measurements missing, not the underlying debate. Failed initial answers block only
  requests actually needing them. The artificial baseline-C execution gate is removed.
- No smaller-tree synthesis is silently substituted: E is unavailable when an intended synthesis
  input is missing. Missing judgments are not ties and missing changes are not zeros.
- All 1,080 planned trajectories get success/failed/blocked/pending outcomes. Shared-prefix
  failures can affect several paths. Retained valid prefixes are explicitly partial observations;
  final analyses must use the relevant observed support, not treat them as complete trajectories.
- `progress.json` and `report.json` distinguish processed questions from fully successful ones,
  report task support by purpose and list exhausted requests. `trajectory-outcomes.json` contains
  all paths and causes; partial question reports retain valid outputs and explicit missingness.
- Known Databricks connection/timeout failures receive bounded identical retries before being
  marked unavailable; their original cost reservation is retained. Independent work continues. Unknown provider,
  billing, configuration and probability-integrity errors still stop for inspection.

Verification: 284 ordinary tests passed (3 optional skips); 64 native-renderer/Databricks/study/scheduler/
failure-policy tests passed. Tests inject exhausted debate, C, initial, B, D-text and E failures,
verify unaffected work finishes and partial results replay without calls, and enforce lifetime
retry limits, immutable originals, exact request hashes, first-valid selection and cost accounting.
Real checkpoint preflight verified **1,078 completed scientific requests**, including the saved
diagnostic retry, with zero API calls. Import contains 1,081 physical records: the original format
failure and two unused legacy supplemental-score failures remain. Inherited estimated cost is
**$5.283197817**, including the diagnostic exactly once.

The full 60-question synthetic rerun also passed: 6,896 requests, 1,080 successful paths,
peak 64 in-flight tasks. Every request hash, parsed output and question A–E record matches
the previous full-bank synthetic run. Synthetic data are not main-study results.

### Live continuation and OAuth concurrency fix

At 20:27:52 UTC, `live-requests64-retries/` started, imported the diagnostic and checkpoint,
and completed 40 new calls (9 questions now complete). It stopped after one worker's
Databricks OAuth initialization failed during concurrent client construction. All in-flight
requests drained: 40 valid and 2 malformed replies, with no pending, uncertain or unknown-billing
records. Cost for this short segment: $0.21520759. The format failures remain eligible for their
two additional attempts; they were not the cause of the global stop.

Databricks workers now share one SDK configuration per profile and serialize authentication/
refresh only. HTTP generation remains concurrent; no payload or scientific setting changes.
An auth-only check initialized 64 clients successfully without model calls. Offline tests verify
one shared configuration, no overlapping auth refresh and overlapping HTTP requests.

At **20:29:30 UTC** the next continuation launched from that checkpoint:

- Root: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-requests64-retries-auth/`
- Log: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-requests64-retries-auth-console.log`
- Launch PID: 50628; sleep guard: 50629. These are launch identities, not a guarantee they remain alive.
- 1,123 physical records imported unchanged; inherited estimated cost **$5.498405407**, no double charge.
- Eight rolling questions, one global cap of 64, same routes/samples/prompts, no spending cap.

That segment stopped after 107 completed calls because a B judgment had `ConnectError`
with no returned response. The format policy successfully recovered a malformed Qwen debate
reply before the stop. The connection failure is distinct: its original uncertain record and
$1.1354112 reservation remain unchanged; no response/zero charge is invented.

### Earlier continuation: local connection failures also isolated

At **20:32:47 UTC**, `live-requests64-resilient/` launched with the policy above. The single
unavailable B judgment is marked missing without regeneration; its discussion and other layers
do not depend on it. A narrow set of known Databricks connection/timeout exceptions receives
the same treatment going forward. Format errors still receive at most two retries.

- Root: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-requests64-resilient/`
- Log: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-requests64-resilient-console.log`
- Launch PID: 50788; sleep guard: 50789 (check current status before using either PID).
- 1,231 physical records imported; 1,225 valid scientific requests verified offline.
- Inherited estimated charges **plus unresolved reservation**: $7.169034890. This is not an invoice.
- Separate counters identify exhausted-format requests versus `transport_missing_requests`.

This segment completed 827 new scientific requests and reached 15 processed questions, then
stopped on Inkling D-text output consisting of `G` plus an unsolicited paragraph in a second
visible block. Native token alignment rejected the concatenated text before the ordinary
single-letter parser ran. This is a malformed answer, not a reason to change probability rules.

The retry classifier now also handles this specific native-alignment case **only when the
same output-contract parser independently rejects the visible text**. Well-formed answers
with invalid probability provenance still stop for inspection; reasoning checks remain.
The new current continuation imports all 2,066 physical records (2,052 valid scientific
requests) without regeneration, preserving six unavailable connection records. Inherited
charges plus outstanding reservations are $21.412409264, not a provider invoice. It retries
the saved malformed D response under the original request and the same two-retry limit.
At 20:36:29 UTC that D retry was confirmed completed, the study was running with 63 requests
in flight, and 16 questions had finished processing. This is a timestamped checkpoint, not
a claim that all 60 are complete. Seven format-failed requests had recovered across the lineage;
seven connection failures remained separately recorded as unavailable, without stopping the run.

The latest root's `trajectory-outcomes.json` covers all planned paths. A trajectory can
succeed while an associated B/C/D measurement is missing; denominators are reported separately.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u -m llm_committee.pivot study-run \
  --bank docs/phase-1-question-bank.json --workers 8 --max-inflight-requests 64 \
  --continue-from runs/main-mixed-60-20260926/live-inkling-retry \
  --retry-format-failures \
  --output runs/main-mixed-60-20260926/live-transport-retry --no-budget-limit
```

## Earlier: incomplete-response investigation finished; main study paused

The user has now authorized up to two retries with logging and isolating failures rather
than stopping unrelated work, while requesting that this specific case be investigated first.
One identical-request diagnostic retry succeeded: `A` plus the full position paragraph,
zero reasoning tokens, cost estimate $0.0027412. The original failure is unchanged and the
successful retry is stored separately, not yet imported. No full-study calls were started.
The main runner's general retry/isolation policy still needs implementation; this is no
longer waiting for retry permission. See the [investigation](terra-position-read-investigation-2026-09-26.md).

## Earlier scheduler upgrade: global 64-request pool (offline verified)

The user approved within-question branch/tone/measurement parallelism and a global limit
of 64 in-flight requests. Implemented as `--workers 8 --max-inflight-requests 64`: at most
eight active questions, each completed question immediately replaced, and ONE shared request
pool across all questions and layers. This does not multiply 64 by question, tone or branch.
This version has **not yet resumed main-study generation**. Retry/isolation was subsequently
authorized and the one-record diagnostic succeeded, as recorded above; the main-run failure
handling and import of the successful retry have not yet been implemented.

Execution invariants:

- Three initial member responses can run concurrently. Baseline C reads then run independently;
  the existing baseline-read gate still precedes debate roots.
- Once baseline prerequisites are available, all three tones and both root paths can proceed.
  A child waits only for its own parent; shared-prefix nodes are submitted once by request key.
- B and C reads run when their own inputs are available; C text judgments wait for both position
  reads. D-text arms wait for the incoming peer message and fixed previous position, not the
  receiver's subsequent reply. Neither measurement nor judge output enters discussion history.
- No-debate synthesis depends on initial answers; each debated synthesis waits for all unique
  nodes of its own tone. The two answer-order judgments wait for the matching synthesis pair.
- Only the coordinator mutates dependency state. API workers have thread-local provider clients
  and separate SQLite connections; reservation/duplicate checks remain transactional.
- Failures stop new dispatch and preserve in-flight results; no automatic resampling was added.
  Every completed question is replayed offline through the original serial report builder,
  verifying exact request hashes before reporting the same A–E measurements.

Verification: 254 ordinary tests passed (3 optional skips), and 20 native-renderer/main-study/
scheduler tests passed. Full approved-bank synthetic run: **60/60 questions, 6,896/6,896 calls,
zero failures, peak 64 in-flight tasks and eight active questions**. All 6,896 request hashes
and parsed outputs, plus all 60 A–E question records, match the previous serial-within-question
dry run exactly. This is offline validation, not a claim that provider rate limits were tested.

Dry-run artifacts: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/dry-run-requests64/`

The actual saved live checkpoint was also inspected read-only with native tokenizers: all
1,077 completed scientific requests match the new graph exactly; the only unresolved
scientific record remains the original missing-paragraph response. The two unused legacy
supplemental-score failures remain archived. No source response or charge was changed.

## Last live execution: eight-question rolling pool (stopped)

**Historical status: stopped on an incomplete model response at 18:55 UTC.** The eight-worker
continuation completed 28 new calls before Terra returned only `A` for
`archived-global-14/mixed_family/C/initial/0`, omitting the required position paragraph.
The saved HTTP response is 200, `finish_reason=stop`, content exactly `A`, zero reasoning
tokens; the request allowed 1,200 output tokens and had no stop sequence. This is not
recoverable redundant formatting: the text is missing from the provider response itself.
The raw response and its charge are retained; no invented paragraph, prompt change or silent
omission was performed. A later authorized diagnostic retry succeeded (see above). All other
in-flight calls finished before the process exited; eight questions remain fully complete.
New segment cost: $0.117801576, excluding the separate diagnostic retry cost.

The user approved increasing question concurrency from three to eight and requested
immediate replenishment whenever a question finishes. The existing scheduler already uses
`FIRST_COMPLETED`: it fills the freed slot from the remaining approved bank, without waiting
for the other questions in a batch. Within-question calls remain serial and branch-isolated.

The three-worker process was stopped cooperatively at 18:54 UTC, after its in-flight calls
finished. Its final `blocked` summary contains only the three intentionally cancelled question
workers, not new API failures. There were no pending or uncertain calls at handoff. The new
eight-worker process imports all 1,051 saved records (1,049 completed and two obsolete,
unused supplemental-score failures). Eight fully completed questions replay offline; partial
questions reuse all completed requests and generate only what is missing. The prior estimated
cost of $5.162655041 is carried forward once; no saved response is regenerated or double-charged.

Historical root: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-workers8/`

Historical console: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-workers8-console.log`

Launch PID: 44838; `caffeinate` PID: 44839 (both historical; the process has exited).
Check current process/status before reusing a PID. Do not rerun the command below without
resolving the saved incomplete response; the journal deliberately rejects automatic retries.
Question/model selection, routes, samples, prompts, reasoning and measurements are unchanged.
No spending cap was added. All previous run directories remain preserved.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u -m llm_committee.pivot study-run \
  --bank docs/phase-1-question-bank.json --workers 8 \
  --continue-from runs/main-mixed-60-20260926/live-json-tolerant \
  --output runs/main-mixed-60-20260926/live-workers8 --no-budget-limit
```

Verification: 247 ordinary tests passed (3 optional tests skipped), plus 13 native-renderer/
main-study tests passed. Added eight-worker output-equivalence/boundedness coverage and a
rolling-refill test that holds one question until the ninth starts. Partial-checkpoint tests
also verify changing concurrency on continuation without duplicate calls. Read-only validation
of the actual handoff confirmed unchanged source records and scientific settings.

## September 26 format recovery and continuation

The original run stopped at 07:55 UTC with six questions complete and three partial.
Qwen's saved reply at `archived-global-9/mixed_family/neutral/debate/n08` contained
`agreement` twice, both exactly `partially_agreed`. This was our local JSON parser,
not DSPy or a transport failure. The user approved tolerating this formatting redundancy.
The parser now accepts identical repeated values; conflicting values and other schema
violations still fail. Prompts, labels, model settings and measurements are unchanged.

At 18:48 UTC a new continuation started with three question workers and no spending cap.
All 813 prior records were imported. Exactly one saved response was reparsed offline,
without regeneration or changing its raw text/charge. The two obsolete supplemental-score
failures remain archived and unused. Six completed questions replayed offline; three partial
questions replayed their cached prefixes and resume only missing calls. Source journals and
logs remain untouched. Prior estimated cost including this run's pilot lineage is $4.072825253;
imported calls have zero new charge, preventing double-counting.

Previous root (preserved): `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-json-tolerant/`

Previous console: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-json-tolerant-console.log`

Continuation PID at launch: 44349; `caffeinate` PID: 44350. Verify live status rather than
assuming these PIDs remain active. This process has now been superseded by the eight-worker continuation above.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u -m llm_committee.pivot study-run \
  --bank docs/phase-1-question-bank.json --workers 3 \
  --continue-from runs/main-mixed-60-20260926/live \
  --reconcile-identical-json-duplicates \
  --output runs/main-mixed-60-20260926/live-json-tolerant --no-budget-limit
```

Recovery verification: 245 ordinary tests passed (3 optional tests skipped), plus 11 native
renderer/main-study tests passed. Tests cover identical/conflicting duplicates, unchanged
raw responses and source journals, zero-cost recovery, and partial-main continuation.
Read-only validation of the actual checkpoint confirmed exactly one recovery and no changed
requests, raw responses or charges. Lint and whitespace checks passed.

## Scope and execution

- All 60 approved questions; friendly, neutral and hostile once per question: 180 debates.
- Terra / Qwen3.8-27B / Inkling; Terra chairman; one Gemini 3.8 Flash B/C/E judge.
- Up to eight active questions, replenished as each finishes. The current scheduler
  shares up to 64 in-flight requests across dependency-ready nodes; earlier stopped segments
  used serial calls within each question. Branch histories remain isolated, with the
  same initial answers across tones. Six depth-five paths per debate, unchanged seeded routing
  and shared B/C/D sample.
- D uses original top-20 probabilities, absent candidates zero-filled, then normalized.
  No supplemental scoring, entropy matching or top-two margin analysis.
- No budget stopping limit. Token usage and estimated cost are still journaled. The latest
  continuation uses the bounded format retries and known-transport isolation above; unknown billing,
  unexpected reasoning or invalid probability provenance still stop for inspection.
- Other rosters, A instruction-removal ablation and human annotation are not included.

The three pilot questions were replayed from saved responses before the first new paid
call, validating exact request hashes without constructing a live provider for that replay.
Their 337 completed calls are reused. Two old failed supplemental-score records remain in
the imported lineage but are unused by the new D policy. The full plan has 6,896 logical
calls, so 6,559 new calls are planned if no failures occur. Technical attempts and imported
records are not additional research repetitions.

## Original run files (preserved)

Root: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live/`

- `progress.json`: status, completed questions, failures, new-call counts and costs.
- `requests.sqlite3`: durable requests, responses, parsed outputs and accounting.
- `questions/<question_id>.json`: per-question A–E records, saved after completion.
- `manifest.json`, `question-bank.json`, `implementation-start.tar.gz`: frozen configuration,
  inputs, original question metadata, routes/sample and source snapshot.
- `report.json`: final or stopped-run summary; appears when the process ends.

Console log: `/Users/Chen/Documents/research/llm-committee/runs/main-mixed-60-20260926/live-console.log`

Initial process PID: 10912. A separate `caffeinate -i -w 10912` process prevents idle sleep
only while this run exists. Process IDs are historical and must be checked before use.
Pending-call amounts in `new_charged_or_reserved_usd` are conservative reservations, not
already-incurred charges. Completed-call amounts are token-based estimates, not invoices.

## Original launch command

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u -m llm_committee.pivot study-run \
  --bank docs/phase-1-question-bank.json --workers 3 \
  --continue-from runs/mixed-family-pilot/offline-topk-zero-fill-20260926 \
  --output runs/main-mixed-60-20260926/live --no-budget-limit
```

The process is detached and logs to the path above. Do not launch another copy while it is
active; a process lock also prevents simultaneous ownership. After a clean interruption,
the same command can resume unchanged requests under the same manifest. Pending, uncertain
or invalid new requests require inspection first; restarting does not retry them. Code or
protocol changes cannot silently resume into this directory.

## Prelaunch verification

- 231 ordinary tests passed, 3 optional native-renderer tests skipped in the project environment.
- The native-renderer and main-study tests passed separately: 10 tests, no model calls.
- Full 60-question synthetic dry run: 60 completed, 6,896 completed mock calls, no failures.
- Bounded concurrent and serial synthetic outputs match exactly at question-data level.
- Tests cover explicit uncapped accounting, fail-stop scheduling, no duplicate dispatch,
  checkpoint expansion, model/protocol mismatch rejection and completed-question replay.
- Approved bank SHA-256: `f3370d6fd67bb958521f6dc77b1e80d7abd2f2a0f70efd578e952415fb589662`.
- Lint and whitespace checks passed. No manuscript edits, commit or push.
