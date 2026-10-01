# Main mixed-committee run: preflight, 2026-09-26

**Launch update:** the user subsequently explicitly requested no budget limit. The main-study
entry point and three-worker scheduler have been implemented and tested; the 60-question
mixed run was launched at 2026-09-26 07:43 UTC. The earlier US$100 proposal below was not
adopted. See [run guide](main-mixed-run-2026-09-26.md) for the authoritative progress path.
The following preflight history is retained to explain the checks and earlier blockers.

Status: user requested starting the run; **no new paid calls or main-study process launched** during this preflight. The approved bank contains 60 questions. The user subsequently changed D to the top-20/zero-fill policy below; no numerical main-run spending ceiling has yet been confirmed.

## Scope checked

Use the existing mixed roster (Terra / Qwen3.8-27B / Inkling), Terra chairman, Gemini 3.8 Flash judge, three tones, seed 20260925, and the current v4 prompts/settings. Same-model and same-family rosters, A instruction ablation, and human annotation are outside this first launch.

Planning directly from `phase-1-question-bank.json`, preserving original question text, option order, screening evidence and archived IDs, gives:

| Item | Planned total |
|---|---:|
| Questions / debates | 60 / 180 |
| Initial answers / formal replies | 180 / 3,513 |
| Shared B/C/D sampled events | 480 |
| C position reads / C text judgments | 818 / 673 |
| B judgments | 480 |
| D-choice reads / D-text pairs | 540 / 316 |
| Chairman syntheses / order-swapped E judgments | 240 / 360 |
| Generation/judging calls before optional supplemental scoring | 6,896 |
| Supplemental scoring calls under the updated rule | 0 |

These are full-bank totals, not new-call counts after pilot reuse. The three pilot question contracts, routes, samples, configuration and prompt version match the current plan. Exact-request checkpoint validation is still required at execution; matching plans alone does not validate failed measurements.

## Outstanding launch checks

- The current CLI and question importer deliberately accept only engineering pilots of 1–10 questions. Main-study input validation and bounded question-level scheduling must be implemented and tested, without relabeling the 60-question bank as a pilot. Preserve sequential dependencies within each question and a shared transactional spending guard.
- **D policy changed by the user after preflight:** use original top-20 values only, approximate absent candidates as zero, and normalize; no supplemental scoring. The separate [offline reanalysis](pivot-topk-zero-fill-2026-09-26.md) calculates all 27 pilot D-choice reads and 17 D-text pairs under this rule, with no new calls. The two old failed supplemental scores remain archived and unused; their discrepancy is not claimed to be fixed. Both D measurements retain explicit truncation metadata.
- A new main-run spending ceiling has not been explicitly agreed numerically. The earlier US$30 authorization covered the pilot, not a fresh US$30 per directory or batch.

## Cost check (dated estimates, not invoices)

The final-version three-question pilot's calls, including imported calls and its rejected scores, cost an estimated US$1.505344433. A simple 20-fold extrapolation is approximately US$30 for 60 questions; it is based on only three questions, includes their already-paid work, and is not an upper bound.

With supplemental scoring removed, the existing deliberately conservative estimator, using the actual 60-question route/sample plan but much larger assumed input/output usage, gives US$165.10 central / US$338.18 high before reserve, or US$214.63 / US$439.64 with 30% reserve. No cache discounts are assumed. These retain the 2026-09-25 price assumptions, not newly verified billing rates. The earlier US$182.56 / US$371.76 figures included supplemental-score allowances that are no longer applicable.

Proposed next step: agree a US$100 estimated-cost stopping ceiling, complete the main-study entry point and bounded scheduling, then launch this mixed-roster run with exact checkpoint reuse and the new D rule. Stop on unresolved validity or accounting errors; do not increase the ceiling automatically or promise completion within it.
