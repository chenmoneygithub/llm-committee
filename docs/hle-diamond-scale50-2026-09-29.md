# HLE-Diamond: 50-question continuation

User decision on 2026-09-29: expand to 50 questions; retain medium reasoning and
retry Qwen length-exhaustion failures with low. This is a medium-first execution
policy, not a uniformly medium-only study or a randomized reasoning-effort study.

## Frozen scope

- Keep all 20 pilot questions and their exact tone-template assignments.
- Uniformly sample 30 of the remaining 55 eligible IDs with seed `20260929`.
  Selection uses IDs only; no performance-based filtering or replacement.
- Dataset remains `cais/hle-diamond`, revision
  `04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23`, text-only reasoning native MCQs.
- Same Terra / Qwen3.8-27B / Inkling roster and original answer options.
- Per question: AB, CA, BC dyads; four debate turns; two continuations sharing
  T1/T2 with changed private tones at T3/T4. Friendly, neutral, and hostile are
  turn-level instructions, not two global-tone arms.
- Reuse the 20 original GOQA schedules and its frozen additional-30 schedules.
  Do not sort all 50 HLE IDs and remap the original 20 schedules.
- C remains deterministic choice/right–wrong transitions by previous/current
  agreement labels. D1 remains neutral reasoning-off option probabilities.
  E remains exact-key scoring of paired Terra syntheses. No B/C text judge or D2.
- Same prompts, output-token limits, native probability readout, and public-history
  context. No tools or web search. No new independent repetitions.

The 50 questions are the sampling units. There are 300 continuation paths, not
300 independent questions. Two paths share a no-debate synthesis baseline.

## Retry and reuse policy

The ordinary initial attempt plus two identical retries remain unchanged. If an
exhausted Qwen initial/debate request ends with an explicitly verified Tinker
length-capped response, retry that exact prompt at low effort, up to three attempts.
Choose the first format-valid response irrespective of its correctness. Do not
change the effort of probability reads or other models; format/transport errors
alone do not authorize a reasoning change.

The original pilot's 788 successful logical requests are imported unchanged with
zero additional inference charge. All 60 failed physical attempts are also retained.
Six exhausted Qwen requests can resume through low fallback; their 46 previously
blocked dependents are generated from the recovered outputs, not copied from a
different trajectory. Existing successful downstream requests are never regenerated.
The separate six-case low diagnostic is not selectively imported.

Planned total: 150 initials + 900 unique debate replies + 600 D1 reads + 450
syntheses = 2,100 logical tasks. After the 788 successful imports, 1,312 remain,
plus any technical retry attempts. Execution uses a global 32-request pool and
eight rolling questions. No new spending cap was imposed.

## Implementation and checks

New entry point: `scripts/hle_diamond_scale.py`; new private report generator:
`scripts/hle_diamond_scale_report.py`. Original pilot code and journals stay unchanged.

The journal explicitly verifies length-failure provenance before creating a separate
low request key. Resume checks exact request hashes. Offline graph reconstruction
verifies the accepted low answer and all dependent prompts. Native audits also
verify low-rendered prompts and D1 token positions.

Focused tests cover preserved schedules, disjoint selection, outcome-independent
selection, unchanged fallback messages, bounded exhaustion, no unauthorized effort
changes, no paid success duplication on resume, downstream reconstruction, answer-key
isolation, and dynamic report sizes. The original pilot and diagnostic tests are
included in the regression suite.

## Private artifacts

These contain protected benchmark content and must not be published or committed.

- Plan: `/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-50-20260929-plan/manifest.json`
- Bank: `/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-50-20260929-plan/question-bank.json`
- Journal: `/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-50-20260929/live/requests.sqlite3`
- Progress: `/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-50-20260929/live/progress.json`
- Report: `/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-50-20260929/hle-diamond-50.html`

Reports identify the medium-first fallback policy and list recovered low requests;
historical pilot cost and newly incurred cost are separate. Unresolved reservations
are not silently treated as zero. Run status and final audit results belong to the
private progress/audit artifacts, not this protocol note.
