# HLE-Diamond: twenty-question paired turn-tone pilot

Status: **twenty-question pilot finished with recorded failures**,
2026-09-29 06:00 UTC. All twenty question records were reconstructed by audit;
788 requests succeeded, 6 logical requests exhausted retries and 46 dependent
requests were blocked. Coverage: 59/60 initial answers, 331/360 unique replies,
and 109/120 complete debate paths. Token-based cost estimate, including invalid
responses: US$15.300; no unresolved billing reservations. Native probability
positions passed for 113 Qwen and 118 Inkling reads.

The gate finished at 2026-09-29 05:42 UTC: 42/42 successful requests, six
complete paths, no failures, twelve native probability-position checks, and
US$0.499 estimated token cost. The full run resumed the same journal.

Read-only inspection of the saved raw token sequences found 52 length-capped
attempts (including retries), all from Qwen. Every one used all 16,384 output
tokens while the prompt-opened `<think>` block remained unclosed; none generated
`</think>`. Thus these attempts did not reach the final-answer phase. The
incomplete-output parser records their unfinished reasoning as visible content
and reports zero parsed reasoning tokens; that zero does **not** mean reasoning
was disabled. Full generated-token accounting remains intact. This inspection
does not establish whether the reasoning was productive or repetitive. No
generation setting, parser, or frozen source was changed during the run.
The initial automatic review blocked transmission of gated benchmark questions.
After the user requested investigation, a public-source terms review was
submitted with the same launch command for explicit reconsideration. The
escalated execution was allowed; no alternate route was used to bypass the
initial rejection. Dataset access was already authorized and verified.

## Data-use review

- The [HLE-Diamond card](https://huggingface.co/datasets/cais/hle-diamond)
  requests no public sharing, re-uploading or distribution of the dataset and
  excludes it from training corpora. The card links the official HLE repository.
- The [official inference script](https://github.com/centerforaisafety/hle/blob/main/hle_eval/run_model_predictions.py)
  explicitly sends questions to a model API. This supports private API
  evaluation as an intended workflow; it is not permission to publish the bank.
- [Tinker terms, section 3(c)](https://thinkingmachines.ai/legal/terms/)
  state that customer prompts and outputs are not used for model training or
  fine-tuning or shared to improve third-party technologies. This is not a claim
  of zero retention. No account-specific Databricks retention setting was
  verified in this review.
- Only question/options and generated debate material enter the existing
  Databricks/Tinker inference endpoints. Reference answer keys remain local,
  and neither training nor public hosting is part of this run.

## Scope and selection

- `cais/hle-diamond`, revision `04eeb7efa7e3e4f83a00cbd5ce436a38fd5dda23`.
- Exactly twenty of the 75 text-only, reasoning-partition, native multiple-choice
  questions. Sort source IDs and sample once with Python seed 20260928. Selection
  is frozen before inference; no solvability screen, model-outcome filtering, or
  silent redraw. Preserve the original answer key, option text, and option order.
- This draw contains five math, five physics, five computer-science, two biology,
  two other, and one logic/puzzle item. These are realized counts, not quotas.
- Raw benchmark material, answer keys, journals, and full-text HTML are private
  files under the git-ignored `runs/` tree. Do not commit, publish, or re-upload.

## Design

- Existing mixed roster: Terra (Databricks), Qwen3.8-27B and Inkling (Tinker).
- AB, CA, BC dyads, four replies each, with two matched tone continuations.
- Reuse exactly the twenty GlobalOpinionQA tone schedules in
  `docs/turn-tone-dyadic-shared-plan-2026-09-26.json`, mapping sorted selected
  HLE IDs to sorted source template IDs. No opinion questions or prior model
  outputs enter the new run. Reuse the exact friendly/neutral/hostile wording.
- T1/T2 are generated once; T3 and T4 use different private tones in the two
  continuations. Across all unique replies there are 120 per tone. Historical
  tone instructions and self-labels are not transmitted to other members.
- Initial answers are shared across paths; updated private choice/position
  fields do not feed back. Formal output is reply, agreement, choice, position.
- Formal reasoning stays medium. Output ceilings are 16,384 tokens including
  reasoning, to accommodate harder problems. No tools or web access.
- C uses exact option changes and correct/incorrect transitions, grouped by
  previous/current peer-agreement labels, T2–T4 pooled. No B/C text judges.
- D1 uses reasoning-off neutral reads on the two open-weight models, all native
  options normalized and original top-20 omissions assigned zero. Readouts through
  T3 have identical inputs across arms and are reused. Report full-distribution
  TV and change in reference-answer probability; no D2.
- D1 first participation compares question-only input with input that also
  contains the member's own initial solution and public messages. Do not label
  this contrast a peer-only persuasion effect. Shared T3 readouts may appear in
  two different self-label conditions but are not independent observations.
- E uses the same Terra chairman and the same dyad initials, with/without the
  four public replies. One baseline synthesis per pair is reused for both
  continuations. Score exact answer-key correctness, not judge preference.
- Counts: 60 initial + 360 unique replies + 240 unique D1 reads + 180 synthesis
  = **840 planned logical calls**, 120 terminal paths, twenty independent questions.
- Maximum 32 concurrent requests and eight rolling questions. Existing bounded
  retries and failure isolation; successful requests are resumed, never rerun.
- One-question gate first (42 requests); audit native probability positions and
  reconstruction before continuing the same journal to twenty questions.

## Files and reviewed execution commands

Private frozen bank:
`/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-20-20260928-plan/question-bank.json`

Private HTML (final pilot results, including coverage and failures):
`/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-20-20260928/hle-diamond-20-pilot.html`

```sh
.venv/bin/python -m scripts.hle_diamond_pilot --prepare

# One-question gate: reviewed and allowed after the data-use investigation above.
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m scripts.hle_diamond_pilot --live --questions 1
# After the gate passes: same command with --questions 20.
```

The implementation is isolated in `scripts/hle_diamond_study.py` and
`scripts/hle_diamond_pilot.py`. Existing frozen `llm_committee/pivot` modules,
SuperGPQA outputs, and all GlobalOpinionQA outputs remain unchanged.

## Follow-up: isolated Qwen low-reasoning retries (2026-09-29)

The six Qwen requests that exhausted the original medium retries were each
sampled once with `low`, in parallel. Only effort and diagnostic storage keys
changed; prompts/history/tone, schema, sampling temperature and the 16,384-token
ceiling stayed unchanged. No downstream turns, D1 probes or synthesis were run.

- Five of six produced complete, parseable answers in 4.6–41.4 seconds
  (531–9,429 total output tokens). One initial-answer request again reached
  16,384 tokens and remained incomplete, after 70.6 seconds.
- One of the five completed answers matched the reference key. This is a
  failure-selected diagnostic, not overall benchmark accuracy, and no fresh
  medium control was run to identify an effort effect independently of sampling.
- Estimated additional inference cost: US$0.2121. All six native input sequences
  were checked against the low renderer. Original study results remain unchanged.
- A post-inference SDK session-close error (`completed` instead of the accepted
  `success` status) interrupted the first HTML export, not the saved requests.
  The executed source was archived with its original hash, the cleanup call was
  corrected, and the report was reconstructed offline without new inference.

Private diagnostic HTML:
`/Users/Chen/Documents/research/llm-committee/runs/hle-diamond-qwen-low-retry-20260929/hle-qwen-low-retry.html`

Rebuild this diagnostic without model calls:
```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python \
  -m scripts.hle_low_reasoning_retry --report-only
```
