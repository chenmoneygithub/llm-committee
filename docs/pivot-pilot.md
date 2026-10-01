# Mixed-family pilot: verify Layers A–E

**2026-09-26 update:** this guide's supplemental-scoring rules describe the historical pilot.
New runs use the user-approved top-20/zero-fill/normalize rule, with no supplemental scoring.
The saved pilot has been reanalyzed offline under that rule; see [current D policy and verification](pivot-topk-zero-fill-2026-09-26.md).
The main question bank is now the approved 60-question bank, not the earlier planned 80.

Updated 2026-09-25 after the user clarified that pilot questions may belong to the main 80
and that the pilot should exercise every layer. This supersedes the earlier same-family-only
pilot and mandatory question holdout. The first live attempt stopped; the user has since
approved [format fixes and continuation](pivot-pilot-format-v3.md). V3 stopped on an
incomplete C response; v4 subsequently finished with one incomplete D-text pair.

## Execution checkpoint (2026-09-25)

**Finished with measurement failures; no process is running.** All three questions and
nine debates completed: 168 formal replies, 40 C reads, 24 B and 32 C-text judgments,
27 complete D-choice distributions, 12 syntheses and 18 E judgments. D-text has **16/17**
complete pairs; the two failed supplemental scores belong to the same Inkling event.
Their metric remains null and the numerical discrepancy is unresolved. No threshold,
sample, prompt or model was changed to accept them.

Final artifacts: `runs/mixed-family-pilot/live-format-v4-resumed/`. Cumulative estimated
cost across ALL attempts is **US$1.557001068**, below the US$30 authorization. Two Gemini
usage-accounting issues were fixed by reparsing saved responses offline, without another
generation. Exact-request replay reproduced the final report with zero new calls or
journal changes. Tests: 201 passed / 3 optional tests skipped, with those native-renderer
tests passed separately. See the [final engineering report](pivot-pilot-live-report.md)
for coverage and the linked continuation record for development history. This is not a
full D acceptance, judge-accuracy validation or main-study inference.

The user approved starting this mixed-family pilot with the single Gemini 3.8 Flash
judge and a US$30 cap. This covers only the three questions and nine debates below,
including core A–E and both D readouts; it does not authorize the main study, additional
A ablations, human audits, deployment, manuscript edits, commit or push.

Correction to the first preflight: missing direct OpenAI/Gemini keys were NOT a credential
blocker. The repository already uses Databricks OAuth profile `un`. Read-only endpoint
inspection confirmed `databricks-gpt-5-6-terra` and `databricks-gemini-3-8-flash` are READY
and identify the exact intended foundation models. The new runner now uses that existing
route, retaining Tinker for native Qwen/Inkling generation and D scoring. No new keys,
endpoint deployments or model substitutions are needed.

The first live attempt in `runs/mixed-family-pilot/live/` stopped after 11 generations
(10 valid, 1 malformed Inkling reply), estimated US$0.042922555. All three initial answers
and C reads succeeded, including complete Qwen/Inkling choice probabilities. The first
friendly debate stopped on an Inkling reply lacking the required JSON fields; no retry or
label inference was attempted. B/C-text, D-text and E were untested at that checkpoint. See the
[first-run report](pivot-pilot-first-run.md), not a full A–E acceptance or research result.
Request/response payloads are saved without authorization headers. Databricks token-cost
estimates include a conservative 10% regional-processing allowance; they are not invoices.

Runtime dependencies are installed in the isolated Python environment
`/tmp/committee-tinker-check.sqHb9X/venv/`, including OpenAI 2.54.0, Tinker 0.30.3,
Cookbook 0.5.7, tml-renderers 0.1.0 and Databricks SDK 0.133.0. The project's existing
`.venv` is unchanged. After the post-run cleanup fix: 177 passed / 3 optional native tests skipped; those
3 native tests passed separately alongside 14 new Databricks tests, using fabricated responses.

## Scope

- Start with three screened GlobalOpinionQA questions from the intended main-study bank,
  not an automatically excluded pilot-only bank. The final 80-question bank is not yet selected.
- Committee: GPT-5.6 Terra, Qwen3.8-27B and full TML Inkling. Terra remains chairman.
- Run friendly, neutral and hostile once per question: nine debates, no independent repetitions.
  Generate each member's initial answer once per question and reuse it across tones and the
  no-debate synthesis. A valid pilot run is the first part of the study, not an extra replicate.
- Retain the agreed routing: two initial sender/receiver pairs, four planned forks, six terminal
  paths of five formal replies. Shared prefixes are generated once; sibling state never mixes.
- Retain one shared sample of eight target events per question across all three tones. B/C/D
  use that sample; D applies only to Qwen/Inkling. Add and deduplicate true previous-own and
  initial C references. Do not resample events to obtain larger changes or more disagreement.
- Formal debate has reasoning enabled; C and both D-text arms use the agreed direct-read mode.
  Endpoint behavior still needs verification, particularly Inkling's no-new-thinking behavior.
- A single Gemini 3.8 Flash B/C/E judge is approved for this pilot, not yet live-validated.
- The revised Databricks/Tinker pilot estimate is $11.52 central / $23.42 high, including 30% reserve,
  no cache savings, and all D-choice/D-text supplemental-scoring allowances. If Inkling's
  temporary discount ends, those estimates become $15.04 / $30.53. Approved spending cap:
  US$30. These are estimates,
  not an invoice or a guarantee all calls finish under the cap. See [budget](pivot-pilot-budget.md).

## What must work

| Layer | Engineering acceptance |
|---|---|
| A | Formal replies preserve all four self-reported agreement labels and the exact peer being judged. |
| B | A blind judge reads peer/reply text without A labels or tone/model metadata; it returns a label and evidence. |
| C | Separate reads return the native option and full own position; compare true previous participation and initial references without feeding readings back into debate. |
| D-choice | At the same C option-output position, obtain all native-option probabilities for Qwen/Inkling; preserve both adjacent and initial comparisons. |
| D-text | Fix the complete prior own position and pre-peer context; compare actual peer argument against length-matched filler without generating or leaking the receiver's subsequent reply. |
| E | Terra synthesizes no-debate/debate from the same initial answers; blind preference judging uses both answer orders. |

D-text requires two extra probability reads per applicable sampled event. Either D-text arm,
and each D-choice read, may need one additional exact-prefix scoring call if top-20 omits any
candidate. All candidates are then rescored together; missing scores are never zero-filled.
The API's one required extra output token is billed, ignored, and never treated as debate data.
It is not safe to budget D as zero cost.
No top-two margin, entropy matching, new statements or separate D debates are introduced.

Check coverage of both open models from the preplanned sample. If an interface has no sampled
event, report it as untested; any extra engineering diagnostic must be separate from the research
sample, not silently added to it. This pilot verifies plumbing and readable outputs, not statistical
power, judge accuracy or general research conclusions. The additional A instruction-removal
ablation and human audit remain separate, not-yet-implemented work before the full study.

## Reusing questions and results

Pilot participation does not disqualify a question. Real records may enter the main results only
if they match the final protocol, prompts, model/settings and applicable quality checks. If debugging
changes a stage, retain old records as development history and rerun the affected measurements
and downstream dependencies under the final version. Do not count old and new runs as independent
replicates, and do not choose which to keep from the observed effect. Document developmental use
of the questions. Synthetic mock responses are never research data.

The current prepared list is `runs/mixed-family-pilot/questions.json`: archive indices **24, 22, 48**.
It preserves the gender/education statement and replaces the ambiguous Church/country and
second-most-extremism items with the greatest-worldwide-threat and free-expression-in-democracy
items. All retain original text, options and archived majority screening. The archive gives the
Church item country selections for Russia and Ukraine; the extremism item spans seven countries/
territories. Neither has a unique country to restore. No country/first choice was invented.
The old candidates and synthetic runs are preserved.

This is explicit, purposive engineering selection for standalone context and both unordered and
ordinal choices, NOT a uniform sample. It used no debate outcomes. The three may be included in
the future 80, with that developmental selection disclosed. Input-review reasons and exact
fingerprints are in the manifest; live execution rejects missing/incomplete input review.
This input review is not a new LLM screen or fabricated human/model annotation.

## Implemented versus still missing

Implemented: deterministic routing/shared sampling, isolated contexts, C side reads, B/C/E,
native Tinker adapters for the exact Qwen/Inkling models, D-choice and D-text, candidate scoring,
request journaling, cost estimates/guards and resumption. Every supplemental scoring call gets
its own budget reservation and journal entry. Rendered prompt and sampled token IDs are saved;
the actual choice position is located without reconstructing its prefix. Scoring checks agreement
with available original logprobs (0.01 natural-log-unit engineering tolerance).

Formal generation uses medium reasoning by default. Native Tinker calls use temperature 1,
top-p 1 and no top-k sampling cutoff, keeping the probability read at T=1 without a separate
rescaling. Top-20 reporting is not a top-k sampling restriction. Direct-mode failures retain raw
responses and stop. Tinker reasoning counts are retokenized diagnostics, not provider-reported
partitions; billing uses full sampled token counts. Public debate text excludes private reasoning.
Planned baseline C reads run before debate so direct-mode failures are caught early.

Offline validation does not establish live account access, endpoint logprob semantics, successful
non-thinking outputs, judge reliability, latency or provider invoices. Local native-renderer tests
use real SDK/tokenizers with fabricated responses and no service session. Full pilot acceptance
requires real evidence from all five layers; mocks are explicitly labeled synthetic.

Install optional live dependencies in a Python 3.11+ environment with `pip install -e '.[pilot,pilot-tinker]'`.
The native-renderer checks were run in an isolated temporary environment; the existing project
environment was not upgraded. Default closed-model transport is `--closed-provider databricks`
with `--databricks-profile un`, using existing SDK OAuth resolution and the workspace-ID
header required by this workspace. Qwen/Inkling use `TINKER_API_KEY`. The optional
`--closed-provider direct` path uses `OPENAI_API_KEY` and `GEMINI_API_KEY`; those keys are
not required for the Databricks path. Do not paste credentials into prompts/files/logs.
The Databricks adapter rejects unsupported explicit cache breakpoints rather than dropping
them; any provider-managed cache usage is recorded. Direct C reads require explicit zero
reasoning-token usage and no thinking content; missing evidence is not treated as zero.

```sh
.venv/bin/python -m llm_committee.pivot plan \
  --roster mixed_family \
  --questions runs/mixed-family-pilot/questions.json \
  --judge-model gemini-3.8-flash
```

This reads the current candidates for planning only, without API calls or credentials. Replace
`plan` with `estimate` to inspect the dated per-model cost assumptions and accounting.

To exercise all layers with explicitly synthetic data only:

```sh
.venv/bin/python -m llm_committee.pivot dry-run \
  --roster mixed_family \
  --questions runs/mixed-family-pilot/questions.json \
  --judge-model gemini-3.8-flash \
  --output runs/mixed-family-pilot/dry-run-databricks

.venv/bin/python -m pytest -o addopts='' -q
```

The original synthetic run at `runs/same-family-pilot/dry-run/` is preserved. Code/config hashes
prevent resuming a changed implementation in its old directory; do not erase or rewrite old
manifests to bypass that check. New imports use `question_fingerprints` and the explicit reuse
policy; legacy fingerprint fields remain readable without imposing a future-study exclusion.

The command used for this first attempt was:

```sh
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 /tmp/committee-tinker-check.sqHb9X/venv/bin/python \
  -m llm_committee.pivot run \
  --questions runs/mixed-family-pilot/questions.json \
  --roster mixed_family --judge-model gemini-3.8-flash \
  --closed-provider databricks --databricks-profile un \
  --output runs/mixed-family-pilot/live --approve-spend-usd 30
```

The attempt is stopped, not running. The code snapshot is archived in that directory;
post-run cleanup fixes changed the implementation hash. Do not reuse or overwrite the
old directory to bypass identity checks. No subsequent paid attempt is running.

Databricks generation uses HTTP without redirects or automatic retries; OAuth refresh is
handled by the existing SDK profile. Configurable SDK retries and application-level resampling are disabled;
Tinker 0.30.3 still has SDK-internal transport/backpressure handling for the same request ID.
Timeouts retain the full budget reservation and block replay; they are not treated as free failures.
Native prefix-cache hits are provider-managed and recorded, even when OpenAI explicit caching
is disabled. Exact software versions, prompts, token prefixes and settings are archived for reuse.

Official sources used for implementation and pricing: OpenAI Docs [Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md);
Databricks [API](https://docs.databricks.com/aws/en/machine-learning/foundation-model-apis/api-reference),
[reasoning](https://docs.databricks.com/aws/en/machine-learning/model-serving/query-reason-models),
[listed rates](https://www.databricks.com/product/pricing/proprietary-foundation-model-serving);
Tinker [sampling/scoring](https://tinker-docs.thinkingmachines.ai/tinker/api-reference/samplingclient/index.md),
[Inkling thinking](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/thinking-effort/index.md),
[rendering](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/index.md),
and [prices](https://tinker-docs.thinkingmachines.ai/tinker/models.json).

The old committee API remains available for historical reproduction. Its cross-branch shared
member state is not used by the new engine. No manuscript, deployment, commit or push is part
of this pilot-plan update. See the [full protocol](pivot-protocol-2026-09-25.md).
