# Mixed pilot budget, 2026-09-25

Prepared questions: archive 24 (gender/education), 22 (worldwide threat), 48 (free expression).
Terra / Qwen3.8-27B / full Inkling; Terra chairman; approved Gemini 3.8 Flash judge.
Three questions, three tones, no independent repetitions. **First live attempt stopped.**
The user approved starting this pilot on 2026-09-25. Existing Databricks OAuth provides
Terra/Gemini access; the earlier claim of missing credentials was incorrect. Open models
remain on Tinker. Eleven generations were dispatched before an invalid-format Inkling
reply stopped execution; estimated cost US$0.042922555. No retries. Current usage is
recorded in `runs/mixed-family-pilot/live/requests.sqlite3`; see the [checkpoint](pivot-pilot-first-run.md).

| Work | Count |
|---|---:|
| Initial answers | 9 |
| Formal debate replies | 168 across nine debates |
| Shared sampled events | 24 |
| C readings, including references | 40 |
| B / C text judgments | 24 / 32 |
| D-choice readings (part of C) | 27 |
| D-text argument/control pairs | 17 pairs = 34 reads |
| Chairman syntheses / E judgments | 12 / 18 |
| Core logical requests | 337 |
| Possible supplemental score calls | 27 for D-choice + 34 for D-text |
| Requests with all scoring allowances used | 398 |

These are planned calls, not independent observations. Native probability reads apply only to
the sampled Qwen/Inkling events. No extra dataset, new debate or repeated generation is used for D.

| Scenario (no cache discounts assumed) | API estimate | With 30% reserve |
|---|---:|---:|
| Central token assumptions, current Inkling discount | $8.86 | $11.52 |
| High token assumptions, current Inkling discount | $18.01 | $23.42 |
| Central, Inkling regular price | $11.57 | $15.04 |
| High, Inkling regular price | $23.49 | $30.53 |

These revise the direct-API estimate by conservatively adding 10% to Terra/Gemini costs
for Databricks regional processing, using the standard published DBU rates at an assumed
US$0.07/DBU. This is not the account invoice and does not assume discounts. The regular-price
high scenario exceeds the approved cap; no cap increase or automatic continuation is authorized.

**Approved cap: US$30 for this pilot only.** The engine stops if the next conservative
per-call reservation cannot fit. This can occur before the nominal cap; it does not guarantee
the entire pilot finishes. These estimates are not provider invoices, tax-inclusive quotes,
authorization, or protection against arbitrary future pricing changes.

The estimator uses the earlier budget's deliberately conservative input assumptions (8K central /
16K high per debate reply). Formal output is 1,800 central / the configured 4,096-token ceiling
high, including reasoning. C is 700 / 1,200 output tokens. D-text is 16 / 128 including native
framing, NOT an instruction to produce explanation/thinking. Each potential supplemental scoring
call has one billed but ignored generated token. High estimates use configured output ceilings;
input lengths remain assumptions, not measured counts. Actual tokens/cache hits are journaled.

Full arithmetic, per-stage/per-model breakdown and dated rates:

```sh
.venv/bin/python -m llm_committee.pivot estimate \
  --questions runs/mixed-family-pilot/questions.json \
  --roster mixed_family --judge-model gemini-3.8-flash
```

Sources: [OpenAI Docs Terra](https://developers.openai.com/api/docs/models/gpt-5.6-terra.md),
[Tinker native rates](https://tinker-docs.thinkingmachines.ai/tinker/models.json), and the existing
[Gemini price reference](https://ai.google.dev/gemini-api/docs/pricing), with
[Databricks standard rates](https://www.databricks.com/product/pricing/proprietary-foundation-model-serving).
Inkling full 64K uses the
native sampler, not a cheaper quantized/serverless or small-model substitute. Unconfirmed cache
savings, additional A ablations, human labor, failures/retries, taxes and main-study calls are excluded.
