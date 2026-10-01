# D readout update: original top-20, absent candidates as zero

On 2026-09-26, after reviewing the pilot's supplemental-score discrepancy, the user directed:
“不在不碍事，归一化就行，没出现的当0计算。” This explicitly replaces the earlier requirement
to obtain exact scores for every candidate. It applies uniformly to D-choice and D-text,
not only the previously failed event. Debate generation, questions, prompts, route, sample,
reasoning settings, temperature, and the definitions of D's differences are unchanged.

## Rule

At the original output position, exponentiate the returned log probabilities for candidate
letters present in the top-20. Assign probability zero to absent candidate letters, then
normalize over the full predefined candidate list. Do not perform supplemental scoring.
This is a top-20 truncation approximation: absence is not proof of true zero probability.
Top-20 is the reporting limit; sampling remains at temperature 1 with no top-k restriction.

Each distribution records its policy version (`topk_zero_fill_v1`), missing candidate labels,
whether zero-fill was used, observed candidate mass, and original observed log probabilities.
Missing log probabilities are not set to zero (which would imply probability one).
If no candidate probability is available, normalization is undefined and the read remains
an error. Invalid returned numbers, wrong token alignment and unexpected thinking are still
rejected. The legacy exact-score path remains available only for archived-policy replay.

## Offline pilot verification

Source: `runs/mixed-family-pilot/live-format-v4-resumed/`.
New derived artifacts: `runs/mixed-family-pilot/offline-topk-zero-fill-20260926/`.

- Zero new API calls or cost. Dispatch is disabled at the journal and provider levels.
- All 27 D-choice reads and all 17 D-text pairs can be calculated under the new rule.
- Of the 61 native probability reads, only two require zero-fill: archived question 24,
  hostile node n14, argument arm missing E; control arm missing B and E.
- Every previously valid D probability distribution is numerically unchanged. A/B/C/E
  outputs, routes and sampling are unchanged. No response was regenerated.
- Original manifest and logical journal hashes still match. Both old failed supplemental
  records remain invalid in the lineage; their values are not used. This is a documented
  analysis-rule change after the pilot, not a repair of the old scoring interface.

New manifests explicitly contain the policy. An ordinary continuation refuses a policy
change. The offline `reanalyze-topk` command permits this specific migration, rechecks all
saved request hashes, and stops if any required response is absent instead of generating it.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -m llm_committee.pivot reanalyze-topk \
  --source runs/mixed-family-pilot/live-format-v4-resumed \
  --output runs/mixed-family-pilot/offline-topk-zero-fill-20260926
```

The output above already exists; choose a new directory to repeat the check. Local cached
tokenizers are needed to reconstruct the exact filler request lengths; no provider session
is constructed. Old artifacts are not overwritten.

Automated validation: 224 tests passed, 3 optional native-renderer tests skipped in the
project environment; all 3 passed separately with cached native tokenizers and fabricated
transport responses, without model calls. Tests cover zero-fill, stable normalization, unchanged complete reads,
invalid/empty reads, absence of supplemental dispatch, new versus archived budgets, explicit
policy migration, offline replay and preservation of original failures. Lint passed.

The 60-question mixed run now plans 6,896 generation/judging calls before checkpoint reuse,
with zero supplemental-scoring allowance (previously 8,068 including allowances). D-text
still needs its two argument/control generation calls. No main-study run was started here.
