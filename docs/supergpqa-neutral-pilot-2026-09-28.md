# SuperGPQA: small closed-book neutral pilot

The user authorized a quick 20-question SuperGPQA pilot with one tone choice.
This study is separate from the completed GlobalOpinionQA runs. None of their
prompts, results, journals, or reports are modified.

## Frozen scope

- Existing mixed roster: A = GPT-5.6 Terra, B = Qwen3.8-27B, C = Inkling.
- Twenty questions; five each from mathematics, physics, computer science,
  and selected engineering fields. This is a quality-screened diagnostic subset,
  not a representative estimate of full SuperGPQA performance.
- AB, CA, BC: three independent four-reply dyads per question. One neutral
  instruction throughout; no paired tone continuations, no independent repeats.
- Sixty initial answers, reused across dyads. Each member retains its own
  initial answer and sees only the public contributions on its current path.
  Updated private choices/positions are recorded but never injected into later
  debate or D1 contexts. Historical labels and tone instructions are not shared.
- Formal output remains reply, agreement, choice, position. The four
  strongly/leaning agreement definitions are unchanged in meaning.
- Prompts explicitly solve an academic question, not express a survey opinion.
  A listed answer must be selected. No evaluated member has tools or browsing.
- Formal generation and synthesis use medium reasoning, with an 8,192-token
  output ceiling including reasoning (increased from the opinion study's 4,096
  ceiling to allow mathematical reasoning). This is a ceiling, not a length goal.
- A and option-level C are observed from the same formal outputs. C also records
  wrong-to-right, right-to-wrong, and wrong-to-wrong transitions against the
  frozen dataset answer. Conditional C/D1 tables use the previous/current peer
  labels for T2–T4; T1 has no prior debate label.
- D1 uses the existing native Tinker probability reader on both open-weight
  models, with reasoning off. It replays the corresponding formal input, not its
  newly generated answer. All options are retained, with original top-20 missing
  candidates assigned zero and normalized. Report total variation and change in
  probability of the reference answer; keep complete distributions and prompts.
- E uses the same Terra chairman with each dyad's same initial texts, either
  alone or with all four public replies. Both arms return a choice plus solution.
  Score exact correctness, not LLM preference. No extra judge, text-C, D2,
  human annotation, damage battery, or length-preference experiment in this pilot.
- Planned new logical requests: 60 initial + 240 debate + 200 D1 + 120 synthesis
  = 620. Maximum 32 concurrent requests, eight rolling questions; no new spend cap.
- Existing bounded retry policy: at most two additional identical attempts for
  recognized format/transport/output-limit failures. Preserve every attempt and
  isolate failed tasks/dependents. Missing answers do not count as incorrect ones.

## Question selection

Pinned dataset: `m-a-p/SuperGPQA`, revision
`4430d4458112c7d4497fdcf94d7cc223313d6acf`.
Original JSONL SHA-256:
`28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910`.

The full 26,529-row file was shuffled once using seed 20260928. Within each
declared domain, inspect candidates in that order and take the first five that
pass the recorded assistant screen. Calculation-tagged questions are used for
mathematics, physics, and engineering; the CS domain includes non-calculation
questions. The full 59 reviewed candidates, including 39 exclusions and reasons,
are saved. These numbers are not an estimate of the whole dataset's error rate.
No committee answer, disagreement, or effect was used for selection.

Checks cover missing figures/parameters, malformed text, equivalent correct
options, and whether a brief independent derivation supports the original key.
This is assistant screening, not independent expert or human validation.
Original question text, options, and answer keys are never rewritten.
The selected set contains multiple difficulty levels; the pilot must establish
whether this subset leaves useful error/disagreement room for our exact roster.

Frozen selection and all original reviewed rows:
`/Users/Chen/Documents/research/llm-committee/runs/supergpqa-neutral-20-20260928-plan/question-bank.json`

## Engineering preflight

Version 1 ran one question. Its 12 formal replies and six syntheses completed,
but Inkling's initial D1 returned a derivation instead of a letter on all three
allowed attempts. Those outputs were marked incomplete and never treated as
option readouts. One malformed Qwen JSON response recovered on retry. Received
response cost was approximately US$0.148676.

The general instruction had requested an explanation, conflicting with the
one-letter readout task. Version 2 removes that unconditional explanation
instruction, explicitly follows the current task's requested format, and
clarifies that the D1 letter must be the only visible output. To avoid mixing
prompt versions, v1 is an engineering reference only; v2 starts fresh and then
resumes its own one-question gate to twenty. No old successful call is silently
overwritten or billed twice in the v2 journal.

V1 journal (retained):
`/Users/Chen/Documents/research/llm-committee/runs/supergpqa-neutral-20-20260928/live`

V1 reference report:
`/Users/Chen/Documents/research/llm-committee/docs/supergpqa-neutral-20-preflight-v1-reference.html`

V2 pilot journal:
`/Users/Chen/Documents/research/llm-committee/runs/supergpqa-neutral-20-20260928/live-v2`

## Verification and interpretation

New tests check graph counts, branch isolation, no private-position feedback,
answer-key-independent request construction, exact resume, denominator handling,
and HTML table structure. A live audit reconstructs saved requests and question
records, and independently verifies the native first visible option token,
reasoning-off settings, candidate IDs, and captured probability positions.

Twenty questions are the independent experimental units, not 240 replies or
60 paths. The chairman comparison spends more computation in the debate arm;
it does not distinguish interaction benefits from additional independent work.
Reference correctness and probability of that reference answer are different
observations, and the reasoning-off distribution need not match one formal
reasoning-on sample. No inference about full-benchmark saturation or a general
debate benefit should be drawn from this convenience pilot alone.

Main report:
`/Users/Chen/Documents/research/llm-committee/docs/supergpqa-neutral-20-pilot-2026-09-28.html`

## Completed pilot: September 28, 2026 (Pacific)

All twenty questions were processed. Fifty-eight of sixty paths completed:
59 initial answers, 232 debate replies, 194 D1 reads, and 116 syntheses (601
successful logical requests). Four requests recovered after format retries;
one Inkling initial answer exhausted its two retries, blocking eighteen
dependent requests and the two paths containing that member for that question.
The other pair still completed. All eight invalid attempts remain in the
journal; missing outputs are not scored as incorrect answers. No extra recovery
run or replacement question was added.

The live audit reconstructed all twenty question records and all 601 successful
requests, and independently verified all 194 native D1 token positions.
Received-response token-cost estimate: US$3.878602 for the scientific v2 pilot,
including retries. Excluded v1 engineering preflight: US$0.148676. Combined
estimate: US$4.027278, not a provider invoice.

Initial correctness was Terra 19/20, Qwen 20/20, and Inkling 19/19. All nineteen
questions with three available initials had identical choices. The remaining
question had a Terra/Qwen disagreement and an unavailable Inkling initial.
Qwen identified Terra's substitution of 2009 where the factorization requires
2007; Terra then corrected J to the keyed answer I (minimum sum 232). This was
the only observed option correction. No right-to-wrong changes were observed.

Both chairman arms were correct for every available paired case: AB 20/20,
CA 19/19, BC 19/19. Neutral replies were strongly agree for 231/232 turns and
strongly disagree for the one Qwen correction. This curated subset is too easy
for these models to offer useful error/disagreement coverage. It is not evidence
that SuperGPQA as a whole is saturated, nor that debate cannot improve answers.
Do not silently replace these items with questions selected using model failures.

D1 has an important interpretation caveat: a member's first participation is
compared with a question-only readout, while the later input contains its own
reasoning-on initial answer and public messages. The large first-entry jump
cannot isolate peer persuasion. The report therefore retains the label-pair
breakdown and adds a first-versus-later comparison diagnostic. Among
strongly-agree/strongly-agree events, mean TV is 81.49% at first participation
versus 0.71% later for Qwen, and 75.24% versus 0.00020% for Inkling. These
post-run descriptive checks use existing reads only; no measurement was rerun.

```sh
HF_HUB_OFFLINE=1 HF_HUB_DISABLE_IMPLICIT_TOKEN=1 \
  /tmp/committee-tinker-check.sqHb9X/venv/bin/python -u \
  -m scripts.supergpqa_pilot --live --questions 1
# After the gate, resume the same v2 journal with --questions 20.
```
