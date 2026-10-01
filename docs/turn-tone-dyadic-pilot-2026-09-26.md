# Turn-level tone pilot: 20 questions, dyadic only

User authorization: move global-tone experiments to supplementary reporting,
create a new report for the next protocol, and first run only the two-member
experiment on 20 questions so the user can inspect results before deciding on
three-member expansion. Do not launch a three-member rerun in this round.

## Frozen design

- Draw 20 of the approved 60 questions with seed 20260927, without reading
  observed labels, effects or earlier success patterns. Retain IDs and source hashes.
- Keep A = GPT-5.6 Terra, B = Qwen3.8-27B, C = Inkling, and directed AB/CA/BC
  pairs. Each pair alternates for four formal replies: 60 trajectories, 240
  new debate replies. No separate global-friendly/neutral/hostile arms.
- Reuse the selected questions' independent tone-free initial answers only.
  Generate all new discussion and measurements; no old debate output is imported.
- Preassign tone per node, balanced within pair × T across questions (6 or 7
  per tone in each block; 80 friendly, 80 neutral, 80 hostile replies total).
  Assignments do not depend on preceding messages or labels. Sequences are
  not forced to switch at each turn, nor screened based on generated outcomes.
- Each debate request receives only its own current tone instruction. Prior
  public text, speaker IDs and reply relationships remain; private historical
  tone instructions, tone metadata and self-labels are not transmitted.
- Neutral C/D probes throughout, including initial reads: BASE plus the same
  measurement question, without friendly or hostile instructions. This was
  disclosed before implementation as the working choice for the new pilot.
  Historical texts can still reflect their authors' tones. Debate reasoning
  remains on; C/D direct reads disable reasoning.
- No Layer E synthesis/quality experiment, extra model roster or human annotation.

## Sampling, measurement and interpretation

For each question, preselect eight of nine pair × T candidates with T=2/3/4,
omitting one balanced node across questions. B/C/D share these 160 events.
T=1 remains in A but has no preceding reply's self-label. C/D retain both the
preceding peer's self-label and the receiver's current self-label, T, and model
where applicable. Also retain both assigned tones as independent metadata;
these are not agreement labels. Never pool all C/D effects into a headline.

C uses original-option change plus a single Gemini 3.8 Flash judgment of full
text change, relative to previous own participation and separately the initial
position at sampled own endpoints. D reuses open-model choice probabilities
and reads endorsement of the full prior position with incoming peer text versus
length-matched filler, before the receiver's new formal reply. Existing top-20
zero-fill approximation at read T=1 remains; no margin or entropy adjustment.

Twenty questions define the independent question sample, not 160 independent
questions. This is a development pilot, not a definitive stratified effect test.
The old fixed-tone experiments also used tone-bearing probes, so any old/new
difference cannot be attributed solely to turn-level versus global tone.

## Execution and artifacts

Use the existing global 64-request, rolling eight-question scheduler and the
same bounded retry policy. Stop only the affected dependency chain after
exhausted recoverable failures; record failed attempts and missing measurements.
No imposed spending cap, consistent with the user's standing instruction.
Offline checks precede a one-question real gate; resume the same frozen journal
for all 20 questions after verification. No outcome-based gate or selection.

New output: `runs/turn-tone-dyadic-20-20260926/live`

New main HTML:
`/Users/Chen/Documents/research/llm-committee/docs/turn-tone-dyadic-pilot-2026-09-26.html`

Separate supplementary HTML for the archived fixed-tone experiments:
`/Users/Chen/Documents/research/llm-committee/docs/fixed-tone-supplementary-results-2026-09-26.html`

The old canonical HTML and all archived raw runs remain unchanged. The new
report links to the supplementary archive rather than mixing its tables into
the pilot's main results. This updates the earlier one-HTML convention for the
new protocol, as explicitly requested by the user.

Status: completed. All 20 questions, 60 trajectories and 240 formal replies
succeeded; all 160 sampled B/C/D events are recorded. Three-member expansion
has not been started. Both HTML reports are generated and verified.

Follow-up: the user subsequently approved a matched continuation after T2.
That separate run adds 120 T3/T4 replies while reusing the exact shared prefix,
bringing the combined pilot to 360 unique replies. Its paired tables and full
side-by-side cases are appended to the same main HTML; the original tables
and this frozen protocol remain intact. See `turn-tone-fork-pilot-2026-09-26.md`.
The user also confirmed reusing this paired trajectory design for the remaining
same-model and same-family configurations. The exact roster-independent design
is saved in `turn-tone-dyadic-shared-plan-2026-09-26.json`; only the design is
reused across configurations, never generated text or measurements. Those two
configurations have not been started.

### Validation and execution log

- Offline full mock: all 20 questions / 60 trajectories / 1,161 requests
  completed. Execution/recovery tests: 53 passed before the real gate.
- Real gate: 1 question, 3 successful trajectories, 56 successful requests;
  no retries or failures. Process time 49.39 seconds; token-estimated charge
  US$0.174800404 (not a provider invoice).
- Read-only gate audit reconstructed all 56 requests and the completed
  question report exactly, verified current-tone-only debate instructions and
  neutral probes, and confirmed zero reasoning tokens for all 23 direct reads.
  All 60 source question records and the frozen implementation hashes match.
- Old canonical report checksums before supplementary export:
  HTML `36cc79eb1df2189fdb7b626080a10d78be33e89b69fe65499320089375df1206`;
  JSON `59ce44c55ea71f0eeb27fdf3826098aaa33118ff678b3a86e91a905b5533fcf5`.
- Full execution resumed the same journal without duplicating the gate.
  The resumed process took 122.37 seconds and reached 64 concurrent requests.
  All 1,161 logical requests succeeded: 240 debate, 280 position, 160 B-judge,
  267 C-judge and 214 D-text requests. Two additional attempts recovered one
  transport failure and one invalid-format response; none exhausted retries.
- Received responses cost an estimated US$3.711557995 by recorded usage,
  including the invalid-format attempt. The original transport-failed attempt
  has unknown billing and retains a conservative US$5.09476 reservation.
  The journal's US$8.806317995 total includes that reservation; it is **not**
  confirmed spending. HTML reports these separately. Provider invoices may differ.
- Final read-only audit reconstructed all 1,161 successful requests and all
  20 question reports exactly. All 494 direct reads had zero reasoning tokens;
  all 400 probability readings used the original top-k data at read T=1.
  Verified current-tone-only instructions, neutral probes, unchanged frozen
  execution code and all 60 unchanged source question files.
- Regression suite: 366 passed, 3 optional native-renderer tests skipped.
  Focused execution/report tests: 21 passed. HTML validation checked all table
  widths, local links, 20 question blocks, 60 trajectory blocks, 160 sampled
  events, and both label dimensions plus T in every C/D table (model also in D).
- Confirmed the old canonical HTML/JSON checksums still match after export;
  every supplementary table and all pre-existing JSON fields match the archive
  source. No old scientific data was pooled into the new pilot.
