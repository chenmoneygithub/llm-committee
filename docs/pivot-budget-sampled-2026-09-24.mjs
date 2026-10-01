// QUOTATION ASSUMPTIONS ONLY: not an approved experiment protocol.
// Superseded by pivot-budget-2026-09-25.mjs for the shared-turn-sampling design.
// Arithmetic only: no network, inference, deployment, or file writes.
// HISTORICAL as of 2026-09-25: the user chose shared turn sampling across all
// 80 questions for B/C/D. This calculator still models whole-question sampling;
// it does NOT estimate the new design, whose event/reference counts are not set.
// Usage: node docs/pivot-budget-sampled-2026-09-24.mjs [questions=80] [measuredQuestions=25] [leaves=6] [sharedPrefix=0]
// User proposed reducing the working question count from 100 to 80. The 25 measured questions
// remain a quotation assumption, not an automatically scaled sample fraction.
import assert from 'node:assert/strict';

// USD / 1M tokens. q27 borrows Qwen3.8-27B prices ONLY as a proxy for
// the selected-but-unverified Qwen3.5-27B endpoint; no model substitution.
const prices = {
  terra: {input: 2, cached: 0.2, output: 12, write: 1.25},
  luna: {input: 0.2, cached: 0.02, output: 1.2, write: 1.25},
  sol: {input: 4, cached: 0.4, output: 20, write: 1.25},
  q27: {input: 1.86, cached: 0.372, output: 5.595, write: 1},
  inkling: {input: 1.87, cached: 0.374, output: 4.68, write: 1},
  gemini: {input: 0.75, cached: 0.075, output: 3.75, write: 1},
  gpt55: {input: 5, cached: 0.5, output: 30, write: 1},
};
const quotationRosters = [
  ['terra', 'terra', 'terra'],
  ['luna', 'terra', 'sol'],
  ['terra', 'q27', 'inkling'],
];
// Token assumptions unchanged from the earlier estimate; outputs include
// billable reasoning, not just visible text. They are NOT generation settings.
const tokens = {
  central: {
    panelIn: 8000, panelOut: 1800, positionIn: 8500, positionOut: 700,
    initialIn: 1500, initialPositionIn: 2500,
    bJudgeIn: 2000, cJudgeIn: 1400, judgeOut: 1000,
    eJudgeIn: 3200, eJudgeOut: 800,
    chairIn: 18000, baselineChairIn: 3500, chairOut: 2500,
    dIn: 8500, dOut: 500, choiceScoreIn: 9200,
  },
  heavy: {
    panelIn: 16000, panelOut: 4500, positionIn: 16500, positionOut: 1800,
    initialIn: 2000, initialPositionIn: 4000,
    bJudgeIn: 3500, cJudgeIn: 2200, judgeOut: 3000,
    eJudgeIn: 5000, eJudgeOut: 2500,
    chairIn: 35000, baselineChairIn: 6000, chairOut: 6000,
    dIn: 16500, dOut: 1500, choiceScoreIn: 18200,
  },
};

function estimate({questions, measuredQuestions, repetitions = 1,
  pathLength = 6, sharedPrefix = 0, leaves = 6, prefixGroups = 1,
  cacheHit = null, tokenScenario = 'central', rosters = quotationRosters,
  judge = 'gemini', ablationNodesPerMeasuredRun = 1,
  includeChoiceFallback = true, qwenPriceMultiplier = 1}) {
  for (const n of [questions, measuredQuestions, repetitions, pathLength, leaves])
    assert(Number.isSafeInteger(n) && n > 0);
  assert(measuredQuestions <= questions);
  assert(Number.isSafeInteger(sharedPrefix) && sharedPrefix >= 0 && sharedPrefix < pathLength);
  assert(Number.isSafeInteger(prefixGroups) && prefixGroups >= 1 && prefixGroups <= leaves);
  assert(cacheHit === null || (cacheHit >= 0 && cacheHit <= 1));
  assert.equal(rosters.length, 3);
  rosters.forEach(r => assert.equal(r.length, 3));
  const s = tokens[tokenScenario];
  assert(s);
  const tones = 3;
  // Cost scenarios, NOT a routing algorithm or a prediction of random forks.
  // No common formal prefix gives the conservative leaves * length node count.
  const uniqueReplies = prefixGroups * sharedPrefix + leaves * (pathLength - sharedPrefix);
  // Chairman reads every unique formal node once. Scale the variable-history
  // portion from the earlier 9-node quote, retaining fixed initial overhead.
  // This is a conservative length assumption, not measured visible text.
  const chairmanInput = s.baselineChairIn
    + (s.chairIn - s.baselineChairIn) * uniqueReplies / 9;
  const runs = questions * rosters.length * tones * repetitions;
  const measuredRuns = measuredQuestions * rosters.length * tones * repetitions;
  const initial = questions * rosters.length * 3 * repetitions;
  const cInitial = measuredQuestions * rosters.length * 3 * repetitions;
  const replies = runs * uniqueReplies;
  const measuredReplies = measuredRuns * uniqueReplies;
  // Upper allowance; deduplicate identical pairs and never create a final
  // position for an inactive member just to fill this allowance.
  const cTerminalPairs = measuredRuns * leaves * 3;
  const cPairs = measuredReplies + cTerminalPairs;
  const ablationReplies = measuredRuns * ablationNodesPerMeasuredRun * 2;
  const dMembers = rosters[2].filter(m => ['q27', 'inkling'].includes(m));
  assert(dMembers.length > 0);
  // Expected counts with symmetric random routing, not per-member quotas.
  const dEvents = measuredQuestions * tones * repetitions * uniqueReplies * dMembers.length / 3;
  const dChoiceFallbackReads = includeChoiceFallback
    ? dEvents + measuredQuestions * repetitions * dMembers.length : 0;
  const eBaselines = questions * rosters.length * repetitions;

  const cost = (model, input, output, reusable = false) => {
    const p = prices[model];
    const factor = model === 'q27' ? qwenPriceMultiplier : 1;
    // cacheHit=null means caching disabled. For enabled-cache scenarios,
    // conservatively charge every missed eligible input token as a write.
    const inputRate = reusable && cacheHit !== null
      ? cacheHit * p.cached + (1 - cacheHit) * p.input * p.write : p.input;
    return factor * (input * inputRate + output * p.output) / 1e6;
  };
  const panel = (input, output, reusable = false) =>
    rosters.flat().reduce((sum, m) => sum + cost(m, input, output, reusable), 0) / 9;
  const dPanel = (input, output) =>
    dMembers.reduce((sum, m) => sum + cost(m, input, output, true), 0) / dMembers.length;
  const components = {
    initial_answers: initial * panel(s.initialIn, s.panelOut),
    formal_debate: replies * panel(s.panelIn, s.panelOut, true),
    C_position_generation: cInitial * panel(s.initialPositionIn, s.positionOut)
      + measuredReplies * panel(s.positionIn, s.positionOut, true),
    B_judging: measuredReplies * cost(judge, s.bJudgeIn, s.judgeOut),
    C_judging: cPairs * cost(judge, s.cJudgeIn, s.judgeOut),
    A_ablation_generation: ablationReplies * panel(s.panelIn, s.panelOut, true),
    A_ablation_judging_allowance: ablationReplies * cost(judge, s.bJudgeIn, s.judgeOut),
    D_choice_scoring_reserve: dChoiceFallbackReads * dPanel(s.choiceScoreIn, 1),
    D_text_argument_control: 2 * dEvents * dPanel(s.dIn, s.dOut),
    E_chairman: runs * cost('terra', chairmanInput, s.chairOut)
      + eBaselines * cost('terra', s.baselineChairIn, s.chairOut),
    E_judging_both_orders: runs * 2 * cost(judge, s.eJudgeIn, s.eJudgeOut),
  };
  const total = Object.values(components).reduce((sum, c) => sum + c, 0);
  const logicalCalls = initial + replies + cInitial + measuredReplies
    + measuredReplies + cPairs + 2 * ablationReplies + dChoiceFallbackReads
    + 2 * dEvents + runs + eBaselines + runs * 2;
  return {
    assumptions: {questions, measuredQuestions, repetitions, tones, pathLength,
      sharedPrefix, prefixGroups, leaves, uniqueReplies, cacheHit, tokenScenario, rosters,
      judge, ablationNodesPerMeasuredRun, includeChoiceFallback, qwenPriceMultiplier,
      chairmanInput},
    counts: {runs, measuredRuns, initial, replies, cInitial, measuredReplies,
      cTerminalPairs, cPairs, ablationReplies, dEvents, dChoiceFallbackReads,
      eBaselines, logicalCalls},
    components, total, with30PercentReserve: total * 1.3,
  };
}

const questions = Number(process.argv[2] ?? 80);
const measuredQuestions = Number(process.argv[3] ?? 25);
const leaves = Number(process.argv[4] ?? 6);
const sharedPrefix = Number(process.argv[5] ?? 0);
const base = {questions, measuredQuestions, leaves, sharedPrefix};
// Preserve the previous quote as a regression check, not as the current shape.
const checkBase = {questions: 50, measuredQuestions: 25, leaves: 2, sharedPrefix: 3};
const check = estimate({...checkBase, cacheHit: 0.5});
assert.equal(check.assumptions.uniqueReplies, 9);
assert.equal(check.counts.runs, 450);
assert.equal(check.counts.replies, 4050);
assert.equal(check.counts.measuredReplies, 2025);
assert.equal(check.counts.cInitial, 225);
assert.equal(check.counts.cPairs, 3375);
assert.equal(check.counts.dEvents, 450);
assert.equal(check.counts.dChoiceFallbackReads, 500);
assert.equal(check.counts.logicalCalls, 15950);
assert(Math.abs(check.total - 264.26318125) < 1e-9);
assert(estimate({...checkBase, cacheHit: 0.8}).total < check.total);
assert(estimate({...checkBase, cacheHit: 0}).total > estimate(checkBase).total);
assert(Math.abs(estimate({...checkBase, questions: 100, measuredQuestions: 50, cacheHit: 0.5}).total - 2 * check.total) < 1e-9);
assert.equal(estimate({...checkBase, cacheHit: 0.5, includeChoiceFallback: false}).components.C_position_generation,
  check.components.C_position_generation);
const six = estimate({questions: 50, measuredQuestions: 25, cacheHit: 0.5});
assert.equal(six.assumptions.uniqueReplies, 36);
assert.equal(six.assumptions.chairmanInput, 61500);
assert.equal(six.counts.replies, 16200);
assert.equal(six.counts.cPairs, 12150);
assert.equal(six.counts.dEvents, 1800);
assert.equal(six.counts.logicalCalls, 53075);
assert.equal(estimate({questions: 50, measuredQuestions: 25, sharedPrefix: 3}).assumptions.uniqueReplies, 21);
assert(estimate({questions: 50, measuredQuestions: 25, cacheHit: 0.5, sharedPrefix: 3}).total < six.total);
const twoStartShape = {pathLength: 5, sharedPrefix: 1, prefixGroups: 2, leaves: 6};
assert.equal(estimate({questions: 50, measuredQuestions: 25, ...twoStartShape}).assumptions.uniqueReplies, 26);

console.log(JSON.stringify({
  status: 'HISTORICAL QUOTATION ONLY. On 2026-09-25 the user confirmed shared turn sampling across all 80 questions for B/C/D, replacing the 25-question full-measurement assumption. This calculator still models whole-question sampling and does NOT price the new design; sampled-event and deduplicated-reference counts remain unset. twoInitialRoutesFiveTurns preserves the two-root, five-turn, six-terminal-path cost shape for historical comparison. No spending approved.',
  asOf: '2026-09-24', prices, tokenAssumptions: tokens,
  cacheScenarios: {
    cachingDisabled: estimate(base),
    enabledButNoHits: estimate({...base, cacheHit: 0}),
    halfEligibleInputCached: estimate({...base, cacheHit: 0.5}),
    eightyPercentEligibleInputCached: estimate({...base, cacheHit: 0.8}),
  },
  highTokenScenarios: {
    halfEligibleInputCached: estimate({...base, tokenScenario: 'heavy', cacheHit: 0.5}),
    enabledButNoHits: estimate({...base, tokenScenario: 'heavy', cacheHit: 0}),
  },
  twoInitialRoutesFiveTurns: {
    // Two distinct first replies, then as many as six separate suffixes of
    // four replies. An upper count conditional on no extra root routes later.
    halfEligibleInputCached: estimate({...base, ...twoStartShape, cacheHit: 0.5}),
    cachingDisabled: estimate({...base, ...twoStartShape}),
    enabledButNoHits: estimate({...base, ...twoStartShape, cacheHit: 0}),
    highTokensHalfCached: estimate({...base, ...twoStartShape, cacheHit: 0.5, tokenScenario: 'heavy'}),
  },
  sixTrajectoryForkScenarios: Object.fromEntries([3, 1, 0].map(prefix => [
    `sharedFormalPrefix${prefix}`,
    {
      halfEligibleInputCached: estimate({...base, leaves: 6, sharedPrefix: prefix, cacheHit: 0.5}),
      cachingDisabled: estimate({...base, leaves: 6, sharedPrefix: prefix}),
      enabledButNoHits: estimate({...base, leaves: 6, sharedPrefix: prefix, cacheHit: 0}),
    },
  ])),
  sensitivity: {
    inlineChoiceLogprobsComplete: estimate({...base, cacheHit: 0.5, includeChoiceFallback: false}),
    qwenHalfProxy: estimate({...base, cacheHit: 0.5, qwenPriceMultiplier: 0.5}).total,
    qwenDoubleProxy: estimate({...base, cacheHit: 0.5, qwenPriceMultiplier: 2}).total,
    // Illustrative reuse of the old GPT-5.5 member, not a new model decision.
    gpt55InSameModelAndMixed: estimate({...base, cacheHit: 0.5, rosters: [
      ['gpt55', 'gpt55', 'gpt55'], ['luna', 'terra', 'sol'], ['gpt55', 'q27', 'inkling'],
    ]}),
    twiceThePathLength: estimate({...base, cacheHit: 0.5, pathLength: 12}),
    twiceTheQuestionsSameMeasuredSample: estimate({...base, cacheHit: 0.5, questions: 2 * questions}),
  },
  pilotThreeQuestionsAllMeasured: estimate({...base, questions: 3, measuredQuestions: 3, cacheHit: 0.5}),
}, null, 2));
