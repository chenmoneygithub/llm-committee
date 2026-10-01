// HISTORICAL six-roster, full-measurement estimate; not the current design.
// The three-roster sampled design has no approved node budget or sample sizes.
// Arithmetic only: no network, model calls, deployment, or file writes.
// Usage: node docs/pivot-budget-estimate-2026-09-24.mjs [questions=100] [repetitions=2]
// USD per million tokens, checked 2026-09-24. q27/q35 are explicit PRICE PROXIES,
// not verified offers for the selected checkpoints and not model substitutions.
import assert from 'node:assert/strict';

const rates = {
  terra: [2, 12], luna: [0.2, 1.2], sol: [4, 20], opus: [5, 25],
  q9: [0.66, 1.995], q27: [1.86, 5.595], q35: [0.54, 1.335],
  inkling: [1.87, 4.68], sonnet: [2, 10], gemini: [0.75, 3.75],
};
const rosters = [
  ['terra', 'terra', 'terra'], ['q27', 'q27', 'q27'],
  ['luna', 'terra', 'sol'], ['q9', 'q27', 'q35'],
  ['terra', 'q27', 'inkling'], ['terra', 'opus', 'inkling'],
];
const judges = ['sol', 'sonnet', 'gemini'];
const dReceivers = ['q27', 'inkling', 'inkling'];
const design = {tones: 3, nodesPerRun: 24, maxLeaves: 3, maxDepth: 12, ablationContextsPerRun: 2};

// These are average BILLABLE token assumptions, not measured lengths or
// approved generation settings. Outputs include hidden reasoning tokens.
const scenarios = {
  lean: {
    panelIn: 4000, panelOut: 900, positionIn: 4500, positionOut: 350,
    initialIn: 800, initialPositionIn: 1800,
    bJudgeIn: 1600, cJudgeIn: 1000, judgeOut: 400,
    eJudgeIn: 2400, eJudgeOut: 400,
    chairIn: 12500, baselineChairIn: 2500, chairOut: 1500,
    dIn: 4500, dOut: 200, choiceScoreIn: 5000,
  },
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

function estimate(questions, repetitions, s, missingQwenPriceMultiplier = 1, judgeRoster = judges) {
  const configs = rosters.length;
  const runs = questions * repetitions * configs * design.tones;
  const initial = questions * repetitions * configs * 3;
  const main = runs * design.nodesPerRun;
  const ablation = runs * design.ablationContextsPerRun * 2;
  // Upper allowance for initial->last C comparisons. Actual identical pairs
  // can be reused; no extra positions are generated for inactive members.
  const terminal = runs * design.maxLeaves * 3;
  // Expected counts under model-symmetric random routing, NOT fixed quotas.
  const dNodes = questions * repetitions * design.tones * design.nodesPerRun;
  const dChoiceReads = dNodes + questions * repetitions * dReceivers.length;
  const adjusted = Object.fromEntries(Object.entries(rates).map(([m, p]) => [
    m, p.map(x => x * (['q27', 'q35'].includes(m) ? missingQwenPriceMultiplier : 1)),
  ]));
  const cost = (m, input, output) => (input * adjusted[m][0] + output * adjusted[m][1]) / 1e6;
  const panel = (input, output) => rosters.flat().reduce((a, m) => a + cost(m, input, output), 0) / 18;
  const judgePanel = (input, output) => judgeRoster.reduce((a, m) => a + cost(m, input, output), 0);
  const probabilityPanel = (input, output) => dReceivers.reduce((a, m) => a + cost(m, input, output), 0) / 3;
  const components = {
    initial_views: initial * panel(s.initialIn, s.panelOut),
    debate: main * panel(s.panelIn, s.panelOut),
    C_position_generation: initial * panel(s.initialPositionIn, s.positionOut) + main * panel(s.positionIn, s.positionOut),
    B_judging: main * judgePanel(s.bJudgeIn, s.judgeOut),
    C_judging: (main + terminal) * judgePanel(s.cJudgeIn, s.judgeOut),
    A_ablation_generation: ablation * panel(s.panelIn, s.panelOut),
    // Included as a budget allowance, not a newly approved analysis requirement.
    A_ablation_text_judging: ablation * judgePanel(s.bJudgeIn, s.judgeOut),
    // Conservatively allow an extra prefix-scoring pass for every D-choice read.
    D_choice_extra_scoring: dChoiceReads * probabilityPanel(s.choiceScoreIn, 1),
    D_text_argument_control: 2 * dNodes * probabilityPanel(s.dIn, s.dOut),
    E_chairman: runs * cost('terra', s.chairIn, s.chairOut)
      + questions * repetitions * configs * cost('terra', s.baselineChairIn, s.chairOut),
    E_judging_both_orders: runs * 2 * judgePanel(s.eJudgeIn, s.eJudgeOut),
  };
  const total = Object.values(components).reduce((a, x) => a + x, 0);
  const calls = initial + main + initial + main + main * judgeRoster.length + (main + terminal) * judgeRoster.length
    + ablation + ablation * judgeRoster.length + dChoiceReads + 2 * dNodes
    + runs + questions * repetitions * configs + runs * 2 * judgeRoster.length;
  return {
    questions, repetitions, runs, initial, main, ablation, terminal,
    dNodes, dChoiceReads, judgeModels: judgeRoster, logicalCalls: calls, components, total,
    with30PercentReserve: total * 1.3,
  };
}

const questions = Number(process.argv[2] ?? 100);
const repetitions = Number(process.argv[3] ?? 2);
assert(Number.isSafeInteger(questions) && questions > 0, 'questions must be a positive integer');
assert(Number.isSafeInteger(repetitions) && repetitions > 0, 'repetitions must be a positive integer');
const check = estimate(100, 2, scenarios.central);
assert.equal(check.runs, 3600);
assert.equal(check.main, 86400);
assert.equal(check.initial, 3600);
assert.equal(check.dNodes, 14400);
assert.equal(check.logicalCalls, 923400);
assert(Math.abs(check.total - 16589.892875) < 1e-6);
assert(Math.abs(estimate(50, 2, scenarios.central).total * 2 - check.total) < 1e-6);
const singleGemini = estimate(100, 2, scenarios.central, 1, ['gemini']);
assert(Math.abs(singleGemini.total - 7483.332875) < 1e-6);
assert.equal(singleGemini.logicalCalls, 469800);
assert.equal(singleGemini.components.C_position_generation, check.components.C_position_generation);
const result = {
  status: 'Historical six-roster full-measurement scenarios only. Superseded by three rosters, one judge, and upfront B/C sampling; models and sample sizes pending. The 24-node assumption is not a current default.',
  asOf: '2026-09-24', design, rates, tokenAssumptions: scenarios,
  estimates: Object.fromEntries(Object.entries(scenarios).map(([name, s]) => [name, estimate(questions, repetitions, s)])),
  singleJudgeCentralComparisons: Object.fromEntries(judges.map(model => [model, estimate(questions, repetitions, scenarios.central, 1, [model])])),
  pilotFiveQuestionsOneRepetition: estimate(5, 1, scenarios.central),
  missingQwenPriceSensitivity: {
    halfProxy: estimate(questions, repetitions, scenarios.central, 0.5).total,
    doubleProxy: estimate(questions, repetitions, scenarios.central, 2).total,
  },
};
console.log(JSON.stringify(result, null, 2));
