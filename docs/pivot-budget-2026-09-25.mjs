// OFFLINE BUDGET MODEL ONLY. No API, network, filesystem writes, or experiment execution.
// Usage: node docs/pivot-budget-2026-09-25.mjs [simulatedQuestions=5000] [seed=20260925]
// Simulations estimate counts for the agreed routing/sampling rule, NOT research outcomes.
// Prices checked 2026-09-25. User approved Qwen3.8-27B; q27 = Qwen/Qwen3.8-27B (64K on Tinker).
// q27 uses the selected endpoint's listed price, not a proxy. Account/logprob behavior is untested.
import assert from 'node:assert/strict';

const questions = 80;
const simulations = Number(process.argv[2] ?? 5000);
const seed = Number(process.argv[3] ?? 20260925);
assert(Number.isSafeInteger(simulations) && simulations > 0);
assert(Number.isSafeInteger(seed));
const rosters = [
  ['terra', 'terra', 'terra'],
  ['luna', 'terra', 'sol'],
  ['terra', 'q27', 'inkling'],
];
const prices = {
  terra: {input: 2, cached: .2, output: 12, write: 1.25},
  luna: {input: .2, cached: .02, output: 1.2, write: 1.25},
  sol: {input: 4, cached: .4, output: 20, write: 1.25},
  q27: {input: 1.86, cached: .372, output: 5.595, write: 1},
  inkling: {input: 1.87, cached: .374, output: 4.68, write: 1},
  gemini: {input: .75, cached: .075, output: 3.75, write: 1},
};
// Average BILLABLE tokens, including hidden reasoning. Assumptions, not measured or approved caps.
const tokens = {
  central: {panelIn: 8000, panelOut: 1800, initialIn: 1500,
    positionIn: 8500, positionOut: 700, initialPositionIn: 2500,
    bJudgeIn: 2000, cJudgeIn: 1400, judgeOut: 1000,
    dIn: 8500, dOut: 500, choiceScoreIn: 9200,
    chairNineNodeIn: 18000, baselineChairIn: 3500, chairOut: 2500,
    eJudgeIn: 3200, eJudgeOut: 800},
  heavy: {panelIn: 16000, panelOut: 4500, initialIn: 2000,
    positionIn: 16500, positionOut: 1800, initialPositionIn: 4000,
    bJudgeIn: 3500, cJudgeIn: 2200, judgeOut: 3000,
    dIn: 16500, dOut: 1500, choiceScoreIn: 18200,
    chairNineNodeIn: 35000, baselineChairIn: 6000, chairOut: 6000,
    eJudgeIn: 5000, eJudgeOut: 2500},
};

// Exhaustively check possible binary/unary tree shapes (ignoring member IDs).
// Every root-to-leaf path has 5 formal replies; two roots and 6 terminal paths.
function shapeSupport(depth) {
  if (depth === 5) return [[1, 1]];
  const children = shapeSupport(depth + 1);
  const result = new Set(children.map(([leaves, size]) => `${leaves}:${size + 1}`));
  for (const [l1, n1] of children) for (const [l2, n2] of children)
    if (l1 + l2 <= 6) result.add(`${l1 + l2}:${n1 + n2 + 1}`);
  return [...result].map(key => key.split(':').map(Number));
}
const rootShapes = shapeSupport(1);
const forestSizes = new Set();
for (const [l1, n1] of rootShapes) for (const [l2, n2] of rootShapes)
  if (l1 + l2 === 6) forestSizes.add(n1 + n2);
assert.equal(Math.min(...forestSizes), 15);
assert.equal(Math.max(...forestSizes), 24);

function rng(initial) {
  let state = initial >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let x = Math.imul(state ^ state >>> 15, 1 | state);
    x ^= x + Math.imul(x ^ x >>> 7, 61 | x);
    return ((x ^ x >>> 14) >>> 0) / 4294967296;
  };
}
const random = rng(seed);
const pick = a => { assert(a.length); return a[Math.floor(random() * a.length)]; };
const others = member => [0, 1, 2].filter(m => m !== member);

function makeRoute() {
  const nodes = [];
  function add(parent, member, sender) {
    const node = {id: nodes.length, parent, member, sender,
      depth: parent === null ? 1 : nodes[parent].depth + 1, children: []};
    nodes.push(node);
    if (parent !== null) nodes[parent].children.push(node.id);
    return node;
  }
  function extend(node) {
    while (node.depth < 5) node = add(node.id, pick(others(node.member)), node.member);
  }
  const pairs = [0, 1, 2].flatMap(sender => others(sender).map(receiver => [sender, receiver]));
  for (let root = 0; root < 2; root++) {
    const index = Math.floor(random() * pairs.length);
    const [sender, receiver] = pairs.splice(index, 1)[0];
    extend(add(null, receiver, sender));
  }
  for (let fork = 0; fork < 4; fork++) {
    const node = pick(nodes.filter(n => n.depth < 5 && n.children.length === 1));
    const existingReceiver = nodes[node.children[0]].member;
    const receiver = others(node.member).find(m => m !== existingReceiver);
    extend(add(node.id, receiver, node.member));
  }
  const leaves = nodes.filter(n => !n.children.length);
  assert.equal(leaves.length, 6);
  assert(nodes.length >= 15 && nodes.length <= 24);
  const finals = new Set();
  for (const leaf of leaves) {
    assert.equal(leaf.depth, 5);
    const seen = new Set();
    for (let n = leaf; n; n = n.parent === null ? null : nodes[n.parent]) {
      if (!seen.has(n.member)) { finals.add(n.id); seen.add(n.member); }
    }
  }
  for (const n of nodes) {
    assert(n.member !== n.sender);
    assert(n.children.length <= 2);
    if (n.parent !== null) assert.equal(n.sender, nodes[n.parent].member);
  }
  assert.equal(nodes.filter(n => n.parent === null).length, 2);
  assert.equal(new Set(nodes.filter(n => n.parent === null).map(n => `${n.sender}:${n.member}`)).size, 2);
  return {nodes, finals};
}

function previousOwn(nodes, node) {
  for (let id = node.parent; id !== null; id = nodes[id].parent)
    if (nodes[id].member === node.member) return id;
  return null;
}

function sampleEight(nodes) {
  // Pool tones FIRST. No per-tone quota and no automatic matched-tone measurements.
  const pools = [1, 2, 3, 4, 5].map(depth => [0, 1, 2].flatMap(tone =>
    nodes.filter(n => n.depth === depth).map(n => ({tone, id: n.id}))));
  const selected = [];
  const take = index => {
    assert(pools[index].length);
    selected.push(pools[index].splice(Math.floor(random() * pools[index].length), 1)[0]);
  };
  for (let i = 0; i < 5; i++) take(i);
  for (let i = 0; i < 3; i++) {
    const u = random() * 5.5;
    take(u < 4 ? Math.floor(u) : 4);
  }
  assert.equal(selected.length, 8);
  assert.equal(new Set(selected.map(e => `${e.tone}:${e.id}`)).size, 8);
  for (let depth = 1; depth <= 5; depth++)
    assert(selected.some(e => nodes[e.id].depth === depth));
  return selected;
}

const ledger = {};
function add(kind, model, calls = 1, summedNodes = 0) {
  const key = `${kind}/${model}`;
  const row = ledger[key] ??= {kind, model, calls: 0, summedNodes: 0};
  row.calls += calls;
  row.summedNodes += summedNodes;
}
const routingHistogram = {};
const sampledByDepth = [0, 0, 0, 0, 0];
let selectedFinalEvents = 0;

for (let q = 0; q < simulations; q++) {
  // Exactly one abstract routing template per question, reused across rosters AND tones.
  const {nodes, finals} = makeRoute();
  routingHistogram[nodes.length] = (routingHistogram[nodes.length] ?? 0) + 1;
  for (let c = 0; c < rosters.length; c++) {
    const roster = rosters[c];
    for (const model of roster) add('initial', model);
    for (const n of nodes) add('debate', roster[n.member], 3);
    add('Ebaseline', 'terra');
    add('Echair', 'terra', 3, 3 * nodes.length);
    add('Ejudge', 'gemini', 6);
    const chosen = sampleEight(nodes);
    const reads = new Map();
    const pairs = new Set();
    function read(tone, id, member) {
      const key = id === null ? `initial:${member}` : `${tone}:${id}`;
      reads.set(key, {id, model: roster[member]});
      return key;
    }
    for (const {tone, id} of chosen) {
      const n = nodes[id];
      sampledByDepth[n.depth - 1]++;
      add('Bjudge', 'gemini');
      const now = read(tone, id, n.member);
      const prev = read(tone, previousOwn(nodes, n), n.member);
      // Choice and D-choice retain baseline->current at every sampled point.
      // Only terminal points add a baseline->current TEXT judging pair.
      const initial = read(tone, null, n.member);
      pairs.add(`${prev}->${now}`);
      if (finals.has(id)) {
        selectedFinalEvents++;
        // Same pair may be both adjacent and initial->final: judge once and reuse.
        pairs.add(`${initial}->${now}`);
      }
      if (c === 2 && n.member !== 0) add('Dtext', roster[n.member], 2);
    }
    for (const {id, model} of reads.values()) {
      add(id === null ? 'Cinitial' : 'Cpost', model);
      if (c === 2 && ['q27', 'inkling'].includes(model)) add('DchoiceFallback', model);
    }
    add('Cjudge', 'gemini', pairs.size);
    assert(pairs.size >= 8 && pairs.size <= 16);
    assert(reads.size <= 19); // 8 current + <=8 previous + <=3 initial.
    // UNCONFIRMED implementation proposal / cost allowance:
    // One matched logical node per question+roster, in friendly and hostile only.
    // Each tone gets a fresh retained reply AND a fresh removed reply.
    const ablationNode = pick(nodes);
    add('Aablation', roster[ablationNode.member], 4);
    add('AjudgeAllowance', 'gemini', 4);
  }
}

const scale = questions / simulations;
const rows = Object.values(ledger).map(row => ({...row,
  calls: row.calls * scale, summedNodes: row.summedNodes * scale}));
const count = kind => rows.filter(r => r.kind === kind).reduce((s, r) => s + r.calls, 0);
assert(Math.abs(count('Bjudge') - 1920) < 1e-8);
assert(Math.abs(count('initial') - 720) < 1e-8);
assert(Math.abs(count('Echair') - 720) < 1e-8);
assert(Math.abs(count('Ebaseline') - 240) < 1e-8);
assert(Math.abs(count('Ejudge') - 1440) < 1e-8);
assert(Math.abs(count('Aablation') - 960) < 1e-8);

function tokenTotals(row, t) {
  const n = row.calls;
  const kinds = {
    initial: [t.initialIn, t.panelOut, false],
    debate: [t.panelIn, t.panelOut, true],
    Cinitial: [t.initialPositionIn, t.positionOut, false],
    Cpost: [t.positionIn, t.positionOut, true],
    Bjudge: [t.bJudgeIn, t.judgeOut, false],
    Cjudge: [t.cJudgeIn, t.judgeOut, false],
    Aablation: [t.panelIn, t.panelOut, true],
    AjudgeAllowance: [t.bJudgeIn, t.judgeOut, false],
    DchoiceFallback: [t.choiceScoreIn, 1, true],
    Dtext: [t.dIn, t.dOut, true],
    Ebaseline: [t.baselineChairIn, t.chairOut, false],
    Ejudge: [t.eJudgeIn, t.eJudgeOut, false],
  };
  if (row.kind === 'Echair') return {
    input: n * t.baselineChairIn + row.summedNodes * (t.chairNineNodeIn - t.baselineChairIn) / 9,
    output: n * t.chairOut, reusable: false,
  };
  const [input, output, reusable] = kinds[row.kind];
  return {input: input * n, output: output * n, reusable};
}

function estimate({cacheHit = null, heavy = false, includeAblation = true,
  includeChoiceFallback = true, qwenMultiplier = 1, lineItems = rows} = {}) {
  const t = heavy ? tokens.heavy : tokens.central;
  const components = {};
  let logicalCalls = 0;
  for (const row of lineItems) {
    if (!includeAblation && ['Aablation', 'AjudgeAllowance'].includes(row.kind)) continue;
    if (!includeChoiceFallback && row.kind === 'DchoiceFallback') continue;
    const p = prices[row.model];
    const {input, output, reusable} = tokenTotals(row, t);
    const rate = reusable && cacheHit !== null
      ? cacheHit * p.cached + (1 - cacheHit) * p.input * p.write : p.input;
    const factor = row.model === 'q27' ? qwenMultiplier : 1;
    const dollars = factor * (input * rate + output * p.output) / 1e6;
    components[row.kind] = (components[row.kind] ?? 0) + dollars;
    logicalCalls += row.calls;
  }
  const total = Object.values(components).reduce((a, b) => a + b, 0);
  return {components, total, with30PercentReserve: total * 1.3, logicalCalls};
}

// Conservative count stress case, not an expected routing nor a dollar guarantee:
// <=24 unique replies/run, <=19 C readings and <=16 C pairs/question+roster.
// Use the most expensive model within each roster for the unconstrained C/debate
// receiver counts; deliberately loose (not a realizable all-Sol routing).
function countStressRows() {
  const out = [];
  const push = (kind, model, calls, summedNodes = 0) => out.push({kind, model, calls, summedNodes});
  for (const roster of rosters) {
    const expensive = [...roster].sort((a, b) => {
      const cost = x => prices[x].input * tokens.central.panelIn + prices[x].output * tokens.central.panelOut;
      return cost(b) - cost(a);
    })[0];
    for (const m of roster) { push('initial', m, questions); push('Cinitial', m, questions); }
    push('debate', expensive, questions * 3 * 24);
    push('Cpost', expensive, questions * 16);
    push('Bjudge', 'gemini', questions * 8);
    push('Cjudge', 'gemini', questions * 16);
    push('Aablation', expensive, questions * 4);
    push('AjudgeAllowance', 'gemini', questions * 4);
    push('Ebaseline', 'terra', questions);
    push('Echair', 'terra', questions * 3, questions * 3 * 24);
    push('Ejudge', 'gemini', questions * 6);
  }
  push('Dtext', 'q27', questions * 8 * 2);
  // Inkling has the slightly higher prefill price; q27 has the higher output price.
  push('DchoiceFallback', 'inkling', questions * 18);
  return out;
}

const halfCached = estimate({cacheHit: .5});
const disabled = estimate();
const noHits = estimate({cacheHit: 0});
assert(halfCached.total < disabled.total && disabled.total < noHits.total);
assert(estimate({cacheHit: .5, includeChoiceFallback: false}).total < halfCached.total);
assert.equal(estimate({cacheHit: .5, includeChoiceFallback: false}).components.Cpost, halfCached.components.Cpost);
assert(estimate({cacheHit: .5, includeAblation: false}).total < halfCached.total);

console.log(JSON.stringify({
  status: 'Offline count simulation + quotation, not model runs, results, power analysis, or spending approval. Main protocol decisions honored; A ablation volume and Gemini judge remain PROPOSALS. User approved Qwen3.8-27B; q27 uses listed Tinker Qwen/Qwen3.8-27B 64K prices, not a proxy. Account access and exact probability reads remain untested.',
  asOf: '2026-09-25', questions, simulations, seed, repetitions: 1,
  rosters, prices, tokenAssumptions: tokens,
  routing: {roots: 2, maxDepth: 5, forks: 4, terminalPaths: 6,
    uniqueReplies: {min: 15, max: 24, simulatedMean: count('debate') / 720},
    histogram: routingHistogram},
  sampling: {perQuestionRosterPooledAcrossTones: 8, depthWeightsForLastThree: [1, 1, 1, 1, 1.5],
    expectedSampleCountsByDepth: sampledByDepth.map(n => n * scale),
    primaryEvents: 1920, expectedFinalEligibleEvents: selectedFinalEvents * scale},
  expectedCounts: Object.fromEntries([...new Set(rows.map(r => r.kind))].map(k => [k, count(k)])),
  scenarios: {halfEligibleInputCached: halfCached, cachingDisabled: disabled,
    enabledButNoHits: noHits,
    heavyHalfCached: estimate({cacheHit: .5, heavy: true}),
    heavyCachingDisabled: estimate({heavy: true}),
    heavyEnabledNoHits: estimate({heavy: true, cacheHit: 0})},
  sensitivities: {
    noUnconfirmedAblationAllowance: estimate({cacheHit: .5, includeAblation: false}),
    completeChoiceLogprobsInline: estimate({cacheHit: .5, includeChoiceFallback: false}),
    qwenTwiceListedPrice: estimate({cacheHit: .5, qwenMultiplier: 2}),
    conservativeCountAndReceiverStressNoHits: estimate({cacheHit: 0, lineItems: countStressRows()}),
    conservativeCountAndReceiverStressHeavyNoHits: estimate({cacheHit: 0, heavy: true, lineItems: countStressRows()}),
  },
  pilotThreeSeparateQuestions: {
    status: 'PROPOSAL ONLY: three non-main-study questions, same design; average cost scaling, not an authorized pilot.',
    halfCachedWithReserve: halfCached.with30PercentReserve * 3 / 80,
    noHitsWithReserve: noHits.with30PercentReserve * 3 / 80,
    heavyNoHitsWithReserve: estimate({heavy: true, cacheHit: 0}).with30PercentReserve * 3 / 80,
  },
  exclusions: ['new question screening', 'human wages', 'GPU hosting', 'tax/storage/engineering',
    'rerunning old supplementary experiments', 'extra tools/search', 'price changes after promotions'],
  sources: [
    'https://developers.openai.com/api/docs/models/gpt-5.6-terra',
    'https://developers.openai.com/api/docs/models/gpt-5.6-luna',
    'https://developers.openai.com/api/docs/models/gpt-5.6-sol',
    'https://developers.openai.com/api/docs/guides/prompt-caching',
    'https://ai.google.dev/gemini-api/docs/pricing',
    'https://tinker-docs.thinkingmachines.ai/tinker/models.json',
    'https://tinker-docs.thinkingmachines.ai/tinker/models/#retired-models',
  ],
}, null, 2));
