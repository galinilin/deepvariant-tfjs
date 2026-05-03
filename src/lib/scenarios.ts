import type { Base } from './palette';
import { WINDOW_LENGTH } from './reference';

export type ScenarioType =
  | 'het'
  | 'hom_alt'
  | 'het_del'
  | 'hom_alt_del'
  | 'het_ins'
  | 'hom_alt_ins'
  | 'noisy_burst'
  | 'strand_bias_het'
  | 'tandem_snps'
  | 'compound_alt'
  | 'low_quality_alt'
  | 'low_mapq_alt';

export interface Scenario {
  position: number;
  type: ScenarioType;
  altBase?: Base;
  altBase2?: Base;
  delLength?: number;
  insLength?: number;
}

const REQUIRED_TYPES: ScenarioType[] = ['het', 'hom_alt'];
const EXTRA_POOL: ScenarioType[] = [
  'het',
  'hom_alt',
  'het_del',
  'hom_alt_del',
  'het_ins',
  'hom_alt_ins',
  'noisy_burst',
  'strand_bias_het',
  'tandem_snps',
  'compound_alt',
  'low_quality_alt',
  'low_mapq_alt',
];

// v3.2: spacing >= window width so the 221-bp pileup window contains at
// most one scenario at a time. Real-BAM density is ~1 per 1000 bp; we're
// ~1 per 500 bp here — denser than real, but no longer pathological.
const MIN_SPACING = 350;
const BASES: Base[] = ['A', 'C', 'G', 'T'];

export interface PlaceScenariosOptions {
  refLength: number;
  rng: () => number;
  count?: number;
}

export function placeScenarios(reference: Base[], rng: () => number): Scenario[] {
  const refLength = reference.length;
  // Minimum predict-column position the user can reach: when windowStart=0,
  // the predict column lands at floor(WINDOW_LENGTH/2). Scenarios placed
  // below this are unreachable as predict targets.
  const reachableStart = Math.floor(WINDOW_LENGTH / 2);
  const reachableEnd = refLength - WINDOW_LENGTH + Math.floor(WINDOW_LENGTH / 2);
  // v3.2: 3 scenarios per generation. Sparser variant density keeps the
  // pileup window single-candidate (matches DV's training expectation),
  // while still exercising 3 distinct variant types per Randomize.
  const total = 3;

  const types: ScenarioType[] = [...REQUIRED_TYPES];
  while (types.length < total) {
    types.push(EXTRA_POOL[Math.floor(rng() * EXTRA_POOL.length)]);
  }
  shuffle(types, rng);

  const positions: number[] = [];
  let attempts = 0;
  while (positions.length < total && attempts < 200) {
    attempts++;
    const span = reachableEnd - reachableStart;
    const candidate = reachableStart + Math.floor(rng() * span);
    const tooClose = positions.some(
      (p) => Math.abs(p - candidate) < MIN_SPACING,
    );
    if (!tooClose) positions.push(candidate);
  }
  positions.sort((a, b) => a - b);

  return positions.map((pos, i) => buildScenario(reference, pos, types[i], rng));
}

function buildScenario(
  reference: Base[],
  position: number,
  type: ScenarioType,
  rng: () => number,
): Scenario {
  const refBase = reference[position] ?? 'N';
  const altBase = pickAltBase(refBase, rng);

  let altBase2: Base | undefined;
  if (type === 'compound_alt') {
    altBase2 = pickAltBase(refBase, rng, [altBase]);
  } else if (type === 'tandem_snps') {
    const ref2 = reference[position + 1] ?? 'N';
    altBase2 = pickAltBase(ref2, rng);
  }

  const delLength =
    type === 'het_del' || type === 'hom_alt_del'
      ? 1 + Math.floor(rng() * 3)
      : undefined;

  const insLength =
    type === 'het_ins' || type === 'hom_alt_ins'
      ? 1 + Math.floor(rng() * 3)
      : undefined;

  return { position, type, altBase, altBase2, delLength, insLength };
}

export function pickAltBase(
  refBase: Base,
  rng: () => number,
  exclude: Base[] = [],
): Base {
  const candidates = BASES.filter(
    (b) => b !== refBase && !exclude.includes(b),
  );
  if (candidates.length === 0) return BASES[0];
  return candidates[Math.floor(rng() * candidates.length)];
}

function shuffle<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
