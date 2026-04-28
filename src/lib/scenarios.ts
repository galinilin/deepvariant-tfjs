import type { Base } from './palette';
import { WINDOW_LENGTH, defaultWindowStart } from './reference';

export type ScenarioType =
  | 'hom_ref'
  | 'het'
  | 'hom_alt'
  | 'het_del'
  | 'hom_alt_del'
  | 'error_burst';

export interface Scenario {
  position: number;
  type: ScenarioType;
  altBase?: Base;
  delLength?: number;
}

const REQUIRED_TYPES: ScenarioType[] = ['hom_ref', 'het', 'hom_alt'];
const EXTRA_POOL: ScenarioType[] = [
  'het_del',
  'hom_alt_del',
  'error_burst',
  'het',
  'hom_alt',
];

const MIN_SPACING = 50;
const BASES: Base[] = ['A', 'C', 'G', 'T'];

export interface PlaceScenariosOptions {
  refLength: number;
  rng: () => number;
  count?: number;
}

export function placeScenarios(reference: Base[], rng: () => number): Scenario[] {
  const refLength = reference.length;
  const reachableStart = defaultWindowStart(refLength);
  const reachableEnd = refLength - WINDOW_LENGTH + Math.floor(WINDOW_LENGTH / 2);
  const total = 8 + Math.floor(rng() * 3); // 8, 9, or 10

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
  const delLength =
    type === 'het_del' || type === 'hom_alt_del'
      ? 1 + Math.floor(rng() * 3)
      : undefined;
  return { position, type, altBase, delLength };
}

export function pickAltBase(refBase: Base, rng: () => number): Base {
  const candidates = BASES.filter((b) => b !== refBase);
  return candidates[Math.floor(rng() * candidates.length)];
}

function shuffle<T>(arr: T[], rng: () => number): void {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}
