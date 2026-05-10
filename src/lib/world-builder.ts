import type { Base } from './palette';
import type { Read } from './reads';
import { buildReads, makeRng } from './reads';
import { buildReference } from './reference';
import { placeScenarios, type Scenario } from './scenarios';

export interface World {
  reference: Base[];
  scenarios: Scenario[];
  reads: Read[];
}

export interface WorldOpts {
  seed: number;
}

export function buildWorld(opts: WorldOpts): World {
  const rng = makeRng(opts.seed);
  const reference = buildReference(undefined, opts.seed);
  const scenarios = placeScenarios(reference, rng);
  const reads = buildReads(reference, scenarios, rng);
  return { reference, scenarios, reads };
}
