import type { Candidate } from './candidate';
import type { Read } from './reads';
import type { Base } from './palette';

/**
 * Tiny shared state object so the top and bottom p5 sketches can talk
 * without a heavier event bus. Top writes; bottom reads.
 *
 * v3.2 expansion: bottom canvas now needs reads + reference + the
 * predict position to encode the pileup tensor and run inference. We
 * also carry a `readsGeneration` counter that increments every time the
 * top sketch regenerates the world (Randomize) — bottom uses it to know
 * when to invalidate its cached prediction.
 */
export const sandboxState: {
  candidate: Candidate;
  reads: Read[] | null;
  reference: Base[] | null;
  predictPos: number | null;
  readsGeneration: number;
} = {
  candidate: null,
  reads: null,
  reference: null,
  predictPos: null,
  readsGeneration: 0,
};
