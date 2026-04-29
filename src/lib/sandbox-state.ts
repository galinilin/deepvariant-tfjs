import type { Candidate } from './candidate';
import type { Genotype } from './DeepVariantModel';

export interface PredictionState {
  probs: Record<Genotype, number>;
  argmax: Genotype;
  confidence: number;
  /** Position the prediction was computed at — used to invalidate when the
   * window slides past it. */
  position: number;
}

/**
 * Tiny shared state object so the top and bottom p5 sketches can talk
 * without a heavier event bus. Top writes; bottom reads.
 */
export const sandboxState: {
  candidate: Candidate;
  pileupTensor: Float32Array | null;
  pileupPosition: number;
  /** Bumped on Randomize so the bottom sketch invalidates predictions even
   * when predictPos lands on the same column. */
  readsGeneration: number;
  prediction: PredictionState | null;
  predicting: boolean;
} = {
  candidate: null,
  pileupTensor: null,
  pileupPosition: -1,
  readsGeneration: 0,
  prediction: null,
  predicting: false,
};
