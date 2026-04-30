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
  /** Real DV-threshold-passing candidate at the predict column (or null). */
  candidate: Candidate;
  /** Whether the current pileupTensor was encoded with a forced (debug)
   * candidate rather than a real threshold-passing one. */
  candidateForced: boolean;
  pileupTensor: Float32Array | null;
  pileupPosition: number;
  /** Bumped on Randomize so the bottom sketch invalidates predictions even
   * when predictPos lands on the same column. */
  readsGeneration: number;
  prediction: PredictionState | null;
  predicting: boolean;
  /** Debug mode: encode + predict at every column, even when the real
   * candidate is null or below threshold. Toggleable from the corner UI. */
  forcePredict: boolean;
  /** Verbose console logs for every prediction: input stats + output probs.
   * Co-controlled by forcePredict. */
  debugLogs: boolean;
} = {
  candidate: null,
  candidateForced: false,
  pileupTensor: null,
  pileupPosition: -1,
  readsGeneration: 0,
  prediction: null,
  predicting: false,
  forcePredict: false,
  debugLogs: false,
};
