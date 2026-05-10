/**
 * Lightweight telemetry buffer the bottom-canvas writes to and the debug
 * modal reads from. Decouples the predict pipeline from the modal so the
 * modal can be optional.
 */
import type { Genotype } from './DeepVariantModel';

interface DebugTelemetry {
  lastProbs: [number, number, number] | null;
  lastArgmax: Genotype | null;
  lastPredictMs: number | null;
  predictCount: number;
}

export const debugTelemetry: DebugTelemetry = {
  lastProbs: null,
  lastArgmax: null,
  lastPredictMs: null,
  predictCount: 0,
};

export function recordPrediction(
  probs: [number, number, number],
  argmax: Genotype,
  ms: number,
): void {
  debugTelemetry.lastProbs = probs;
  debugTelemetry.lastArgmax = argmax;
  debugTelemetry.lastPredictMs = ms;
  debugTelemetry.predictCount += 1;
}
