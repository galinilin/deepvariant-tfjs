import { DeepVariantModel, DV_CLASSES, DV_INPUT_SHAPE } from './DeepVariantModel';
import { readNpyFloat32 } from './npy';

/**
 * The bundled model is the uint8-quantized variant (`tfjs_dv_wgs_uint8`),
 * while the parity fixtures were almost certainly generated from the
 * full-precision float32 reference. We tolerate quantization-level drift
 * on individual class probabilities (~2-3% is normal) but require the
 * argmax (final genotype call) to match exactly.
 */
const PROB_TOLERANCE = 3e-2;

export interface ParityResult {
  passed: boolean;
  argmaxMatches: boolean;
  numSamples: number;
  maxDelta: number;
  details: string[];
}

function argmax3(probs: [number, number, number]): 0 | 1 | 2 {
  if (probs[1] > probs[0] && probs[1] >= probs[2]) return 1;
  if (probs[2] > probs[0] && probs[2] > probs[1]) return 2;
  return 0;
}

/**
 * Round-trip the parity fixtures (`public/fixtures/parity_inputs.npy` →
 * `parity_outputs.npy`) through the converted DV model. Confirms the
 * loaded model + inference path match DV's Python reference output to
 * within `TOLERANCE`. Run this before trusting the encoder we're about
 * to build — it isolates "model loads and predicts correctly" from
 * "encoder produces correct tensors."
 */
export async function verifyParity(opts?: {
  fixtureBaseUrl?: string;
  modelBaseUrl?: string;
  onProgress?: (s: string) => void;
}): Promise<ParityResult> {
  const base = import.meta.env.BASE_URL;
  const fixtureBase = (opts?.fixtureBaseUrl ?? `${base}fixtures/`).replace(
    /\/?$/,
    '/',
  );
  const modelBase = opts?.modelBaseUrl ?? `${base}models/`;
  const onProgress = opts?.onProgress ?? (() => {});

  onProgress('Loading model…');
  const model = await DeepVariantModel.load({ modelBaseUrl: modelBase });

  onProgress('Loading parity fixtures…');
  const [inputBuf, outputBuf] = await Promise.all([
    fetch(`${fixtureBase}parity_inputs.npy`).then((r) => r.arrayBuffer()),
    fetch(`${fixtureBase}parity_outputs.npy`).then((r) => r.arrayBuffer()),
  ]);
  const inputs = readNpyFloat32(inputBuf);
  const outputs = readNpyFloat32(outputBuf);

  const [n, h, w, c] = inputs.shape;
  if (
    h !== DV_INPUT_SHAPE[0] ||
    w !== DV_INPUT_SHAPE[1] ||
    c !== DV_INPUT_SHAPE[2]
  ) {
    model.dispose();
    throw new Error(
      `fixture input shape [${inputs.shape}] does not match model input [N, ${DV_INPUT_SHAPE.join(', ')}]`,
    );
  }
  if (outputs.shape[0] !== n || outputs.shape[1] !== 3) {
    model.dispose();
    throw new Error(
      `fixture output shape [${outputs.shape}] does not match expected [${n}, 3]`,
    );
  }

  onProgress(`Predicting ${n} sample${n === 1 ? '' : 's'}…`);
  const results = await model.predictBatch({ data: inputs.data, batch: n });

  const details: string[] = [];
  let maxDelta = 0;
  let argmaxAllMatch = true;
  let probAllPass = true;
  for (let i = 0; i < n; i++) {
    const got: [number, number, number] = [
      results[i].probs.hom_ref,
      results[i].probs.het,
      results[i].probs.hom_alt,
    ];
    const expected: [number, number, number] = [
      outputs.data[i * 3],
      outputs.data[i * 3 + 1],
      outputs.data[i * 3 + 2],
    ];
    const deltas = got.map((v, j) => Math.abs(v - expected[j]));
    const local = Math.max(...deltas);
    if (local > maxDelta) maxDelta = local;
    const probOk = local <= PROB_TOLERANCE;
    if (!probOk) probAllPass = false;
    const gotArg = argmax3(got);
    const expArg = argmax3(expected);
    const argOk = gotArg === expArg;
    if (!argOk) argmaxAllMatch = false;
    details.push(
      `sample ${i}: argmax=${DV_CLASSES[gotArg]}` +
        (argOk ? '' : ` (exp ${DV_CLASSES[expArg]})`) +
        ` probs=[${got.map((v) => v.toFixed(4)).join(', ')}]` +
        ` exp=[${expected.map((v) => v.toFixed(4)).join(', ')}]` +
        ` Δmax=${local.toExponential(2)}` +
        ` ${argOk && probOk ? '✓' : argOk ? '~ (within quant noise)' : '✗'}`,
    );
  }

  model.dispose();
  return {
    passed: argmaxAllMatch && probAllPass,
    argmaxMatches: argmaxAllMatch,
    numSamples: n,
    maxDelta,
    details,
  };
}

/** DV 1.8 WGS channel order (channel IDs [1,2,3,4,5,6,19] per
 * `testdata/example_info.json`). Channel 7 is insert_size. */
export const CHANNEL_NAMES = [
  'read_base',
  'base_quality',
  'mapping_quality',
  'strand',
  'supports_variant',
  'differs_from_ref',
  'insert_size',
] as const;

/**
 * Per-channel statistics on a fixture. Used to reverse-engineer DV's
 * encoding constants (which discrete values represent A/C/G/T/-, the
 * strand high/low intensities, etc.) by inspecting the real
 * `golden_pileups.npy` extracted from upstream calling examples.
 */
export async function inspectChannels(opts: {
  fixtureUrl: string;
  topK?: number;
}): Promise<void> {
  const buf = await fetch(opts.fixtureUrl).then((r) => r.arrayBuffer());
  const fixture = readNpyFloat32(buf);
  const [n, h, w, c] = fixture.shape;
  const sampleStride = h * w * c;
  const topK = opts.topK ?? 10;
  console.log(`fixture: ${n} samples, shape [${h}, ${w}, ${c}]`);
  for (let ch = 0; ch < c; ch++) {
    let min = Infinity;
    let max = -Infinity;
    const histogram = new Map<number, number>();
    let total = 0;
    for (let s = 0; s < n; s++) {
      for (let y = 0; y < h; y++) {
        for (let x = 0; x < w; x++) {
          const v = fixture.data[s * sampleStride + y * w * c + x * c + ch];
          if (v < min) min = v;
          if (v > max) max = v;
          const bucket = Math.round(v);
          histogram.set(bucket, (histogram.get(bucket) ?? 0) + 1);
          total++;
        }
      }
    }
    const sorted = Array.from(histogram.entries()).sort((a, b) => b[1] - a[1]);
    const top = sorted
      .slice(0, topK)
      .map(([k, count]) => `${k}×${count}`)
      .join(', ');
    console.log(
      `[${ch}] ${CHANNEL_NAMES[ch] ?? `ch${ch}`}: min=${min.toFixed(2)} max=${max.toFixed(2)} unique=${histogram.size} top${topK}: ${top}`,
    );
  }
}

/** Synthetic-noise fixture inspector kept for diagnostics. */
export async function inspectFixtureChannels(opts?: {
  fixtureBaseUrl?: string;
}): Promise<void> {
  const base = import.meta.env.BASE_URL;
  const fixtureBase = (opts?.fixtureBaseUrl ?? `${base}fixtures/`).replace(
    /\/?$/,
    '/',
  );
  await inspectChannels({ fixtureUrl: `${fixtureBase}parity_inputs.npy` });
}

/** Real-pileup fixture inspector. Run after extract_golden.py has populated
 * `public/fixtures/golden_pileups.npy`. Reveals real DV encoding constants. */
export async function inspectGoldenChannels(opts?: {
  fixtureBaseUrl?: string;
}): Promise<void> {
  const base = import.meta.env.BASE_URL;
  const fixtureBase = (opts?.fixtureBaseUrl ?? `${base}fixtures/`).replace(
    /\/?$/,
    '/',
  );
  await inspectChannels({ fixtureUrl: `${fixtureBase}golden_pileups.npy` });
}

export interface GoldenLabel {
  index: number;
  pred: 'hom_ref' | 'het' | 'hom_alt';
  probs: { hom_ref: number; het: number; hom_alt: number };
  variant_summary: string;
}

/**
 * Round-trip the real golden pileup fixture through our TFJS uint8 model
 * and compare to upstream Keras predictions stored alongside. A stronger
 * semantic test than `verifyParity` (which uses synthetic noise): this
 * confirms the model behaves on actual DV data the way Python does.
 */
export async function verifyGoldenParity(opts?: {
  fixtureBaseUrl?: string;
  modelBaseUrl?: string;
  onProgress?: (s: string) => void;
}): Promise<ParityResult & { labels: GoldenLabel[] }> {
  const base = import.meta.env.BASE_URL;
  const fixtureBase = (opts?.fixtureBaseUrl ?? `${base}fixtures/`).replace(
    /\/?$/,
    '/',
  );
  const modelBase = opts?.modelBaseUrl ?? `${base}models/`;
  const onProgress = opts?.onProgress ?? (() => {});

  onProgress('Loading model…');
  const model = await DeepVariantModel.load({ modelBaseUrl: modelBase });

  onProgress('Loading golden fixtures…');
  const [inputBuf, outputBuf, labelsResp] = await Promise.all([
    fetch(`${fixtureBase}golden_pileups.npy`).then((r) => r.arrayBuffer()),
    fetch(`${fixtureBase}golden_outputs.npy`).then((r) => r.arrayBuffer()),
    fetch(`${fixtureBase}golden_labels.json`).then((r) => r.json()),
  ]);
  const inputs = readNpyFloat32(inputBuf);
  const outputs = readNpyFloat32(outputBuf);
  const labels = labelsResp as GoldenLabel[];

  const [n, h, w, c] = inputs.shape;
  if (
    h !== DV_INPUT_SHAPE[0] ||
    w !== DV_INPUT_SHAPE[1] ||
    c !== DV_INPUT_SHAPE[2]
  ) {
    model.dispose();
    throw new Error(
      `golden input shape [${inputs.shape}] != [N, ${DV_INPUT_SHAPE.join(', ')}]`,
    );
  }

  onProgress(`Predicting ${n} golden sample${n === 1 ? '' : 's'}…`);
  const results = await model.predictBatch({ data: inputs.data, batch: n });

  const details: string[] = [];
  let maxDelta = 0;
  let argmaxAllMatch = true;
  let probAllPass = true;
  for (let i = 0; i < n; i++) {
    const got: [number, number, number] = [
      results[i].probs.hom_ref,
      results[i].probs.het,
      results[i].probs.hom_alt,
    ];
    const expected: [number, number, number] = [
      outputs.data[i * 3],
      outputs.data[i * 3 + 1],
      outputs.data[i * 3 + 2],
    ];
    const deltas = got.map((v, j) => Math.abs(v - expected[j]));
    const local = Math.max(...deltas);
    if (local > maxDelta) maxDelta = local;
    const probOk = local <= PROB_TOLERANCE;
    if (!probOk) probAllPass = false;
    const gotArg = argmax3(got);
    const expArg = argmax3(expected);
    const argOk = gotArg === expArg;
    if (!argOk) argmaxAllMatch = false;
    const lbl = labels[i];
    details.push(
      `golden ${i}: argmax=${DV_CLASSES[gotArg]}` +
        (argOk ? '' : ` (exp ${DV_CLASSES[expArg]})`) +
        ` Δmax=${local.toExponential(2)}` +
        ` ${argOk && probOk ? '✓' : argOk ? '~' : '✗'}` +
        (lbl ? `  ${lbl.variant_summary.slice(0, 60)}` : ''),
    );
  }

  model.dispose();
  return {
    passed: argmaxAllMatch && probAllPass,
    argmaxMatches: argmaxAllMatch,
    numSamples: n,
    maxDelta,
    details,
    labels,
  };
}
