/**
 * match/3.1 validator: load all golden fixtures + run our TS pipeline +
 * compare against DV's outputs at three levels:
 *   L1 candidate  — does our candidate engine pick an alt in DV's alt set?
 *   L2 encoder    — does our pileup tensor live in DV's allowed value space?
 *   L3 prediction — does our model's argmax match DV's, modulo uint8 noise?
 *
 * Run via `npm run match` (uses tsx).
 */

// tfjs-node 4.22 calls a couple of helpers removed from Node's `util` in
// Node 18+. Polyfill before tfjs-node loads — must use createRequire to
// get a writable handle (the ESM `import * as util` namespace is frozen).
import { createRequire } from 'node:module';
const _util = createRequire(import.meta.url)('util') as Record<string, unknown>;
if (typeof _util.isNullOrUndefined !== 'function') {
  _util.isNullOrUndefined = (v: unknown) => v === null || v === undefined;
}
if (typeof _util.isArray !== 'function') {
  _util.isArray = Array.isArray;
}

import * as tf from '@tensorflow/tfjs-node';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import path from 'node:path';

import { encodePileup, validateEncodedTensor } from '../src/lib/pileup-encoder';
import {
  deriveCandidateOutcome,
  type Candidate,
  type CandidateInfo,
  type CandidateOutcome,
} from '../src/lib/candidate';
import type { Read } from '../src/lib/reads';
import type { Base, Cell } from '../src/lib/palette';
import { readNpyFloat32 } from '../src/lib/npy';

const FIXTURES_DIR = path.resolve('fixtures/match');
const MODEL_DIR = path.resolve('public/models/tfjs_dv_wgs_uint8');
const DRAFTS_DIR = path.resolve('drafts');
const FAILURES_OUT = path.join(DRAFTS_DIR, 'match-failures.json');

const CLASSES = ['hom_ref', 'het', 'hom_alt'] as const;
type Genotype = (typeof CLASSES)[number];

const SAMPLE_FLOATS = 100 * 221 * 7;
const POSITION_IN_WINDOW = 110;
const UINT8_NOISE_TOLERANCE = 0.05;

interface ManifestSampleMeta {
  index: number;
  chrom: string;
  position_genomic: number;
  position_1_based: number;
  ref_alleles: string;
  alt_alleles: string[];
  alt_indices: number[];
  emitted_alts: string[];
  primary_kind: 'snv' | 'del' | 'ins' | 'complex' | 'unknown';
  n_reads: number;
  dv_argmax: Genotype;
}

interface Manifest {
  n: number;
  bam: string;
  fasta: string;
  tfrecords: string[];
  shape: number[];
  samples: ManifestSampleMeta[];
}

interface SampleFile {
  index: number;
  chrom: string;
  position_genomic: number;
  position_in_window: number;
  ref_window_start: number;
  ref_window: string;
  ref_alleles: string;
  alt_alleles: string[];
  alt_indices: number[];
  emitted_alts: string[];
  primary_kind: string;
  reads: Array<{
    id: string;
    startCol: number;
    bases: string[];
    qualities: number[];
    strand: 'forward' | 'reverse';
    mapq: number;
    insertSize: number;
    row: number;
    insertions: Array<{
      offset: number;
      bases: string[];
      qualities: number[];
    }> | null;
  }>;
  dv_argmax: Genotype;
  dv_probs: { hom_ref: number; het: number; hom_alt: number };
}

type LevelStatus = 'pass' | 'fail' | 'skip';

interface FailureEntry {
  sampleIndex: number;
  level: 1 | 2 | 3;
  variant: string;
  expected: unknown;
  got: unknown;
  detail?: unknown;
}

function argmax3(probs: [number, number, number]): 0 | 1 | 2 {
  if (probs[1] > probs[0] && probs[1] >= probs[2]) return 1;
  if (probs[2] > probs[0] && probs[2] > probs[1]) return 2;
  return 0;
}

function readsFromSample(sample: SampleFile): Read[] {
  return sample.reads.map((r, i) => ({
    id: r.id,
    startCol: r.startCol,
    bases: r.bases as Cell[],
    qualities: new Uint8Array(r.qualities),
    strand: r.strand,
    mapq: r.mapq,
    insertSize: r.insertSize,
    row: i,
    insertions: r.insertions
      ? r.insertions.map((ins) => ({
          offset: ins.offset,
          bases: ins.bases as Base[],
          qualities: new Uint8Array(ins.qualities),
        }))
      : undefined,
  }));
}

/**
 * Build a Candidate for the encoder using DV's chosen alt for THIS example
 * (from emitted_alts). Returns null for combined multi-allelic (alt_indices
 * length > 1) or 'complex' kinds — those are skipped at L2/L3.
 */
function dvCandidateForExample(meta: ManifestSampleMeta): Candidate {
  if (meta.alt_indices.length !== 1) return null;
  const alt = meta.emitted_alts[0];
  if (!alt) return null;
  const ref = meta.ref_alleles;
  if (!ref) return null;

  // SNV
  if (ref.length === 1 && alt.length === 1) {
    return {
      kind: 'snv',
      refBase: ref[0] as Base,
      altBase: alt[0] as Base,
      supportingReads: 0,
      qualifyingReads: meta.n_reads,
    };
  }
  // Deletion: REF longer, REF starts with ALT (anchor base shared)
  if (ref.length > alt.length && alt && ref.startsWith(alt)) {
    return {
      kind: 'del',
      refBase: ref[0] as Base,
      supportingReads: 0,
      qualifyingReads: meta.n_reads,
    };
  }
  // Insertion: ALT longer, ALT starts with REF
  if (alt.length > ref.length && ref && alt.startsWith(ref)) {
    return {
      kind: 'ins',
      refBase: ref[0] as Base,
      altSequence: alt.slice(ref.length).split('') as Base[],
      supportingReads: 0,
      qualifyingReads: meta.n_reads,
    };
  }
  // complex
  return null;
}

/**
 * For L1 we don't gate on alt_indices — we ask: across all alts at this
 * position, does our engine's primary pick agree on (kind, alt) with any
 * of DV's alts?
 */
function level1Match(
  meta: ManifestSampleMeta,
  outcome: CandidateOutcome,
): { status: LevelStatus; reason?: string } {
  if (meta.primary_kind === 'complex') return { status: 'skip', reason: 'complex' };
  if (outcome.kind !== 'accepted') {
    return {
      status: 'fail',
      reason: `outcome=${outcome.kind} (DV called ${meta.primary_kind} ${meta.alt_alleles.join(',')})`,
    };
  }
  const info = outcome.info;
  // Build the set of (kind, alt-string) DV emitted at this position from
  // alt_alleles (not just emitted_alts — any alt at the position counts).
  const ourKey = candidateKey(info);
  for (const alt of meta.alt_alleles) {
    const key = altKey(meta.ref_alleles, alt);
    if (key === ourKey) return { status: 'pass' };
  }
  return {
    status: 'fail',
    reason: `our pick ${ourKey} not in DV alts ${meta.alt_alleles
      .map((a) => altKey(meta.ref_alleles, a))
      .join(', ')}`,
  };
}

function candidateKey(info: CandidateInfo): string {
  switch (info.kind) {
    case 'snv':
      return `snv:${info.altBase}`;
    case 'del':
      return `del`;
    case 'ins':
      return `ins:${info.altSequence.join('')}`;
  }
}

function altKey(ref: string, alt: string): string {
  if (ref.length === 1 && alt.length === 1) return `snv:${alt}`;
  if (ref.length > alt.length && alt && ref.startsWith(alt)) return `del`;
  if (alt.length > ref.length && ref && alt.startsWith(ref))
    return `ins:${alt.slice(ref.length)}`;
  return `complex:${ref}>${alt}`;
}

interface Level2Result {
  status: LevelStatus;
  channelDiffs?: Array<{ channel: string; ours: number[]; missing: number[] }>;
  reason?: string;
}

/**
 * L2 = our encoded tensor's per-channel value sets fall within the DV
 * allowed sets (validateEncodedTensor). This catches encoder bugs without
 * being thrown off by read-selection differences (DV picked different 95
 * reads than us).
 */
function level2Match(tensor: Float32Array): Level2Result {
  const report = validateEncodedTensor(tensor);
  if (report.passed) return { status: 'pass' };
  return {
    status: 'fail',
    reason: report.issues.join(' | '),
  };
}

interface Level3Result {
  status: LevelStatus;
  ourArgmax: Genotype;
  goldenArgmax: Genotype;
  ourProbs: [number, number, number];
  goldenProbs: [number, number, number];
  uintNoise: boolean;
  maxDelta: number;
}

async function level3Match(
  model: tf.LayersModel,
  ourTensor: Float32Array,
  goldenProbs: [number, number, number],
): Promise<Level3Result> {
  const x = tf.tensor4d(ourTensor, [1, 100, 221, 7]);
  const y = model.predict(x) as tf.Tensor;
  const flat = (await y.data()) as Float32Array;
  x.dispose();
  y.dispose();
  const ourProbs: [number, number, number] = [flat[0], flat[1], flat[2]];
  const ourArgmax = CLASSES[argmax3(ourProbs)];
  const goldenArgmax = CLASSES[argmax3(goldenProbs)];
  const maxDelta = Math.max(
    Math.abs(ourProbs[0] - goldenProbs[0]),
    Math.abs(ourProbs[1] - goldenProbs[1]),
    Math.abs(ourProbs[2] - goldenProbs[2]),
  );
  if (ourArgmax === goldenArgmax) {
    return {
      status: 'pass',
      ourArgmax,
      goldenArgmax,
      ourProbs,
      goldenProbs,
      uintNoise: false,
      maxDelta,
    };
  }
  if (maxDelta < UINT8_NOISE_TOLERANCE) {
    return {
      status: 'pass',
      ourArgmax,
      goldenArgmax,
      ourProbs,
      goldenProbs,
      uintNoise: true,
      maxDelta,
    };
  }
  return {
    status: 'fail',
    ourArgmax,
    goldenArgmax,
    ourProbs,
    goldenProbs,
    uintNoise: false,
    maxDelta,
  };
}

function pct(num: number, denom: number): string {
  if (denom === 0) return '—';
  return ((num / denom) * 100).toFixed(1) + '%';
}

async function main(): Promise<void> {
  console.log(`Loading manifest from ${FIXTURES_DIR}…`);
  const manifest = JSON.parse(
    readFileSync(path.join(FIXTURES_DIR, 'manifest.json'), 'utf-8'),
  ) as Manifest;
  console.log(
    `  ${manifest.n} samples · BAM=${manifest.bam} · shape=${manifest.shape.join('×')}`,
  );

  console.log('Loading golden outputs…');
  // pileups.npy.gz ships in fixtures/match/ for reference + debugging but
  // is not loaded by this CLI — L2 uses validateEncodedTensor against
  // GOLDEN_CHANNEL_RANGES (cited in pileup-encoder.ts), which catches
  // encoder bugs without loading 50 MB of golden tensors per run.
  const outputsBuf = readFileSync(path.join(FIXTURES_DIR, 'outputs.npy'));
  const goldenOutputs = readNpyFloat32(toAB(outputsBuf));

  console.log(`Loading model from ${MODEL_DIR}…`);
  const model = await tf.loadLayersModel(`file://${MODEL_DIR}/model.json`);

  let l1Pass = 0,
    l1Fail = 0,
    l1Skip = 0;
  let l2Pass = 0,
    l2Fail = 0,
    l2Skip = 0;
  let l3Pass = 0,
    l3Fail = 0,
    l3Skip = 0,
    l3Noise = 0;
  const failures: FailureEntry[] = [];
  const failureLines: string[] = [];

  for (const meta of manifest.samples) {
    const sampleFile = JSON.parse(
      readFileSync(
        path.join(FIXTURES_DIR, `sample_${meta.index}.json`),
        'utf-8',
      ),
    ) as SampleFile;
    const reads = readsFromSample(sampleFile);
    const reference = sampleFile.ref_window.split('') as Base[];
    const variantStr = `${meta.chrom}:${meta.position_1_based} ${meta.ref_alleles}>${meta.alt_alleles.join(',')} idx=${JSON.stringify(meta.alt_indices)}`;

    // L1
    const outcome = deriveCandidateOutcome(reads, reference, POSITION_IN_WINDOW);
    const l1 = level1Match(meta, outcome);
    if (l1.status === 'pass') l1Pass++;
    else if (l1.status === 'skip') l1Skip++;
    else {
      l1Fail++;
      failures.push({
        sampleIndex: meta.index,
        level: 1,
        variant: variantStr,
        expected: { kind: meta.primary_kind, alts: meta.alt_alleles },
        got: outcome,
        detail: l1.reason,
      });
      failureLines.push(`  L1: sample ${meta.index}  ${variantStr}  ${l1.reason}`);
    }

    // L2 / L3 require a single-alt example AND a non-complex kind
    const dvCand = dvCandidateForExample(meta);
    if (!dvCand) {
      l2Skip++;
      l3Skip++;
      continue;
    }

    const ourTensor = encodePileup(reads, reference, POSITION_IN_WINDOW, dvCand);
    if (!ourTensor) {
      l2Skip++;
      l3Skip++;
      continue;
    }

    // L2
    const l2 = level2Match(ourTensor);
    if (l2.status === 'pass') l2Pass++;
    else if (l2.status === 'skip') l2Skip++;
    else {
      l2Fail++;
      failures.push({
        sampleIndex: meta.index,
        level: 2,
        variant: variantStr,
        expected: 'within DV allowed value sets',
        got: l2.reason,
      });
      failureLines.push(`  L2: sample ${meta.index}  ${variantStr}  ${l2.reason}`);
    }

    // L3
    const goldenProbsTuple: [number, number, number] = [
      goldenOutputs.data[meta.index * 3],
      goldenOutputs.data[meta.index * 3 + 1],
      goldenOutputs.data[meta.index * 3 + 2],
    ];
    const l3 = await level3Match(model, ourTensor, goldenProbsTuple);
    if (l3.status === 'pass') {
      l3Pass++;
      if (l3.uintNoise) l3Noise++;
    } else if (l3.status === 'skip') {
      l3Skip++;
    } else {
      l3Fail++;
      failures.push({
        sampleIndex: meta.index,
        level: 3,
        variant: variantStr,
        expected: { argmax: l3.goldenArgmax, probs: l3.goldenProbs },
        got: { argmax: l3.ourArgmax, probs: l3.ourProbs },
        detail: { maxDelta: l3.maxDelta },
      });
      failureLines.push(
        `  L3: sample ${meta.index}  ${variantStr}  ours=${l3.ourArgmax} dv=${l3.goldenArgmax} Δ=${l3.maxDelta.toFixed(3)}`,
      );
    }
  }

  model.dispose();

  const totalNonSkippedL1 = manifest.n - l1Skip;
  const totalNonSkippedL2 = manifest.n - l2Skip;
  const totalNonSkippedL3 = manifest.n - l3Skip;

  console.log('');
  console.log(
    `  Candidate match  : ${l1Pass}/${totalNonSkippedL1}  (${pct(l1Pass, totalNonSkippedL1)})  [skipped ${l1Skip}]`,
  );
  console.log(
    `  Encoder match    : ${l2Pass}/${totalNonSkippedL2}  (${pct(l2Pass, totalNonSkippedL2)})  [skipped ${l2Skip}]`,
  );
  const noiseSuffix = l3Noise > 0 ? `  [+${l3Noise} within uint8 noise]` : '';
  console.log(
    `  Prediction match : ${l3Pass}/${totalNonSkippedL3}  (${pct(l3Pass, totalNonSkippedL3)})  [skipped ${l3Skip}]${noiseSuffix}`,
  );

  if (failureLines.length > 0) {
    console.log('\nFailures:');
    for (const line of failureLines) console.log(line);
  }

  mkdirSync(DRAFTS_DIR, { recursive: true });
  writeFileSync(FAILURES_OUT, JSON.stringify(failures, null, 2));
  console.log(`\nWrote ${FAILURES_OUT} (${failures.length} entries)`);
}

function toAB(buf: Buffer): ArrayBuffer {
  return buf.buffer.slice(
    buf.byteOffset,
    buf.byteOffset + buf.byteLength,
  ) as ArrayBuffer;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
