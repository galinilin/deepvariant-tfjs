/**
 * Generate a synthetic world, encode the pileup at a het scenario, and
 * print per-channel statistics at the predict column. Used to debug why
 * v3.2's predictions are biased toward hom_alt.
 *
 * Run: npx tsx scripts/debug-encode.ts
 */
import { buildReference, defaultWindowStart, WINDOW_LENGTH } from '../src/lib/reference';
import { buildReads, makeRng } from '../src/lib/reads';
import { placeScenarios } from '../src/lib/scenarios';
import { deriveCandidateOutcome } from '../src/lib/candidate';
import { encodePileup, validateEncodedTensor } from '../src/lib/pileup-encoder';

const SEED = 42;
const READ_COUNT = parseInt(process.argv[2] ?? '250', 10);
console.log('using READ_COUNT:', READ_COUNT);
const rng = makeRng(SEED);
const reference = buildReference(undefined, SEED);
const scenarios = placeScenarios(reference, rng);
console.log('reference length:', reference.length);
console.log('scenarios:', scenarios.map((s) => `${s.type}@${s.position}`).join(', '));

const reads = buildReads(reference, scenarios, rng, READ_COUNT);
console.log('reads count:', reads.length);

// Pick a het scenario
const het = scenarios.find((s) => s.type === 'het') ?? scenarios[0];
console.log('\nfocusing on:', het.type, '@', het.position, 'altBase:', het.altBase);

const outcome = deriveCandidateOutcome(reads, reference, het.position);
console.log('outcome.kind:', outcome.kind);
if (outcome.kind === 'accepted') {
  const c = outcome.info;
  console.log('candidate:', c.kind, 'ref=', c.refBase,
    'alt=', c.kind === 'snv' ? c.altBase : c.kind === 'ins' ? '+' + c.altSequence.join('') : 'del',
    'support', c.supportingReads + '/' + c.qualifyingReads);

  const tensor = encodePileup(reads, reference, het.position, c);
  if (!tensor) {
    console.log('encoder returned null');
    process.exit(1);
  }

  // Per-channel stats at predict column
  const W = 221;
  const C = 7;
  const PREDICT_COL = 110;
  const names = ['read_base', 'base_quality', 'mapping_quality', 'strand',
                 'supports_variant', 'differs_from_ref', 'insert_size'];
  console.log('\nPER-CHANNEL @ predict column (col 110):');
  for (let ch = 0; ch < C; ch++) {
    const counts = new Map<number, number>();
    for (let row = 0; row < 100; row++) {
      const v = Math.round(tensor[row * W * C + PREDICT_COL * C + ch]);
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]);
    console.log(`  [${ch}] ${names[ch].padEnd(20)}:`, sorted.map(([v, n]) => `${v}×${n}`).join(', '));
  }

  // Total signal across whole image, just per channel
  console.log('\nWHOLE-IMAGE PER-CHANNEL HISTOGRAM (top 5):');
  for (let ch = 0; ch < C; ch++) {
    const counts = new Map<number, number>();
    for (let i = ch; i < tensor.length; i += C) {
      const v = Math.round(tensor[i]);
      counts.set(v, (counts.get(v) ?? 0) + 1);
    }
    const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 5);
    console.log(`  [${ch}] ${names[ch].padEnd(20)}:`, sorted.map(([v, n]) => `${v}×${n}`).join(', '));
  }

  // Validate
  const validation = validateEncodedTensor(tensor);
  console.log('\nvalidation passed:', validation.passed);
  if (!validation.passed) {
    for (const issue of validation.issues) console.log('  issue:', issue);
  }

  // Run inference using browser-compatible @tensorflow/tfjs in Node
  // (CPU backend; slower than tfjs-node but functional and we already
  // depend on it).
  console.log('\n[predict] loading model + running inference…');
  const tf = await import('@tensorflow/tfjs');
  const path = await import('node:path');
  const fs = await import('node:fs');

  // Use float32 model if MODEL_FP32=1; otherwise the bundled uint8.
  const modelDir = process.env.MODEL_FP32
    ? '/root/dv-tfjs/out/tfjs_dv_wgs'
    : path.resolve('public/models/tfjs_dv_wgs_uint8');
  console.log('using model:', modelDir);
  // tfjs needs a custom IOHandler for node fs (it expects fetch by default).
  const ioHandler: import('@tensorflow/tfjs').io.IOHandler = {
    load: async () => {
      const modelJsonPath = path.join(modelDir, 'model.json');
      const modelJson = JSON.parse(fs.readFileSync(modelJsonPath, 'utf-8'));
      const weightsManifest = modelJson.weightsManifest as Array<{
        paths: string[];
        weights: unknown[];
      }>;
      const weightSpecs: unknown[] = [];
      const weightChunks: ArrayBuffer[] = [];
      for (const group of weightsManifest) {
        for (const w of group.weights) weightSpecs.push(w);
        for (const p of group.paths) {
          const buf = fs.readFileSync(path.join(modelDir, p));
          weightChunks.push(
            buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer,
          );
        }
      }
      const totalLen = weightChunks.reduce((a, c) => a + c.byteLength, 0);
      const weightData = new Uint8Array(totalLen);
      let off = 0;
      for (const c of weightChunks) {
        weightData.set(new Uint8Array(c), off);
        off += c.byteLength;
      }
      return {
        modelTopology: modelJson.modelTopology,
        format: modelJson.format,
        generatedBy: modelJson.generatedBy,
        convertedBy: modelJson.convertedBy,
        weightSpecs: weightSpecs as import('@tensorflow/tfjs').io.WeightsManifestEntry[],
        weightData: weightData.buffer,
        userDefinedMetadata: modelJson.userDefinedMetadata,
      };
    },
  };
  await tf.setBackend('cpu');
  const model = await tf.loadLayersModel(ioHandler);

  async function predictWith(label: string, t: Float32Array) {
    const x = tf.tensor4d(t, [1, 100, 221, 7]);
    const y = model.predict(x) as import('@tensorflow/tfjs').Tensor;
    const p = (await y.data()) as Float32Array;
    console.log(
      `  ${label.padEnd(40)}  hom_ref=${p[0].toFixed(4)} het=${p[1].toFixed(4)} hom_alt=${p[2].toFixed(4)}`,
    );
    x.dispose();
    y.dispose();
  }

  await predictWith('original encoded', tensor);

  // Probe 1: zero out supports_variant entirely (no read supports variant)
  const t2 = tensor.slice();
  for (let i = 4; i < t2.length; i += 7) t2[i] = 0;
  await predictWith('supports_variant all 0 (empty)', t2);

  // Probe 2: all supports_variant = NO (152) for non-empty rows (where read_base != 0)
  const t3 = tensor.slice();
  for (let row = 0; row < 100; row++) {
    let nonEmpty = false;
    for (let col = 0; col < 221; col++) {
      if (tensor[(row * 221 + col) * 7] !== 0) { nonEmpty = true; break; }
    }
    if (nonEmpty) {
      for (let col = 0; col < 221; col++) {
        const idx = (row * 221 + col) * 7 + 4;
        if (tensor[idx] !== 0) t3[idx] = 152;
      }
    }
  }
  await predictWith('supports_variant flipped to all 152', t3);

  // Probe 3: all supports_variant = YES (254) for non-empty rows
  const t4 = tensor.slice();
  for (let row = 0; row < 100; row++) {
    for (let col = 0; col < 221; col++) {
      const idx = (row * 221 + col) * 7 + 4;
      if (tensor[idx] !== 0) t4[idx] = 254;
    }
  }
  await predictWith('supports_variant flipped to all 254', t4);

  // Probe 4: zero entire image (sanity)
  const t5 = new Float32Array(tensor.length);
  await predictWith('all zeros', t5);

  // Probe 5: zero out differs_from_ref entirely
  const t6 = tensor.slice();
  for (let i = 5; i < t6.length; i += 7) t6[i] = 0;
  await predictWith('differs_from_ref all 0', t6);

  // Probe 6: zero ref rows (only read rows remain)
  const t7 = tensor.slice();
  for (let row = 0; row < 5; row++) {
    for (let i = 0; i < 221 * 7; i++) t7[row * 221 * 7 + i] = 0;
  }
  await predictWith('ref rows zeroed', t7);

  // Probe 7: zero channel 4 in TOP half rows (mirrors dv-tfjs ablation C)
  const t8 = tensor.slice();
  for (let row = 0; row < 50; row++) {
    for (let col = 0; col < 221; col++) t8[(row * 221 + col) * 7 + 4] = 0;
  }
  await predictWith('top 50 rows ch4 zeroed', t8);

  // Probe: keep only first N read rows
  for (const keepN of [10, 15, 20, 30]) {
    const t = tensor.slice();
    for (let row = 5 + keepN; row < 100; row++) {
      for (let i = 0; i < 221 * 7; i++) t[row * 221 * 7 + i] = 0;
    }
    await predictWith(`keep only first ${keepN} read rows`, t);
  }

  // Probe: shuffle the read rows (DV uses different row order than ours)
  {
    const t = tensor.slice();
    const rowOrder: number[] = [];
    for (let row = 5; row < 100; row++) rowOrder.push(row);
    // Fisher-Yates shuffle
    for (let i = rowOrder.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [rowOrder[i], rowOrder[j]] = [rowOrder[j], rowOrder[i]];
    }
    for (let newIdx = 0; newIdx < rowOrder.length; newIdx++) {
      const oldRow = rowOrder[newIdx];
      const newRow = 5 + newIdx;
      if (oldRow === newRow) continue;
      for (let col = 0; col < 221; col++) {
        for (let ch = 0; ch < 7; ch++) {
          t[(newRow * 221 + col) * 7 + ch] = tensor[(oldRow * 221 + col) * 7 + ch];
        }
      }
    }
    await predictWith('rows 5-99 shuffled', t);
  }

  // Probe: pack supporting rows to BOTTOM half
  {
    const t = new Float32Array(tensor.length);
    // copy ref rows
    for (let row = 0; row < 5; row++) {
      for (let i = 0; i < 221 * 7; i++) {
        t[row * 221 * 7 + i] = tensor[row * 221 * 7 + i];
      }
    }
    const supportingRows: number[] = [];
    const nonSupportingRows: number[] = [];
    for (let row = 5; row < 100; row++) {
      const v = tensor[(row * 221 + 110) * 7 + 4];
      if (v === 254) supportingRows.push(row);
      else if (v === 152) nonSupportingRows.push(row);
    }
    // Place non-supporting rows first (5..), then supporting rows (then empties stay)
    let dest = 5;
    for (const src of nonSupportingRows) {
      for (let i = 0; i < 221 * 7; i++) {
        t[dest * 221 * 7 + i] = tensor[src * 221 * 7 + i];
      }
      dest++;
    }
    for (const src of supportingRows) {
      for (let i = 0; i < 221 * 7; i++) {
        t[dest * 221 * 7 + i] = tensor[src * 221 * 7 + i];
      }
      dest++;
    }
    await predictWith('NO rows first, YES rows last', t);
  }

  // Try zeroing channel 4 only on rows that actually have a read (excluding empties)
  // for the full set of reads
  for (const fracZero of [0.25, 0.5, 0.75]) {
    const t = tensor.slice();
    const rowsWithReads: number[] = [];
    for (let row = 5; row < 100; row++) {
      const v = tensor[(row * 221 + 110) * 7 + 4];
      if (v === 254 || v === 152) rowsWithReads.push(row);
    }
    const nZero = Math.floor(rowsWithReads.length * fracZero);
    for (let i = 0; i < nZero; i++) {
      const row = rowsWithReads[i];
      for (let col = 0; col < 221; col++) t[(row * 221 + col) * 7 + 4] = 0;
    }
    await predictWith(`zero ch4 on ${nZero}/${rowsWithReads.length} read-rows`, t);
  }

  // Probe 8: zero channel 4 only on NON-supporting rows
  const t9 = tensor.slice();
  for (let row = 0; row < 100; row++) {
    let isNo = false;
    for (let col = 0; col < 221; col++) {
      const v = tensor[(row * 221 + col) * 7 + 4];
      if (v === 152) { isNo = true; break; }
      if (v === 254) break;
    }
    if (isNo) {
      for (let col = 0; col < 221; col++) t9[(row * 221 + col) * 7 + 4] = 0;
    }
  }
  await predictWith('non-supporting rows ch4 zeroed', t9);

  model.dispose();
}
