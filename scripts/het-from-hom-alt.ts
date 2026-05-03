/**
 * Take a real golden hom_alt tensor (sample 0) and synthetically convert
 * it to look like a het: flip ~half of the supports_variant=254 rows to
 * supports_variant=152, AND flip the read_base + differs_from_ref at the
 * predict column for those rows from alt → ref. If the model still says
 * hom_alt, the model is not calibrated to predict het on this kind of
 * input.
 */
import { gunzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';

const W = 221;
const H = 100;
const C = 7;
const PREDICT_COL = 110;
const SAMPLE = H * W * C;

async function main() {
  const gz = readFileSync('/tmp/pileups.npy.gz');
  const buf = gunzipSync(gz);
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  const headerLen = new DataView(ab).getUint16(8, true);
  const dataStart = 10 + headerLen;
  const u8 = new Uint8Array(ab.slice(dataStart));

  // Load model
  const modelDir = process.env.MODEL_FP32
    ? '/root/dv-tfjs/out/tfjs_dv_wgs'
    : path.resolve('public/models/tfjs_dv_wgs_uint8');
  const ioHandler: tf.io.IOHandler = {
    load: async () => {
      const mj = JSON.parse(readFileSync(path.join(modelDir, 'model.json'), 'utf-8'));
      const specs: unknown[] = [];
      const chunks: ArrayBuffer[] = [];
      for (const g of mj.weightsManifest) {
        for (const w of g.weights) specs.push(w);
        for (const p of g.paths) {
          const b = readFileSync(path.join(modelDir, p));
          chunks.push(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
        }
      }
      const total = chunks.reduce((a, c) => a + c.byteLength, 0);
      const data = new Uint8Array(total);
      let off = 0;
      for (const c of chunks) { data.set(new Uint8Array(c), off); off += c.byteLength; }
      return {
        modelTopology: mj.modelTopology,
        format: mj.format,
        weightSpecs: specs as tf.io.WeightsManifestEntry[],
        weightData: data.buffer,
      };
    },
  };
  await tf.setBackend('cpu');
  const model = await tf.loadLayersModel(ioHandler);

  async function predict(label: string, t: Float32Array) {
    const x = tf.tensor4d(t, [1, H, W, C]);
    const y = model.predict(x) as tf.Tensor;
    const p = await y.data() as Float32Array;
    console.log(`  ${label.padEnd(50)} hom_ref=${p[0].toFixed(4)} het=${p[1].toFixed(4)} hom_alt=${p[2].toFixed(4)}`);
    x.dispose(); y.dispose();
  }

  // Sample 0: NA12878 chr20:10000117 G>A hom_alt
  const orig = Float32Array.from(u8.slice(0, SAMPLE));
  await predict('original golden sample 0 (hom_alt)', orig);

  // Find rows with supports_variant=254 (alt-supporters)
  const altRows: number[] = [];
  for (let r = 0; r < H; r++) {
    const v = orig[r * W * C + PREDICT_COL * C + 4];
    if (v === 254) altRows.push(r);
  }
  console.log(`  found ${altRows.length} alt-supporting rows`);

  // Pick refBase (G) and altBase (A) intensities for sample 0
  const refIntensity = orig[0 * W * C + PREDICT_COL * C + 0]; // ref-row read_base at predict col
  console.log(`  ref intensity at predict col (from ref row 0):`, refIntensity);

  // Probe: flip ALL alt rows to ref-supporters — should give hom_ref
  {
    const t = orig.slice();
    for (const row of altRows) {
      for (let col = 0; col < W; col++) {
        const base = row * W * C + col * C;
        if (t[base + 4] === 254) t[base + 4] = 152; // supports_variant YES → NO
      }
      const cellRb = row * W * C + PREDICT_COL * C + 0;
      const cellDfr = row * W * C + PREDICT_COL * C + 5;
      t[cellRb] = refIntensity;
      t[cellDfr] = 50;
    }
    await predict('flipped ALL alt→ref (should be hom_ref)', t);
  }

  // Same but only at the predict column (more minimal change)
  {
    const t = orig.slice();
    for (const row of altRows) {
      const cellRb = row * W * C + PREDICT_COL * C + 0;
      const cellSv = row * W * C + PREDICT_COL * C + 4;
      const cellDfr = row * W * C + PREDICT_COL * C + 5;
      t[cellRb] = refIntensity;
      t[cellSv] = 152;
      t[cellDfr] = 50;
    }
    await predict('flipped ALL alt only AT predict col (per-cell)', t);
  }

  // FLIP: convert ~half of alt rows to ref-supporting rows
  for (const flipRatio of [0.3, 0.5, 0.7]) {
    const t = orig.slice();
    const nFlip = Math.floor(altRows.length * flipRatio);
    for (let i = 0; i < nFlip; i++) {
      const row = altRows[i];
      // Flip read_base at predict col from alt (G=180? no — for sample 0 ref=G, alt=A; alt support means row's read_base at col 110 = A=250)
      // Actually let me just check the actual value in the row at predict col
      const cellRb = row * W * C + PREDICT_COL * C + 0;
      const cellSv = row * W * C + PREDICT_COL * C + 4;
      const cellDfr = row * W * C + PREDICT_COL * C + 5;
      // Read base at predict for an alt-supporting row in sample 0 should be 250 (A)
      // Flip to ref base intensity (G=180)
      t[cellRb] = refIntensity;
      // Flip ALL columns of supports_variant for this row to NO (152)
      for (let col = 0; col < W; col++) {
        const idx = row * W * C + col * C + 4;
        if (t[idx] !== 0) t[idx] = 152;
      }
      // Flip differs_from_ref at predict col from YES (254) to NO (50)
      t[cellDfr] = 50;
    }
    await predict(`flipped ${nFlip}/${altRows.length} alt→ref rows`, t);
  }
  model.dispose();
}

main().catch((e) => { console.error(e); process.exit(1); });
