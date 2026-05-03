/**
 * Sanity check: feed a known-good golden DV tensor through our TFJS model
 * and check argmax. This isolates "is the model loaded correctly" from
 * "are our synthetic inputs the issue."
 */
import { gunzipSync } from 'node:zlib';
import { readFileSync } from 'node:fs';
import path from 'node:path';
import * as tf from '@tensorflow/tfjs';
import { readNpyFloat32 } from '../src/lib/npy';

const W = 221;
const H = 100;
const C = 7;
const SAMPLE = H * W * C;

async function main() {
  const pileupsGz = readFileSync('/tmp/pileups.npy.gz');
  const pileupsBuf = gunzipSync(pileupsGz);
  const ab = pileupsBuf.buffer.slice(
    pileupsBuf.byteOffset,
    pileupsBuf.byteOffset + pileupsBuf.byteLength,
  ) as ArrayBuffer;
  // Parse uint8 npy header
  const view = new DataView(ab);
  const magic = String.fromCharCode(...new Uint8Array(ab, 0, 6));
  if (magic !== '\x93NUMPY') throw new Error('not npy');
  const major = view.getUint8(6);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const dataStart = (major === 1 ? 10 : 12) + headerLen;
  const header = new TextDecoder().decode(
    new Uint8Array(ab, major === 1 ? 10 : 12, headerLen),
  );
  console.log('npy header:', header.trim());
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/)!;
  const shape = shapeMatch[1].split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  console.log('shape:', shape);
  const u8 = new Uint8Array(ab.slice(dataStart));
  const pileupsF32 = Float32Array.from(u8);
  console.log('total samples:', shape[0]);

  // Outputs
  const outputsBuf = readFileSync('/tmp/outputs.npy');
  const outputsAB = outputsBuf.buffer.slice(
    outputsBuf.byteOffset,
    outputsBuf.byteOffset + outputsBuf.byteLength,
  ) as ArrayBuffer;
  const outputs = readNpyFloat32(outputsAB);

  // Load TFJS model
  const modelDir = path.resolve('public/models/tfjs_dv_wgs_uint8');
  const ioHandler: tf.io.IOHandler = {
    load: async () => {
      const modelJsonPath = path.join(modelDir, 'model.json');
      const modelJson = JSON.parse(readFileSync(modelJsonPath, 'utf-8'));
      const wm = modelJson.weightsManifest;
      const specs: unknown[] = [];
      const chunks: ArrayBuffer[] = [];
      for (const g of wm) {
        for (const w of g.weights) specs.push(w);
        for (const p of g.paths) {
          const b = readFileSync(path.join(modelDir, p));
          chunks.push(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer);
        }
      }
      const total = chunks.reduce((a, c) => a + c.byteLength, 0);
      const data = new Uint8Array(total);
      let off = 0;
      for (const c of chunks) {
        data.set(new Uint8Array(c), off);
        off += c.byteLength;
      }
      return {
        modelTopology: modelJson.modelTopology,
        format: modelJson.format,
        generatedBy: modelJson.generatedBy,
        convertedBy: modelJson.convertedBy,
        weightSpecs: specs as tf.io.WeightsManifestEntry[],
        weightData: data.buffer,
      };
    },
  };
  await tf.setBackend('cpu');
  const model = await tf.loadLayersModel(ioHandler);

  console.log('\nrunning predictions on each golden sample...');
  for (let i = 0; i < shape[0]; i++) {
    const t = pileupsF32.slice(i * SAMPLE, (i + 1) * SAMPLE);
    const x = tf.tensor4d(t, [1, H, W, C]);
    const y = model.predict(x) as tf.Tensor;
    const p = (await y.data()) as Float32Array;
    const dv = [outputs.data[i * 3], outputs.data[i * 3 + 1], outputs.data[i * 3 + 2]];
    const ourArgmax = p[0] > p[1] && p[0] > p[2] ? 'hom_ref' : p[1] > p[2] ? 'het' : 'hom_alt';
    const dvArgmax = dv[0] > dv[1] && dv[0] > dv[2] ? 'hom_ref' : dv[1] > dv[2] ? 'het' : 'hom_alt';
    const match = ourArgmax === dvArgmax ? '✓' : '✗';
    console.log(
      `  sample ${i.toString().padStart(2)}: ours=${ourArgmax.padEnd(7)} (${p[0].toFixed(3)},${p[1].toFixed(3)},${p[2].toFixed(3)})  dv=${dvArgmax.padEnd(7)} (${dv[0].toFixed(3)},${dv[1].toFixed(3)},${dv[2].toFixed(3)}) ${match}`,
    );
    x.dispose();
    y.dispose();
    if (i >= 9) break;
  }
  model.dispose();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
