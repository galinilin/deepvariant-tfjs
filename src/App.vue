<script setup lang="ts">
import { ref } from 'vue';
import { DeepVariantModel } from './lib/DeepVariantModel';
import { readNpyFloat32 } from './lib/npy';

type Status = 'idle' | 'loading' | 'predicting' | 'ok' | 'err';

const status = ref<Status>('idle');
const log = ref<string[]>([]);
const result = ref<{
  backend: string;
  params: number;
  precision: string;
  maxDiff: number;
  meanDiff: number;
  ms: number;
  pass: boolean;
} | null>(null);

const UINT8_TOL = 0.2;
const BASE = import.meta.env.BASE_URL;

function append(line: string) {
  log.value = [...log.value, line];
}

async function fetchBuf(url: string): Promise<ArrayBuffer> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url} -> ${r.status}`);
  return r.arrayBuffer();
}

async function runSmokeTest() {
  status.value = 'loading';
  log.value = [];
  result.value = null;
  try {
    append(`loading uint8 model from ${BASE}models/tfjs_dv_wgs_uint8/...`);
    const dv = await DeepVariantModel.load({
      precision: 'uint8',
      modelBaseUrl: `${BASE}models/`,
      onProgress: (f) => append(`  download ${(f * 100).toFixed(0)}%`),
    });
    append(`loaded: backend=${dv.backend}  params=${dv.countParams().toLocaleString()}`);

    append(`fetching parity fixtures from ${BASE}fixtures/...`);
    const [inputsBuf, expectedBuf] = await Promise.all([
      fetchBuf(`${BASE}fixtures/parity_inputs.npy`),
      fetchBuf(`${BASE}fixtures/parity_outputs.npy`),
    ]);
    const inputs = readNpyFloat32(inputsBuf);
    const expected = readNpyFloat32(expectedBuf);
    const batch = inputs.shape[0];
    append(`inputs shape=[${inputs.shape.join(',')}]  expected shape=[${expected.shape.join(',')}]`);

    status.value = 'predicting';
    const t0 = performance.now();
    const preds = await dv.predictBatch({ data: inputs.data, batch });
    const ms = performance.now() - t0;

    let maxDiff = 0;
    let sumDiff = 0;
    let n = 0;
    for (let i = 0; i < preds.length; i++) {
      const p = preds[i];
      const probs = [p.probs.hom_ref, p.probs.het, p.probs.hom_alt];
      for (let c = 0; c < 3; c++) {
        const d = Math.abs(probs[c] - expected.data[i * 3 + c]);
        if (d > maxDiff) maxDiff = d;
        sumDiff += d;
        n++;
      }
    }
    const meanDiff = sumDiff / n;
    const pass = maxDiff <= UINT8_TOL;
    result.value = {
      backend: dv.backend,
      params: dv.countParams(),
      precision: dv.precision,
      maxDiff,
      meanDiff,
      ms,
      pass,
    };
    append(
      `inference: ${ms.toFixed(0)}ms  max|diff|=${maxDiff.toExponential(2)}  mean|diff|=${meanDiff.toExponential(2)}`,
    );
    append(pass ? `PASS (max|diff| <= ${UINT8_TOL})` : `FAIL (max|diff| > ${UINT8_TOL})`);
    status.value = pass ? 'ok' : 'err';
    dv.dispose();
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    append(`ERROR: ${msg}`);
    status.value = 'err';
  }
}
</script>

<template>
  <main>
    <h1>dv-browser</h1>
    <p class="meta">
      DeepVariant 1.8 WGS InceptionV3 in TensorFlow.js. Click below to confirm
      the bundled <code>DeepVariantModel</code> class still produces the same
      numbers as the dv-tfjs verification suite.
    </p>

    <button :disabled="status === 'loading' || status === 'predicting'" @click="runSmokeTest">
      {{ status === 'idle' || status === 'ok' || status === 'err'
        ? 'Run parity smoke test'
        : status === 'loading' ? 'Loading model...' : 'Predicting...' }}
    </button>

    <pre class="log">{{ log.join('\n') || 'idle.' }}</pre>

    <pre v-if="result" :class="['result', result.pass ? 'ok' : 'bad']">{{
      `${result.pass ? 'OK' : 'FAIL'}  precision=${result.precision}  backend=${result.backend}
params=${result.params.toLocaleString()}
max|diff|=${result.maxDiff.toExponential(3)}  mean|diff|=${result.meanDiff.toExponential(3)}
inference=${result.ms.toFixed(0)}ms`
    }}</pre>
  </main>
</template>

<style>
:root { color-scheme: dark; }
body { margin: 0; background: #0b0d10; color: #e6e6e6; font-family: ui-monospace, Menlo, monospace; }
main { max-width: 880px; padding: 24px; margin: 0 auto; }
h1 { font-size: 1.1rem; margin: 0 0 8px; }
.meta { color: #888; line-height: 1.5; margin: 0 0 16px; }
button { padding: 8px 14px; background: #1f2a37; color: #e6e6e6; border: 1px solid #334155; border-radius: 6px; cursor: pointer; font: inherit; }
button:disabled { opacity: 0.6; cursor: progress; }
.log { background: #111418; padding: 12px; border-radius: 6px; min-height: 80px; white-space: pre-wrap; margin-top: 16px; }
.result { padding: 12px; border-radius: 6px; margin-top: 12px; }
.result.ok  { background: #0f2a1c; color: #6cf09c; border: 1px solid #1d5a3a; }
.result.bad { background: #2a0f12; color: #f08080; border: 1px solid #5a1d23; }
code { background: #1f2937; padding: 1px 5px; border-radius: 3px; }
</style>
