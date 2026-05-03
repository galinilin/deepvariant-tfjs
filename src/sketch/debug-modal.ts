/**
 * Floating debug overlay. Toggled by 'd' key (or the Debug corner
 * button). Shows live information about:
 *   - TFJS backend (webgl/cpu) + GPU vendor
 *   - Sandbox state (candidate, n_reads, n_scenarios, predictPos)
 *   - Last prediction probabilities + latency
 *   - Encoder validation report (validateEncodedTensor on current tensor)
 *
 * The modal subscribes to a per-frame requestAnimationFrame loop while
 * visible and stops polling when hidden, so it costs nothing when off.
 */
import * as tf from '@tensorflow/tfjs';
import { sandboxState } from '../lib/sandbox-state';
import {
  encodePileup,
  validateEncodedTensor,
  type ValidationReport,
} from '../lib/pileup-encoder';
import { debugTelemetry } from '../lib/debug-telemetry';

const N_CHANNELS = 7;
const CHANNEL_NAMES = [
  'read_base',
  'base_quality',
  'mapping_quality',
  'strand',
  'supports_variant',
  'differs_from_ref',
  'insert_size',
];

interface GpuInfo {
  backend: string;
  webglVendor: string | null;
  webglRenderer: string | null;
}

function detectGpu(): GpuInfo {
  const backend = tf.getBackend() ?? 'unknown';
  let vendor: string | null = null;
  let renderer: string | null = null;
  try {
    const canvas = document.createElement('canvas');
    const gl =
      (canvas.getContext('webgl2') as WebGL2RenderingContext | null) ??
      (canvas.getContext('webgl') as WebGLRenderingContext | null);
    if (gl) {
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      if (ext) {
        vendor = gl.getParameter(ext.UNMASKED_VENDOR_WEBGL) as string;
        renderer = gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) as string;
      } else {
        vendor = gl.getParameter(gl.VENDOR) as string;
        renderer = gl.getParameter(gl.RENDERER) as string;
      }
    }
  } catch {
    /* ignore */
  }
  return { backend, webglVendor: vendor, webglRenderer: renderer };
}

function esc(s: string): string {
  return s.replace(/[&<>]/g, (c) =>
    c === '&' ? '&amp;' : c === '<' ? '&lt;' : '&gt;',
  );
}

function rowHtml(k: string, v: string, vClass = 'v'): string {
  return `<div class="row"><span class="k">${esc(k)}</span><span class="${vClass}">${esc(v)}</span></div>`;
}

function formatProbs(p: [number, number, number] | null): string {
  if (!p) return '—';
  return `[${p[0].toFixed(3)}, ${p[1].toFixed(3)}, ${p[2].toFixed(3)}]`;
}

function summarizeChannelDist(
  tensor: Float32Array | null,
  ch: number,
): string {
  if (!tensor) return '—';
  const counts = new Map<number, number>();
  for (let i = ch; i < tensor.length; i += N_CHANNELS) {
    const v = Math.round(tensor[i]);
    counts.set(v, (counts.get(v) ?? 0) + 1);
  }
  const sorted = Array.from(counts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 4);
  return sorted.map(([val, n]) => `${val}×${n}`).join(', ');
}

function validationHtml(report: ValidationReport | null): string {
  if (!report) return rowHtml('encoder validation', '—');
  if (report.passed) {
    return rowHtml('encoder validation', 'pass', 'v ok');
  }
  const issues = report.issues
    .slice(0, 3)
    .map((s) => `<div class="err" style="margin-top:4px">${esc(s)}</div>`)
    .join('');
  return rowHtml('encoder validation', 'fail', 'v err') + issues;
}

export function mountDebugModal(): void {
  const elMaybe = document.getElementById('debug-modal');
  const toggleBtn = document.getElementById('toggle-debug');
  if (!elMaybe) return;
  const el: HTMLElement = elMaybe;

  let visible = false;
  let rafId: number | null = null;
  const gpu = detectGpu();

  const setVisible = (v: boolean): void => {
    visible = v;
    el.style.display = v ? 'block' : 'none';
    if (v) loop();
    else if (rafId !== null) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
  };

  const toggle = () => setVisible(!visible);

  toggleBtn?.addEventListener('click', toggle);
  window.addEventListener('keydown', (ev) => {
    if (
      ev.target instanceof HTMLInputElement ||
      ev.target instanceof HTMLTextAreaElement
    ) {
      return;
    }
    if (ev.key === 'd' || ev.key === 'D') toggle();
  });

  function render(): void {
    const c = sandboxState.candidate;
    const reads = sandboxState.reads;
    const reference = sandboxState.reference;
    const pos = sandboxState.predictPos;
    const gen = sandboxState.readsGeneration;

    let tensor: Float32Array | null = null;
    let report: ValidationReport | null = null;
    if (c && reads && reference && pos !== null) {
      tensor = encodePileup(reads, reference, pos, c);
      if (tensor) report = validateEncodedTensor(tensor);
    }

    const candidateLabel = c
      ? c.kind === 'snv'
        ? `snv ${c.refBase}>${c.altBase} (${c.supportingReads}/${c.qualifyingReads})`
        : c.kind === 'del'
          ? `del ${c.refBase} (${c.supportingReads}/${c.qualifyingReads})`
          : `ins ${c.refBase}>+${c.altSequence.join('')} (${c.supportingReads}/${c.qualifyingReads})`
      : 'none';

    const channelLines = CHANNEL_NAMES.map(
      (name, ch) => rowHtml(`  ${name}`, summarizeChannelDist(tensor, ch))
    ).join('');

    const t = debugTelemetry;
    const html = [
      `<div class="section">`,
      `<h3>TFJS backend</h3>`,
      rowHtml('backend', gpu.backend),
      gpu.webglVendor ? rowHtml('vendor', gpu.webglVendor) : '',
      gpu.webglRenderer ? rowHtml('renderer', gpu.webglRenderer) : '',
      `</div>`,
      `<div class="section">`,
      `<h3>State</h3>`,
      rowHtml('candidate', candidateLabel),
      rowHtml('predict pos', pos === null ? '—' : String(pos)),
      rowHtml('reads', reads ? String(reads.length) : '—'),
      rowHtml('reference len', reference ? String(reference.length) : '—'),
      rowHtml('generation', String(gen)),
      `</div>`,
      `<div class="section">`,
      `<h3>Last prediction</h3>`,
      rowHtml('probs', formatProbs(t.lastProbs)),
      rowHtml('argmax', t.lastArgmax ?? '—'),
      rowHtml(
        'latency',
        t.lastPredictMs !== null ? `${t.lastPredictMs.toFixed(1)} ms` : '—',
      ),
      rowHtml(
        'predict count',
        String(t.predictCount),
      ),
      `</div>`,
      `<div class="section">`,
      `<h3>Encoder</h3>`,
      validationHtml(report),
      tensor ? rowHtml('tensor cells', String(tensor.length)) : '',
      tensor ? `<div style="margin-top:6px">${channelLines}</div>` : '',
      `</div>`,
      `<div class="section">`,
      `<h3>Hotkeys</h3>`,
      rowHtml('d', 'toggle this modal'),
      rowHtml('← / →', 'prev / next candidate'),
      rowHtml('r', 'random candidate'),
      `</div>`,
    ].join('');

    el.innerHTML = html;
  }

  function loop(): void {
    render();
    rafId = requestAnimationFrame(loop);
  }
}
