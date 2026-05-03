import p5 from 'p5';
import { sandboxState } from '../lib/sandbox-state';
import { encodePileup } from '../lib/pileup-encoder';
import { DeepVariantModel, type Genotype } from '../lib/DeepVariantModel';
import { formatAlt } from '../lib/candidate';
import { recordPrediction } from '../lib/debug-telemetry';

export interface BottomHandle {
  destroy: () => void;
}

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const MUTED_COLOR: [number, number, number] = [120, 120, 120];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];
const PLACEHOLDER_COLOR: [number, number, number] = [90, 90, 90];
// (ACCEPT_COLOR removed in v5.4 — superseded by AMBER readonly tuple
// used for animated bar color targets.)

const MARGIN = 40;
const GAP = 40;
const HEADER_GAP = 8;
const HEADER_HEIGHT = 28;

const CHANNEL_NAMES = [
  'read_base',
  'base_quality',
  'mapping_quality',
  'strand',
  'supports_variant',
  'differs_from_ref',
  'insert_size',
];

const TENSOR_W = 221;
const TENSOR_H = 100;
const N_CHANNELS = 7;

const PREDICT_DEBOUNCE_MS = 220;

interface CachedPrediction {
  pos: number;
  generation: number;
  candidateKey: string;
  probs: [number, number, number];
  argmax: Genotype;
  channelImages: p5.Image[]; // 7 grayscale strips
}

// v5.4 — per-frame tween of probability bars + argmax color. Uses
// exponential decay so the lerp is time-step independent and frame-rate
// changes (background tab → foreground) don't snap or speed up.
const ANIM_TC_MS = 80; // ~63% in 80ms, ~95% in 240ms
const ANIM_DT_CLAMP_MS = 100;
const AMBER: readonly [number, number, number] = [255, 220, 110];
const BAR_GRAY: readonly [number, number, number] = [80, 80, 80];

const CLASS_INDEX: Record<Genotype, 0 | 1 | 2> = {
  hom_ref: 0,
  het: 1,
  hom_alt: 2,
};

export function mountBottomSketch(
  container: HTMLElement,
  model: DeepVariantModel,
): BottomHandle {
  let modelError: string | null = null;
  let predicting = false;
  let cached: CachedPrediction | null = null;
  let pendingTimer: number | null = null;
  let pendingTarget: { pos: number; generation: number; candidateKey: string } | null = null;

  // Animated display state — tweens toward `cached` each frame so bars
  // grow/shrink + amber crossfades smoothly between predictions.
  const animState = {
    probs: [0, 0, 0] as [number, number, number],
    barColors: [
      [BAR_GRAY[0], BAR_GRAY[1], BAR_GRAY[2]],
      [BAR_GRAY[0], BAR_GRAY[1], BAR_GRAY[2]],
      [BAR_GRAY[0], BAR_GRAY[1], BAR_GRAY[2]],
    ] as [
      [number, number, number],
      [number, number, number],
      [number, number, number],
    ],
    lastFrameMs: 0,
  };

  function tickPredictionAnimation(): void {
    if (!cached) return;
    const now = performance.now();
    const dt =
      animState.lastFrameMs === 0
        ? 16
        : Math.min(ANIM_DT_CLAMP_MS, now - animState.lastFrameMs);
    animState.lastFrameMs = now;
    const alpha = 1 - Math.exp(-dt / ANIM_TC_MS);

    const argmaxIdx = CLASS_INDEX[cached.argmax];
    for (let i = 0; i < 3; i++) {
      animState.probs[i] += (cached.probs[i] - animState.probs[i]) * alpha;
      const target = i === argmaxIdx ? AMBER : BAR_GRAY;
      for (let c = 0; c < 3; c++) {
        animState.barColors[i][c] +=
          (target[c] - animState.barColors[i][c]) * alpha;
      }
    }
  }

  // How "amber-like" is a bar's current color, 0..1? Threshold this for
  // styling decisions (label color, prob font size) so styles don't
  // flicker mid-tween.
  function amberProgress(rgb: [number, number, number]): number {
    const dr = rgb[0] - BAR_GRAY[0];
    const dg = rgb[1] - BAR_GRAY[1];
    const db = rgb[2] - BAR_GRAY[2];
    const tr = AMBER[0] - BAR_GRAY[0];
    const tg = AMBER[1] - BAR_GRAY[1];
    const tb = AMBER[2] - BAR_GRAY[2];
    const num = dr * tr + dg * tg + db * tb;
    const den = tr * tr + tg * tg + tb * tb;
    return Math.max(0, Math.min(1, num / den));
  }

  const candidateKey = (): string => {
    const c = sandboxState.candidate;
    if (!c) return 'null';
    switch (c.kind) {
      case 'snv':
        return `snv:${c.refBase}>${c.altBase}`;
      case 'del':
        return `del:${c.refBase}`;
      case 'ins':
        return `ins:${c.refBase}>${c.altSequence.join('')}`;
    }
  };

  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  const instance = new p5((p: p5) => {
    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
      // Model is pre-loaded by the welcome flow and passed in. No lazy
      // load here — by the time bottom is mounted, model is ready.
    };

    p.windowResized = () => {
      const { w, h } = size();
      p.resizeCanvas(w, h);
    };

    p.draw = () => {
      p.background(0);

      const c = sandboxState.candidate;
      const reads = sandboxState.reads;
      const reference = sandboxState.reference;
      const pos = sandboxState.predictPos;
      const gen = sandboxState.readsGeneration;

      // Schedule a predict if state changed and we have everything we need.
      if (c && reads && reference && pos !== null) {
        const key = candidateKey();
        const stale =
          !cached ||
          cached.pos !== pos ||
          cached.generation !== gen ||
          cached.candidateKey !== key;
        if (stale && !predicting) {
          schedulePredict(pos, gen, key);
        }
      }

      tickPredictionAnimation();

      const innerW = p.width - MARGIN * 2 - GAP;
      const innerH = p.height - MARGIN * 2;
      const leftW = Math.round(innerW * 0.72);
      const rightW = innerW - leftW;

      drawPileupPanel(p, MARGIN, MARGIN, leftW, innerH);
      drawPredictionPanel(p, MARGIN + leftW + GAP, MARGIN, rightW, innerH);
    };
  }, container);

  function schedulePredict(pos: number, generation: number, key: string): void {
    // CRITICAL: only reset the timer when the target actually changes.
    // p.draw runs at 60 Hz; if we reset the 220 ms debounce on every frame,
    // it never elapses while the tab is foregrounded — predicts only fire
    // when rAF is throttled (e.g., tab switch). Symptom users hit: have to
    // alt-tab away and back to see predictions update.
    if (
      pendingTarget &&
      pendingTarget.pos === pos &&
      pendingTarget.generation === generation &&
      pendingTarget.candidateKey === key
    ) {
      return;
    }
    pendingTarget = { pos, generation, candidateKey: key };
    if (pendingTimer !== null) window.clearTimeout(pendingTimer);
    pendingTimer = window.setTimeout(() => {
      pendingTimer = null;
      runPredict();
    }, PREDICT_DEBOUNCE_MS);
  }

  async function runPredict(): Promise<void> {
    if (predicting || !pendingTarget) return;
    const target = pendingTarget;
    pendingTarget = null;

    // Re-check live state under the same target to avoid running on
    // stale snapshots if something changed during the debounce.
    const c = sandboxState.candidate;
    const reads = sandboxState.reads;
    const reference = sandboxState.reference;
    const pos = sandboxState.predictPos;
    if (
      !c ||
      !reads ||
      !reference ||
      pos !== target.pos ||
      sandboxState.readsGeneration !== target.generation ||
      candidateKey() !== target.candidateKey
    ) {
      // State changed — let the next p.draw schedule a fresh predict.
      return;
    }

    const tensor = encodePileup(reads, reference, pos, c);
    if (!tensor) return;

    // DV preprocessing: the network was trained on (uint8_pixel - 128) / 128
    // in [-1, 1]. Source: dv-tfjs/scripts/convert.py + deepvariant/dv_utils.py.
    // We render strips from the pre-preprocessed tensor (raw [0, 254]) but
    // feed a normalized copy to the model.
    const modelInput = new Float32Array(tensor.length);
    for (let i = 0; i < tensor.length; i++) {
      modelInput[i] = (tensor[i] - 128) / 128;
    }

    predicting = true;
    const t0 = performance.now();
    try {
      const result = await model.predict(modelInput);
      const probs: [number, number, number] = [
        result.probs.hom_ref,
        result.probs.het,
        result.probs.hom_alt,
      ];
      recordPrediction(probs, result.argmax, performance.now() - t0);
      const channelImages = await buildChannelImages(instance, tensor);
      cached = {
        pos: target.pos,
        generation: target.generation,
        candidateKey: target.candidateKey,
        probs,
        argmax: result.argmax,
        channelImages,
      };
    } catch (err) {
      modelError = err instanceof Error ? err.message : 'predict failed';
    } finally {
      predicting = false;
    }
  }

  function drawPileupPanel(
    p: p5,
    x: number,
    y: number,
    w: number,
    h: number,
  ): void {
    drawPanelChrome(p, x, y, w, h, 'Pileup Image');

    const c = sandboxState.candidate;
    if (!c) {
      drawCenteredText(p, x, y, w, h, 'no candidate', PLACEHOLDER_COLOR, 13);
      return;
    }

    if (modelError) {
      drawCenteredText(
        p,
        x,
        y,
        w,
        h,
        `model error: ${modelError}`,
        [200, 100, 100],
        12,
      );
      return;
    }

    if (!cached || cached.candidateKey !== candidateKey()) {
      drawCenteredText(
        p,
        x,
        y,
        w,
        h,
        'predicting…',
        PLACEHOLDER_COLOR,
        12,
      );
      return;
    }

    // 7 channel strips stacked vertically
    const stripsTop = y + HEADER_HEIGHT;
    const stripsBottom = y + h - 6;
    const labelW = 130;
    const stripsLeft = x + 12 + labelW;
    const stripsRight = x + w - 12;
    const stripsWidth = stripsRight - stripsLeft;
    const totalGap = 4 * (N_CHANNELS - 1);
    const stripH = Math.max(
      14,
      Math.floor((stripsBottom - stripsTop - totalGap) / N_CHANNELS),
    );

    p.textSize(11);
    p.noStroke();
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      const stripY = stripsTop + ch * (stripH + 4);
      p.fill(MUTED_COLOR[0], MUTED_COLOR[1], MUTED_COLOR[2]);
      p.textAlign(p.RIGHT, p.CENTER);
      p.text(CHANNEL_NAMES[ch], stripsLeft - 8, stripY + stripH / 2);
      p.image(cached.channelImages[ch], stripsLeft, stripY, stripsWidth, stripH);
      p.noFill();
      p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
      p.strokeWeight(0.5);
      p.rect(stripsLeft, stripY, stripsWidth, stripH);
      p.noStroke();
    }
  }

  function drawPredictionPanel(
    p: p5,
    x: number,
    y: number,
    w: number,
    h: number,
  ): void {
    drawPanelChrome(p, x, y, w, h, 'Prediction');

    const c = sandboxState.candidate;
    if (!c) {
      drawCenteredText(p, x, y, w, h, 'no candidate', PLACEHOLDER_COLOR, 13);
      return;
    }

    let cursorY = y + HEADER_HEIGHT;
    const innerLeft = x + 12;
    const innerRight = x + w - 12;
    const innerW = innerRight - innerLeft;

    // Variant header
    const altLabel = formatAlt(c);
    p.noStroke();
    p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
    p.textSize(13);
    p.textAlign(p.LEFT, p.TOP);
    p.text(`${c.refBase}→${altLabel}`, innerLeft, cursorY);
    p.fill(MUTED_COLOR[0], MUTED_COLOR[1], MUTED_COLOR[2]);
    p.textSize(11);
    p.text(
      `support ${c.supportingReads}/${c.qualifyingReads}`,
      innerLeft,
      cursorY + 18,
    );
    cursorY += 40;

    if (modelError) {
      drawCenteredText(
        p,
        x,
        y + HEADER_HEIGHT + 50,
        w,
        h - HEADER_HEIGHT - 50,
        `model error: ${modelError}`,
        [200, 100, 100],
        12,
      );
      return;
    }

    // First-predict-of-session placeholder: only shown if we've never
    // had a successful prediction yet. Otherwise bars stay visible from
    // their last animated state and tween into the new prediction once
    // it arrives.
    if (!cached) {
      drawCenteredText(
        p,
        x,
        cursorY,
        w,
        h - (cursorY - y) - 6,
        'predicting…',
        PLACEHOLDER_COLOR,
        12,
      );
      return;
    }

    // If cached is stale (a new candidate is in flight), surface a
    // small subtitle below the variant header but keep the bars rendered
    // from the animated state.
    const isStale = cached.candidateKey !== candidateKey();
    if (isStale) {
      p.fill(PLACEHOLDER_COLOR[0], PLACEHOLDER_COLOR[1], PLACEHOLDER_COLOR[2]);
      p.textSize(10);
      p.textAlign(p.LEFT, p.TOP);
      p.text('predicting…', innerLeft, cursorY - 14);
    }

    // Softmax bars — values + colors come from animState (lerped each
    // frame toward `cached`). Pulse on the bar that's most amber-like.
    const classes: Genotype[] = ['hom_ref', 'het', 'hom_alt'];
    const labelColW = 70;
    const valueColW = 60;
    const barX = innerLeft + labelColW;
    const barW = innerW - labelColW - valueColW;
    const barH = 14;
    const rowH = 28;
    const now = performance.now();
    // Subtle "alive" pulse on the argmax bar: alpha 0.92 ↔ 1.00 over
    // 1.8 s. Non-argmax bars stay at full alpha.
    const pulse = 0.96 + 0.04 * Math.sin((now * 2 * Math.PI) / 1800);

    for (let i = 0; i < 3; i++) {
      const cls = classes[i];
      const prob = animState.probs[i];
      const color = animState.barColors[i];
      const amberness = amberProgress(color);
      const fillW = Math.round(barW * Math.max(0, Math.min(1, prob)));

      // Class label — fades from gray → amber as the bar gains argmax.
      const labelColor: [number, number, number] = lerp3(
        LABEL_COLOR,
        AMBER as [number, number, number],
        amberness,
      );
      p.fill(labelColor[0], labelColor[1], labelColor[2]);
      p.textSize(12);
      p.textAlign(p.LEFT, p.CENTER);
      p.text(cls, innerLeft, cursorY + barH / 2);

      // Bar background.
      p.noStroke();
      p.fill(28, 28, 28);
      p.rect(barX, cursorY, barW, barH);

      // Bar fill.
      const fillAlpha = amberness > 0.5 ? pulse : 1;
      const ctx = p.drawingContext as CanvasRenderingContext2D;
      ctx.save();
      ctx.globalAlpha = fillAlpha;
      if (amberness > 0.5) {
        // Subtle vertical gradient on the argmax bar — top slightly
        // brighter, bottom slightly darker. Adds dimension without
        // shouting.
        const grad = ctx.createLinearGradient(0, cursorY, 0, cursorY + barH);
        const top = lerp3(BAR_GRAY as [number, number, number], [255, 230, 130], amberness);
        const bot = lerp3(BAR_GRAY as [number, number, number], [220, 180, 80], amberness);
        grad.addColorStop(0, `rgb(${top[0]|0}, ${top[1]|0}, ${top[2]|0})`);
        grad.addColorStop(1, `rgb(${bot[0]|0}, ${bot[1]|0}, ${bot[2]|0})`);
        ctx.fillStyle = grad;
      } else {
        ctx.fillStyle = `rgb(${color[0]|0}, ${color[1]|0}, ${color[2]|0})`;
      }
      // Square fill — matches the rest of the visual language.
      if (fillW > 0) ctx.fillRect(barX, cursorY, fillW, barH);
      ctx.restore();

      // Probability value — amber-tinted + larger when this is the argmax.
      p.fill(labelColor[0], labelColor[1], labelColor[2]);
      p.textSize(amberness > 0.5 ? 13 : 12);
      p.textAlign(p.RIGHT, p.CENTER);
      p.text(prob.toFixed(3), innerRight, cursorY + barH / 2);

      cursorY += rowH;
    }
  }

  function lerp3(
    a: [number, number, number] | readonly [number, number, number],
    b: [number, number, number] | readonly [number, number, number],
    t: number,
  ): [number, number, number] {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function drawPanelChrome(
    p: p5,
    x: number,
    y: number,
    w: number,
    h: number,
    label: string,
  ): void {
    p.noStroke();
    p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
    p.textSize(14);
    p.textAlign(p.LEFT, p.BOTTOM);
    p.text(label, x, y - HEADER_GAP);

    p.noFill();
    p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
    p.strokeWeight(0.5);
    p.rect(x, y, w, h);
    p.noStroke();
  }

  function drawCenteredText(
    p: p5,
    x: number,
    y: number,
    w: number,
    h: number,
    msg: string,
    color: [number, number, number],
    size: number,
  ): void {
    p.noStroke();
    p.fill(color[0], color[1], color[2]);
    p.textSize(size);
    p.textAlign(p.CENTER, p.CENTER);
    p.text(msg, x + w / 2, y + h / 2);
  }

  /**
   * Build the 7 grayscale channel strips. Each strip is 100×221 = 22,100
   * RGBA pixels (~88 KB). Total ~615 KB across 7 channels. The work is
   * mostly synchronous Uint8 writes which can pause the main thread for
   * 5-10 ms — enough to drop a p5 frame on the top canvas. Yielding
   * between channels lets requestAnimationFrame tick in between, keeping
   * pan/scrub/scrubber smooth even on slower hardware.
   */
  async function buildChannelImages(p: p5, tensor: Float32Array): Promise<p5.Image[]> {
    const out: p5.Image[] = [];
    for (let ch = 0; ch < N_CHANNELS; ch++) {
      const img = p.createImage(TENSOR_W, TENSOR_H);
      img.loadPixels();
      for (let yy = 0; yy < TENSOR_H; yy++) {
        for (let xx = 0; xx < TENSOR_W; xx++) {
          const v = tensor[(yy * TENSOR_W + xx) * N_CHANNELS + ch] | 0;
          const i = (yy * TENSOR_W + xx) * 4;
          img.pixels[i] = v;
          img.pixels[i + 1] = v;
          img.pixels[i + 2] = v;
          img.pixels[i + 3] = 255;
        }
      }
      img.updatePixels();
      out.push(img);
      // Yield between channels so the top canvas's draw loop can tick.
      // The tradeoff is the prediction panel shows "predicting…" for
      // ~7 extra frames after the model returns, but the top canvas
      // stays buttery.
      if (ch < N_CHANNELS - 1) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }
    return out;
  }

  return {
    destroy: () => {
      if (pendingTimer !== null) window.clearTimeout(pendingTimer);
      // Model is owned by main.ts; we don't dispose here.
      instance.remove();
    },
  };
}
