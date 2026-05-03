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
const ACCEPT_COLOR: [number, number, number] = [255, 220, 110];

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

export function mountBottomSketch(container: HTMLElement): BottomHandle {
  let model: DeepVariantModel | null = null;
  let modelPromise: Promise<DeepVariantModel> | null = null;
  let modelError: string | null = null;
  let predicting = false;
  let cached: CachedPrediction | null = null;
  let pendingTimer: number | null = null;
  let pendingTarget: { pos: number; generation: number; candidateKey: string } | null = null;

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

      // Lazy-start model load; first draw frame after this resolves
      // unblocks the predict pipeline.
      modelPromise = DeepVariantModel.load({});
      modelPromise
        .then((m) => {
          model = m;
        })
        .catch((err) => {
          modelError =
            err instanceof Error ? err.message : 'unknown model load error';
        });
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
      if (model && c && reads && reference && pos !== null) {
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
    if (!model || predicting || !pendingTarget) return;
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
      const channelImages = buildChannelImages(instance, tensor);
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
        model ? 'predicting…' : 'loading model…',
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

    if (!cached || cached.candidateKey !== candidateKey()) {
      const msg = model
        ? predicting || pendingTimer !== null
          ? 'predicting…'
          : 'predicting…'
        : 'loading model…';
      drawCenteredText(
        p,
        x,
        cursorY,
        w,
        h - (cursorY - y) - 6,
        msg,
        PLACEHOLDER_COLOR,
        12,
      );
      return;
    }

    // Softmax bars
    const classes: Genotype[] = ['hom_ref', 'het', 'hom_alt'];
    const labelColW = 70;
    const valueColW = 56;
    const barX = innerLeft + labelColW;
    const barW = innerW - labelColW - valueColW;
    const barH = 14;
    const rowH = 26;

    p.textSize(12);
    for (let i = 0; i < 3; i++) {
      const cls = classes[i];
      const prob = cached.probs[i];
      const isArgmax = cls === cached.argmax;
      const fillW = Math.round(barW * Math.max(0, Math.min(1, prob)));

      p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
      p.textAlign(p.LEFT, p.CENTER);
      p.text(cls, innerLeft, cursorY + barH / 2);

      // Bar background
      p.fill(28, 28, 28);
      p.rect(barX, cursorY, barW, barH);

      if (isArgmax) {
        p.fill(ACCEPT_COLOR[0], ACCEPT_COLOR[1], ACCEPT_COLOR[2]);
      } else {
        p.fill(80, 80, 80);
      }
      p.rect(barX, cursorY, fillW, barH);

      p.fill(MUTED_COLOR[0], MUTED_COLOR[1], MUTED_COLOR[2]);
      p.textAlign(p.RIGHT, p.CENTER);
      p.text(prob.toFixed(3), innerRight, cursorY + barH / 2);

      cursorY += rowH;
    }
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

  function buildChannelImages(p: p5, tensor: Float32Array): p5.Image[] {
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
    }
    return out;
  }

  return {
    destroy: () => {
      if (pendingTimer !== null) window.clearTimeout(pendingTimer);
      if (model) model.dispose();
      instance.remove();
    },
  };
}
