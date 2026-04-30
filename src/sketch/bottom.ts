import p5 from 'p5';
import { sandboxState } from '../lib/sandbox-state';
import { DeepVariantModel, DV_CLASSES, type Genotype } from '../lib/DeepVariantModel';
import {
  PILEUP_HEIGHT,
  PILEUP_WIDTH,
  PILEUP_CHANNELS,
  REF_ROWS,
  PREDICT_COL,
} from '../lib/dv-channels';
import { CHANNEL_NAMES } from '../lib/parity';

export interface BottomHandle {
  destroy: () => void;
}

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];
const PLACEHOLDER_COLOR: [number, number, number] = [90, 90, 90];
const NO_CANDIDATE_COLOR: [number, number, number] = [120, 120, 120];
const ROW_HIGHLIGHT_COLOR: [number, number, number] = [255, 220, 110];

const MARGIN = 40;
const GAP = 40;
const HEADER_GAP = 8;

const VISIBLE_ROWS = 30; // 5 ref rows + up to 25 read rows; rest is zero-pad
const PILEUP_INNER_GAP = 4;
const PILEUP_LABEL_FONT_SIZE = 10;

/** Lazy global model handle. First tensor encode triggers the load. */
let modelPromise: Promise<DeepVariantModel> | null = null;
let modelInstance: DeepVariantModel | null = null;
let modelError: string | null = null;
let modelLoadProgress = 0;
let lastPredictedPosition = -1;
let lastPredictedGen = -1;
let predictDebounceTimer: ReturnType<typeof setTimeout> | null = null;
/** What the active debounce timer (or pending-after-predict slot) is aiming
 * to predict. Lets us skip the per-frame clear+reset that was preventing the
 * timer from ever firing. */
let pendingTarget: { pos: number; gen: number } | null = null;

function ensureModel(): Promise<DeepVariantModel> {
  if (modelInstance) return Promise.resolve(modelInstance);
  if (modelPromise) return modelPromise;
  const base = import.meta.env.BASE_URL;
  modelLoadProgress = 0;
  modelPromise = DeepVariantModel.load({
    modelBaseUrl: `${base}models/`,
    onProgress: (frac) => {
      modelLoadProgress = frac;
    },
  })
    .then((m) => {
      modelInstance = m;
      modelLoadProgress = 1;
      return m;
    })
    .catch((err) => {
      modelError = String(err);
      modelPromise = null;
      throw err;
    });
  return modelPromise;
}

async function runPrediction(
  tensor: Float32Array,
  position: number,
  generation: number,
): Promise<void> {
  if (sandboxState.predicting) return;
  if (
    sandboxState.pileupPosition !== position ||
    sandboxState.readsGeneration !== generation
  ) {
    return;
  }
  sandboxState.predicting = true;
  try {
    const m = await ensureModel();
    if (
      sandboxState.pileupPosition !== position ||
      sandboxState.readsGeneration !== generation
    ) {
      return;
    }
    const result = await m.predict(tensor);
    if (
      sandboxState.pileupPosition !== position ||
      sandboxState.readsGeneration !== generation
    ) {
      return;
    }
    sandboxState.prediction = { ...result, position };
    lastPredictedPosition = position;
    lastPredictedGen = generation;
  } catch (err) {
    console.error('prediction failed:', err);
  } finally {
    sandboxState.predicting = false;
  }
}

function maybeTriggerPrediction(): void {
  if (!sandboxState.pileupTensor) return;
  const pos = sandboxState.pileupPosition;
  const gen = sandboxState.readsGeneration;
  if (pos === lastPredictedPosition && gen === lastPredictedGen) return;

  // Already-scheduled timer aiming at the same target? Don't disturb it —
  // resetting every frame is what kept the timer from ever elapsing.
  if (
    predictDebounceTimer &&
    pendingTarget &&
    pendingTarget.pos === pos &&
    pendingTarget.gen === gen
  ) {
    return;
  }

  if (sandboxState.predicting) {
    // Predict in flight. Record what we'd want next, so the post-finally
    // p.draw sees the mismatch and schedules a fresh timer.
    pendingTarget = { pos, gen };
    return;
  }

  if (predictDebounceTimer) clearTimeout(predictDebounceTimer);
  pendingTarget = { pos, gen };
  const tensor = sandboxState.pileupTensor;
  predictDebounceTimer = setTimeout(() => {
    predictDebounceTimer = null;
    pendingTarget = null;
    void runPrediction(tensor, pos, gen);
  }, 180);
}

export function mountBottomSketch(container: HTMLElement): BottomHandle {
  const size = () => ({
    w: container.clientWidth,
    h: container.clientHeight,
  });

  let channelImage: p5.Image | null = null;

  const instance = new p5((p: p5) => {
    p.setup = () => {
      const { w, h } = size();
      p.createCanvas(w, h);
      p.pixelDensity(p.displayDensity());
      p.textFont('Inconsolata');
      // Composite channel image: VISIBLE_ROWS tall × (PILEUP_WIDTH * 7 + gaps) wide
      channelImage = p.createImage(
        PILEUP_WIDTH * PILEUP_CHANNELS + PILEUP_INNER_GAP * (PILEUP_CHANNELS - 1),
        VISIBLE_ROWS,
      );
    };

    p.windowResized = () => {
      const { w, h } = size();
      p.resizeCanvas(w, h);
    };

    p.draw = () => {
      p.background(0);
      maybeTriggerPrediction();
      drawPanels(p, channelImage);
    };
  }, container);

  return {
    destroy: () => instance.remove(),
  };
}

function drawPanels(p: p5, channelImage: p5.Image | null): void {
  const innerW = p.width - MARGIN * 2 - GAP;
  const innerH = p.height - MARGIN * 2;
  const leftW = Math.round(innerW * 0.74);
  const rightW = innerW - leftW;

  drawPileupImagePanel(p, MARGIN, MARGIN, leftW, innerH, channelImage);
  drawPredictionPanel(p, MARGIN + leftW + GAP, MARGIN, rightW, innerH);
}

function drawPanelChrome(
  p: p5,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  rightLabel?: string,
): void {
  p.noStroke();
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textSize(14);
  p.textAlign(p.LEFT, p.BOTTOM);
  p.text(label, x, y - HEADER_GAP);

  if (rightLabel) {
    p.fill(140, 140, 140);
    p.textSize(11);
    p.textAlign(p.RIGHT, p.BOTTOM);
    p.text(rightLabel, x + w, y - HEADER_GAP);
  }

  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);
}

function drawCenterText(
  p: p5,
  x: number,
  y: number,
  w: number,
  h: number,
  text: string,
  color: [number, number, number],
  size = 13,
): void {
  p.noStroke();
  p.fill(color[0], color[1], color[2]);
  p.textSize(size);
  p.textAlign(p.CENTER, p.CENTER);
  p.text(text, x + w / 2, y + h / 2);
}

function drawPileupImagePanel(
  p: p5,
  x: number,
  y: number,
  w: number,
  h: number,
  channelImage: p5.Image | null,
): void {
  drawPanelChrome(p, x, y, w, h, 'Pileup Image');

  const tensor = sandboxState.pileupTensor;
  if (!tensor || !sandboxState.candidate) {
    drawCenterText(p, x, y, w, h, 'No candidate', NO_CANDIDATE_COLOR);
    return;
  }
  if (!channelImage) {
    drawCenterText(p, x, y, w, h, 'Loading…', PLACEHOLDER_COLOR);
    return;
  }

  // Repaint the composite channel image from the current tensor.
  channelImage.loadPixels();
  const stride = PILEUP_WIDTH + PILEUP_INNER_GAP;
  const stridedW = channelImage.width;
  for (let ch = 0; ch < PILEUP_CHANNELS; ch++) {
    const colOffset = ch * stride;
    for (let row = 0; row < VISIBLE_ROWS; row++) {
      for (let col = 0; col < PILEUP_WIDTH; col++) {
        const tensorIdx =
          row * PILEUP_WIDTH * PILEUP_CHANNELS + col * PILEUP_CHANNELS + ch;
        const v = Math.max(0, Math.min(255, Math.round(tensor[tensorIdx])));
        const pix = (row * stridedW + colOffset + col) * 4;
        channelImage.pixels[pix + 0] = v;
        channelImage.pixels[pix + 1] = v;
        channelImage.pixels[pix + 2] = v;
        channelImage.pixels[pix + 3] = 255;
      }
    }
    // gap between channels: dim grey
    if (ch < PILEUP_CHANNELS - 1) {
      for (let row = 0; row < VISIBLE_ROWS; row++) {
        for (let g = 0; g < PILEUP_INNER_GAP; g++) {
          const pix = (row * stridedW + colOffset + PILEUP_WIDTH + g) * 4;
          channelImage.pixels[pix + 0] = 22;
          channelImage.pixels[pix + 1] = 22;
          channelImage.pixels[pix + 2] = 26;
          channelImage.pixels[pix + 3] = 255;
        }
      }
    }
  }
  channelImage.updatePixels();

  const innerPad = 14;
  const labelStripH = PILEUP_LABEL_FONT_SIZE + 6;
  const footerH = 16;
  const imgX = x + innerPad;
  const imgY = y + innerPad + labelStripH;
  const imgW = w - innerPad * 2;
  // Stretch height to fill the panel — the source aspect (1571×30) is too
  // wide to preserve. The model sees this dense pixel grid; we expose its
  // structure (channel separation, predict column) over per-cell fidelity.
  const imgH = h - innerPad * 2 - labelStripH - footerH;

  // Crisp pixel scaling: the source channel image is small enough that
  // smoothing makes adjacent rows blur together.
  const ctx = p.drawingContext as CanvasRenderingContext2D;
  const prevSmooth = ctx.imageSmoothingEnabled;
  ctx.imageSmoothingEnabled = false;
  p.image(channelImage, imgX, imgY, imgW, imgH);
  ctx.imageSmoothingEnabled = prevSmooth;

  // Channel labels above each strip
  const colPxRatio = imgW / channelImage.width;
  const gapPx = PILEUP_INNER_GAP * colPxRatio;
  const channelW = PILEUP_WIDTH * colPxRatio;
  p.noStroke();
  p.fill(150, 150, 160);
  p.textSize(PILEUP_LABEL_FONT_SIZE);
  p.textAlign(p.CENTER, p.BOTTOM);
  for (let ch = 0; ch < PILEUP_CHANNELS; ch++) {
    const cx = imgX + ch * (channelW + gapPx) + channelW / 2;
    p.text(CHANNEL_NAMES[ch], cx, imgY - 4);
  }

  // Predict-column highlight on every channel strip
  p.stroke(ROW_HIGHLIGHT_COLOR[0], ROW_HIGHLIGHT_COLOR[1], ROW_HIGHLIGHT_COLOR[2], 200);
  p.strokeWeight(1);
  for (let ch = 0; ch < PILEUP_CHANNELS; ch++) {
    const colCenterCacheX = ch * (PILEUP_WIDTH + PILEUP_INNER_GAP) + PREDICT_COL + 0.5;
    const lineX = imgX + colCenterCacheX * colPxRatio;
    p.line(lineX, imgY - 2, lineX, imgY + imgH + 2);
  }

  // Footer note
  p.noStroke();
  p.fill(120, 120, 130);
  p.textSize(10);
  p.textAlign(p.LEFT, p.TOP);
  p.text(
    `${REF_ROWS} ref rows + reads · top ${VISIBLE_ROWS}/${PILEUP_HEIGHT}`,
    imgX,
    imgY + imgH + 4,
  );
}

function drawPredictionPanel(
  p: p5,
  x: number,
  y: number,
  w: number,
  h: number,
): void {
  const positionLabel = sandboxState.candidate
    ? `pos ${sandboxState.pileupPosition + 1}`
    : '';
  drawPanelChrome(p, x, y, w, h, 'Prediction', positionLabel);

  if (!sandboxState.candidate) {
    drawCenterText(p, x, y, w, h, 'No candidate', NO_CANDIDATE_COLOR);
    return;
  }
  if (modelError) {
    drawCenterText(p, x, y, w, h, 'Model load failed', [200, 100, 100]);
    return;
  }
  if (!sandboxState.prediction || sandboxState.predicting) {
    let text: string;
    if (!modelInstance) {
      const pct = Math.round(modelLoadProgress * 100);
      text = modelPromise
        ? `Loading model… ${pct}%`
        : 'Loading model…';
    } else {
      text = 'Predicting…';
    }
    drawCenterText(p, x, y, w, h, text, PLACEHOLDER_COLOR);
    return;
  }
  const stale = sandboxState.prediction.position !== sandboxState.pileupPosition;
  const pred = sandboxState.prediction;

  const innerPad = 16;
  const barAreaY = y + innerPad + 24;
  const barAreaH = h - (barAreaY - y) - innerPad - 30;
  const barH = (barAreaH - 12) / 3;
  const barAreaX = x + innerPad;
  const barAreaW = w - innerPad * 2;

  p.textFont('Inconsolata');
  p.textAlign(p.LEFT, p.TOP);
  p.textSize(13);
  p.noStroke();
  p.fill(stale ? [110, 110, 110] : [240, 220, 130]);
  p.text(`${pred.argmax}`, x + innerPad, y + innerPad);
  p.textSize(11);
  p.fill(140, 140, 140);
  p.textAlign(p.RIGHT, p.TOP);
  p.text(`${(pred.confidence * 100).toFixed(1)}%`, x + w - innerPad, y + innerPad);

  for (let i = 0; i < DV_CLASSES.length; i++) {
    const cls = DV_CLASSES[i] as Genotype;
    const prob = pred.probs[cls];
    const by = barAreaY + i * (barH + 6);
    p.noStroke();
    p.fill(40, 40, 46);
    p.rect(barAreaX, by, barAreaW, barH);
    const filled = barAreaW * prob;
    const isArgmax = cls === pred.argmax;
    if (isArgmax) {
      p.fill(stale ? [120, 110, 70] : [240, 220, 130]);
    } else {
      p.fill(110, 110, 120);
    }
    p.rect(barAreaX, by, filled, barH);
    p.fill(220, 220, 220);
    p.textAlign(p.LEFT, p.CENTER);
    p.textSize(11);
    p.text(cls, barAreaX + 6, by + barH / 2);
    p.textAlign(p.RIGHT, p.CENTER);
    p.text(`${(prob * 100).toFixed(1)}%`, barAreaX + barAreaW - 6, by + barH / 2);
  }

  if (stale) {
    p.noStroke();
    p.fill(120, 120, 130);
    p.textAlign(p.LEFT, p.BOTTOM);
    p.textSize(10);
    p.text('updating…', x + innerPad, y + h - innerPad);
  }
}
