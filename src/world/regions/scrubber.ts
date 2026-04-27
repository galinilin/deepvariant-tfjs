import type p5 from 'p5';
import { igvColor, type Base } from '../../lib/palette';
import { WINDOW_LENGTH } from '../../lib/reference';

export const SCRUBBER_HEIGHT = 16;

const LABEL_GAP = 12;
const LABEL_SIZE = 14;
const META_SIZE = 11;
const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const META_COLOR: [number, number, number] = [120, 120, 120];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];
const WINDOW_BORDER: [number, number, number] = [220, 220, 220];

const OUT_DIM = 0.35;

export interface ScrubberState {
  reference: Base[];
  windowStart: number;
  origin: { x: number; y: number };
  width: number;
}

export function drawScrubber(p: p5, state: ScrubberState): void {
  const { x, y } = state.origin;
  const w = state.width;
  const h = SCRUBBER_HEIGHT;
  const len = state.reference.length;
  const cellW = w / len;

  // Label on the left
  p.noStroke();
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textFont('Inconsolata');
  p.textStyle(p.NORMAL);
  p.textSize(LABEL_SIZE);
  p.textAlign(p.RIGHT, p.CENTER);
  p.text('Window', x - LABEL_GAP, y + h / 2);

  // Position meta on the right
  p.fill(META_COLOR[0], META_COLOR[1], META_COLOR[2]);
  p.textSize(META_SIZE);
  p.textAlign(p.LEFT, p.CENTER);
  p.text(
    `${state.windowStart + 1}–${state.windowStart + WINDOW_LENGTH} / ${len}`,
    x + w + LABEL_GAP,
    y + h / 2,
  );

  // Bases (dim outside window, full inside)
  p.noStroke();
  for (let i = 0; i < len; i++) {
    const inWindow = i >= state.windowStart && i < state.windowStart + WINDOW_LENGTH;
    const dim = inWindow ? 1.0 : OUT_DIM;
    const col = igvColor(state.reference[i] ?? 'N');
    p.fill(col[0] * dim, col[1] * dim, col[2] * dim);
    p.rect(x + i * cellW, y, Math.ceil(cellW) + 0.5, h);
  }

  // Window highlight border
  const winLeft = x + (state.windowStart / len) * w;
  const winRight = x + ((state.windowStart + WINDOW_LENGTH) / len) * w;
  p.noFill();
  p.stroke(WINDOW_BORDER[0], WINDOW_BORDER[1], WINDOW_BORDER[2]);
  p.strokeWeight(1);
  p.rect(winLeft, y, winRight - winLeft, h);

  // Outer border
  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);
}

export function isInsideScrubber(state: ScrubberState, wx: number, wy: number): boolean {
  return (
    wx >= state.origin.x &&
    wx <= state.origin.x + state.width &&
    wy >= state.origin.y &&
    wy <= state.origin.y + SCRUBBER_HEIGHT
  );
}

export function windowStartFromWorldX(state: ScrubberState, wx: number): number {
  const localX = wx - state.origin.x;
  const fraction = localX / state.width;
  const targetCenter = fraction * state.reference.length;
  return Math.round(targetCenter - WINDOW_LENGTH / 2);
}
