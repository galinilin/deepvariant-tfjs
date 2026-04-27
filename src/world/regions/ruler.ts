import type p5 from 'p5';
import { WINDOW_LENGTH } from '../../lib/reference';

export const RULER_HEIGHT = 28;

const TICK_GAP = 4;
const TICK_HEIGHT = 6;
const LABEL_GAP = 4;
const LABEL_SIZE = 12;

const TICK_COLOR: [number, number, number] = [120, 120, 120];
const LABEL_COLOR: [number, number, number] = [140, 140, 140];

const TICK_INTERVAL = 10;

export interface RulerState {
  windowStart: number;
  origin: { x: number; y: number };
  cellWidth: number;
  zoom: number;
}

export function drawRuler(p: p5, state: RulerState): void {
  const { x, y } = state.origin;
  const startLabel =
    Math.ceil((state.windowStart + 1) / TICK_INTERVAL) * TICK_INTERVAL;
  const endLabel = state.windowStart + WINDOW_LENGTH;
  const tickY = y + TICK_GAP;
  const labelY = tickY + TICK_HEIGHT + LABEL_GAP;

  for (let labelPos = startLabel; labelPos <= endLabel; labelPos += TICK_INTERVAL) {
    const zeroIndexed = labelPos - 1;
    const tickX =
      x + (zeroIndexed - state.windowStart) * state.cellWidth + state.cellWidth / 2;

    p.stroke(TICK_COLOR[0], TICK_COLOR[1], TICK_COLOR[2]);
    p.strokeWeight(1 / state.zoom);
    p.line(tickX, tickY, tickX, tickY + TICK_HEIGHT);

    p.noStroke();
    p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
    p.textFont('Inconsolata');
    p.textStyle(p.NORMAL);
    p.textSize(LABEL_SIZE);
    p.textAlign(p.CENTER, p.TOP);
    p.text(String(labelPos), tickX, labelY);
  }
}
