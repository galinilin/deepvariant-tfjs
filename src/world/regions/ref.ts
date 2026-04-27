import type p5 from 'p5';
import { igvColor, type Base } from '../../lib/palette';
import { WINDOW_LENGTH } from '../../lib/reference';

export const REF_COUNT = WINDOW_LENGTH;
export const CELL_W = 14;
export const CELL_H = 26;

const LABEL_GAP = 12;
const LABEL_SIZE = 14;
const BASE_SIZE = 16;

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];

export interface RefState {
  origin: { x: number; y: number };
}

export function drawRefFrame(p: p5, state: RefState): void {
  const { x, y } = state.origin;
  const w = REF_COUNT * CELL_W;
  const h = CELL_H;

  p.noStroke();
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textFont('Inconsolata');
  p.textStyle(p.NORMAL);
  p.textSize(LABEL_SIZE);
  p.textAlign(p.RIGHT, p.CENTER);
  p.text('Ref', x - LABEL_GAP, y + h / 2);

  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);
}

/** Paints the ref bases onto any p5-compatible draw target (canvas or offscreen graphics). */
export function paintRefBases(
  target: p5 | p5.Graphics,
  reference: Base[],
  windowStart: number,
): void {
  target.noStroke();
  target.textFont('Inconsolata');
  target.textStyle(target.NORMAL);
  target.textSize(BASE_SIZE);
  target.textAlign(target.CENTER, target.CENTER);
  for (let i = 0; i < REF_COUNT; i++) {
    const base: Base = reference[windowStart + i] ?? 'N';
    const c = igvColor(base);
    target.fill(c[0], c[1], c[2]);
    target.text(base, i * CELL_W + CELL_W / 2, CELL_H / 2 + 1);
  }
}
