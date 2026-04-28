import type p5 from 'p5';
import { igvColor } from '../../lib/palette';
import type { Read } from '../../lib/reads';

export const READ_ROW_H = 18;

const LABEL_GAP = 12;
const LABEL_SIZE = 14;
const BASE_SIZE = 14;

const LABEL_COLOR: [number, number, number] = [210, 210, 210];
const BORDER_COLOR: [number, number, number] = [48, 48, 48];

export interface ReadsState {
  origin: { x: number; y: number };
  readsCount: number;
  width: number;
}

export function drawReadsFrame(p: p5, state: ReadsState): void {
  const { x, y } = state.origin;
  const w = state.width;
  const h = state.readsCount * READ_ROW_H;

  p.noStroke();
  p.fill(LABEL_COLOR[0], LABEL_COLOR[1], LABEL_COLOR[2]);
  p.textFont('Inconsolata');
  p.textStyle(p.NORMAL);
  p.textSize(LABEL_SIZE);
  p.textAlign(p.RIGHT, p.TOP);
  p.text('Reads', x - LABEL_GAP, y);

  p.noFill();
  p.stroke(BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2]);
  p.strokeWeight(0.5);
  p.rect(x, y, w, h);
}

/**
 * Paints all reads at their absolute genomic positions onto the target.
 * The target buffer is expected to be wide enough to hold the full
 * reference length (i.e., reference.length * cellW).
 *
 * Uses the underlying CanvasRenderingContext2D directly because p5 v2's
 * textAlign() does not apply reliably to p5.Graphics buffers.
 */
export function paintReadsBases(
  target: p5 | p5.Graphics,
  reads: Read[],
  cellW: number,
): void {
  const ctx = target.drawingContext as CanvasRenderingContext2D;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = `${BASE_SIZE}px Inconsolata, ui-monospace, Menlo, monospace`;

  for (let r = 0; r < reads.length; r++) {
    const read = reads[r];
    const rowY = r * READ_ROW_H + READ_ROW_H / 2;
    for (let i = 0; i < read.bases.length; i++) {
      const absCol = read.startCol + i;
      const base = read.bases[i];
      const c = igvColor(base);
      ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
      ctx.fillText(base, absCol * cellW + cellW / 2, rowY);
    }
  }
}
