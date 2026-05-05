import type { Candidate } from './candidate';
import type { Read } from './reads';
import type { Base } from './palette';

/**
 * Tiny shared state object so the top and bottom p5 sketches can talk
 * without a heavier event bus. Top writes; bottom reads.
 *
 * v3.2 expansion: bottom canvas needs reads + reference + the predict
 * position to encode the pileup tensor and run inference. We also carry
 * a `readsGeneration` counter that increments every time the top sketch
 * regenerates the world (Randomize) — bottom uses it to know when to
 * invalidate its cached prediction.
 *
 * v6.0 expansion: bottom-pixel hover writes a `hover` record that the
 * top canvas reads to render a column-highlight + read-outline. Cleared
 * to null when mouse leaves the channel image.
 */
export interface ChannelHover {
  /** Genomic position (column index in the reference) that the user is
   * hovering. The top canvas draws a vertical highlight at this column. */
  genomicPos: number;
  /** Encoder image-row 0..99 (0..4 = ref, 5..99 = read rows). */
  imageRow: number;
  /** read.id at imageRow if any, else null (ref row or empty padding). */
  readId: string | null;
  /** Tensor cell value at (row, col, channel) for tooltip display. */
  cellValue: number;
  /** Channel index that's currently active (0..6). */
  channel: number;
}

export const sandboxState: {
  candidate: Candidate;
  reads: Read[] | null;
  reference: Base[] | null;
  predictPos: number | null;
  readsGeneration: number;
  hover: ChannelHover | null;
  /** When true (default), hovering a pixel in the bottom canvas's
   * active channel pans + zooms the top canvas to center on the
   * corresponding (column, read row). When false, hover still draws
   * highlights (outline + column line + crosshair) but the camera
   * doesn't move. Toggled by the corner "Auto-focus" button. */
  autoFocus: boolean;
} = {
  candidate: null,
  reads: null,
  reference: null,
  predictPos: null,
  readsGeneration: 0,
  hover: null,
  autoFocus: true,
};
