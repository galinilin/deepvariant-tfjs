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
  /** Where the hover originated — used to suppress circular feedback
   * (e.g. don't auto-pan the top canvas when the user is hovering the
   * top canvas itself) and to relax channel-match checks for hovers
   * that don't carry a meaningful channel. */
  source: 'top' | 'bottom';
  /** Genomic position (column index in the reference) that the user is
   * hovering. The top canvas draws a vertical highlight at this column. */
  genomicPos: number;
  /** Encoder image-row 0..99 (0..4 = ref, 5..99 = read rows). */
  imageRow: number;
  /** read.id at imageRow if any, else null (ref row or empty padding). */
  readId: string | null;
  /** Tensor cell value at (row, col, channel) for tooltip display.
   * 0 if unknown (e.g. when hover originates from the top canvas). */
  cellValue: number;
  /** Channel index that the cellValue is sampled from, or -1 if unknown
   * (top-canvas hover doesn't track which channel is active). */
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
  /** How the encoder orders read rows in the 100×221 image.
   *
   *   'igv-aligned' (default): sort by (IGV-pack-row, startCol). Each
   *     IGV row's reads land contiguously in the encoder image, so
   *     encoder-row N visually corresponds to top-canvas IGV-row N-5
   *     (subject to filtering). Friendlier hover correspondence.
   *
   *   'dv-style': sort by (startCol, id) globally. Matches DV's
   *     production sort (modulo haplotype tags we don't carry) and
   *     produces the diagonal "top-left to bottom-right" stripe seen
   *     in DV's blog images. The model output is essentially identical
   *     either way (InceptionV3 is row-order robust) — this is purely
   *     a visualization choice.
   */
  rowSort: 'igv-aligned' | 'dv-style';
  /** Latest encoder row→read.id mapping. Bottom canvas publishes after
   * each successful predict; top canvas reads it to resolve a read's
   * encoder row for cross-canvas hover linking. */
  latestRowToReadId: (string | null)[] | null;
} = {
  candidate: null,
  reads: null,
  reference: null,
  predictPos: null,
  readsGeneration: 0,
  hover: null,
  autoFocus: true,
  rowSort: 'igv-aligned',
  latestRowToReadId: null,
};
