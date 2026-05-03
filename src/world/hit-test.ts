import type { Read } from '../lib/reads';
import type { Base, Cell } from '../lib/palette';

export type ReadHit =
  | {
      kind: 'cell';
      read: Read;
      absCol: number;
      cellIdx: number;
      base: Cell;
      quality: number;
    }
  | {
      kind: 'insertion';
      read: Read;
      absCol: number;
      offset: number;
      sequence: Base[];
      qualities: Uint8Array;
    };

const INSERTION_TICK_HIT_HALF = 3; // ±3 px around the column boundary

/**
 * Resolve a world-coordinate point to a (read, cell) hit, if the point lies
 * inside the visible reads region and there is a packed read at that row
 * covering the column. Returns null otherwise.
 *
 * The reads region renders a windowed slice of the full pileup: the cache
 * is the full ref width, but `image()` blits only the windowStart..+window
 * range into the visible region. So world coords have to be translated back
 * into absolute genomic coords (cache coord space) before comparing against
 * read positions.
 *
 * Insertion ticks (between-column events) are tested first: hovering on a
 * tick returns a `'insertion'` hit, otherwise the standard `'cell'` hit.
 */
export function hitTestReads(
  wx: number,
  wy: number,
  reads: Read[],
  readsOrigin: { x: number; y: number },
  visibleColumns: number,
  cellW: number,
  rowH: number,
  maxRows: number,
  windowStart: number,
): ReadHit | null {
  const dx = wx - readsOrigin.x;
  if (dx < 0 || dx >= visibleColumns * cellW) return null;

  const dy = wy - readsOrigin.y;
  if (dy < 0 || dy >= maxRows * rowH) return null;

  const rowIdx = Math.floor(dy / rowH);
  if (rowIdx < 0 || rowIdx >= maxRows) return null;

  const absX = dx + windowStart * cellW;

  for (const read of reads) {
    if (read.row !== rowIdx) continue;
    const startX = read.startCol * cellW;
    const endX = (read.startCol + read.bases.length) * cellW;
    if (absX < startX || absX >= endX) continue;

    if (read.insertions) {
      for (const ins of read.insertions) {
        const tickCenterX = (read.startCol + ins.offset + 1) * cellW;
        if (Math.abs(absX - tickCenterX) <= INSERTION_TICK_HIT_HALF) {
          return {
            kind: 'insertion',
            read,
            absCol: read.startCol + ins.offset,
            offset: ins.offset,
            sequence: ins.bases,
            qualities: ins.qualities,
          };
        }
      }
    }

    const cellIdx = Math.floor((absX - startX) / cellW);
    if (cellIdx < 0 || cellIdx >= read.bases.length) continue;

    return {
      kind: 'cell',
      read,
      absCol: read.startCol + cellIdx,
      cellIdx,
      base: read.bases[cellIdx],
      quality: read.qualities[cellIdx] ?? 0,
    };
  }
  return null;
}
