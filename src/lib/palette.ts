export type Base = 'A' | 'C' | 'G' | 'T' | 'N';
export type Cell = Base | '-';

const IGV: Record<Cell, [number, number, number]> = {
  A: [120, 210, 130],
  C: [130, 180, 240],
  G: [230, 195, 105],
  T: [240, 130, 135],
  N: [160, 160, 160],
  '-': [85, 85, 85],
};

export function igvColor(cell: Cell): [number, number, number] {
  return IGV[cell] ?? IGV.N;
}
