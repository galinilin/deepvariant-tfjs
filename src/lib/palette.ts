export type Base = 'A' | 'C' | 'G' | 'T' | 'N';

const IGV: Record<Base, [number, number, number]> = {
  A: [120, 210, 130],
  C: [130, 180, 240],
  G: [230, 195, 105],
  T: [240, 130, 135],
  N: [160, 160, 160],
};

export function igvColor(base: Base): [number, number, number] {
  return IGV[base] ?? IGV.N;
}
