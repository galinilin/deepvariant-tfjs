export interface NpyArray {
  data: Float32Array;
  shape: number[];
}

export function readNpyFloat32(buf: ArrayBuffer): NpyArray {
  const view = new DataView(buf);
  const magic = String.fromCharCode(...new Uint8Array(buf, 0, 6));
  if (magic !== '\x93NUMPY') throw new Error('not a .npy file');
  const major = view.getUint8(6);
  let headerLen: number;
  let dataStart: number;
  if (major === 1) {
    headerLen = view.getUint16(8, true);
    dataStart = 10 + headerLen;
  } else {
    headerLen = view.getUint32(8, true);
    dataStart = 12 + headerLen;
  }
  const headerOffset = major === 1 ? 10 : 12;
  const header = new TextDecoder('ascii').decode(
    new Uint8Array(buf, headerOffset, headerLen),
  );
  const shapeMatch = header.match(/'shape':\s*\(([^)]*)\)/);
  const dtypeMatch = header.match(/'descr':\s*'([^']+)'/);
  if (!shapeMatch || !dtypeMatch) throw new Error(`unparseable .npy header: ${header}`);
  if (dtypeMatch[1] !== '<f4') throw new Error(`unsupported dtype ${dtypeMatch[1]}`);
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  // Float32Array view requires byteOffset%4 == 0; header padding is not always
  // 4-aligned, so slice into a fresh buffer.
  const slice = buf.slice(dataStart, dataStart + n * 4);
  return { data: new Float32Array(slice), shape };
}
