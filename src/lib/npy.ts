export interface NpyArray<T = Float32Array> {
  data: T;
  shape: number[];
}

interface NpyHeader {
  dtype: string;
  shape: number[];
  dataStart: number;
  n: number;
}

function parseHeader(buf: ArrayBuffer): NpyHeader {
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
  const shape = shapeMatch[1]
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const n = shape.reduce((a, b) => a * b, 1);
  return { dtype: dtypeMatch[1], shape, dataStart, n };
}

export function readNpyFloat32(buf: ArrayBuffer): NpyArray<Float32Array> {
  const h = parseHeader(buf);
  if (h.dtype !== '<f4') throw new Error(`expected <f4 dtype, got ${h.dtype}`);
  // Float32Array view requires byteOffset%4 == 0; header padding is not always
  // 4-aligned, so slice into a fresh buffer.
  const slice = buf.slice(h.dataStart, h.dataStart + h.n * 4);
  return { data: new Float32Array(slice), shape: h.shape };
}

export function readNpyUint8(buf: ArrayBuffer): NpyArray<Uint8Array> {
  const h = parseHeader(buf);
  if (h.dtype !== '|u1' && h.dtype !== '<u1' && h.dtype !== 'u1') {
    throw new Error(`expected uint8 dtype, got ${h.dtype}`);
  }
  const slice = buf.slice(h.dataStart, h.dataStart + h.n);
  return { data: new Uint8Array(slice), shape: h.shape };
}
