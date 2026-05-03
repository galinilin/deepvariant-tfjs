"""Extract ALL DV calling examples from the testdata BAM into a unified
fixture bundle that match/3.1's TS validator consumes.

Combines the work of `extract_golden.py` + `extract_reads_for_golden.py`
into one pass. Emits:

  <out>/manifest.json    per-candidate metadata
                         { n, bam, samples: [{index, chrom, position, ...}] }
  <out>/sample_<i>.json  reads + 221-bp ref window per example
                         { reads, ref_window, position_in_window=110, ... }
  <out>/pileups.npy      float32, shape [N, 100, 221, 7]   DV's golden tensors
  <out>/outputs.npy      float32, shape [N, 3]             DV's softmax outputs

Inputs (hardcoded — DV r1.8 testdata):
  /tmp/dv-source/deepvariant/testdata/input/NA12878_S1.chr20.10_10p1mb.bam
  /tmp/dv-source/deepvariant/testdata/input/ucsc.hg19.chr20.unittest.fasta.gz
  /root/dv-tfjs/testdata/golden.calling_examples.tfrecord.gz-*

Usage (from /root/deepvariant-tf-js):
  npm run extract-match-fixtures

Or directly:
  cd /root/dv-tfjs
  uv run python /root/deepvariant-tf-js/scripts/extract_match_fixtures.py <out_dir>
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pysam
import tensorflow as tf

DV_TFJS = Path("/root/dv-tfjs")
sys.path.insert(0, str(DV_TFJS / "vendor"))
from dv_model import inceptionv3  # noqa: E402

CKPT = DV_TFJS / "checkpoints" / "wgs" / "deepvariant.wgs.ckpt"
TFRECORDS = sorted(
    (DV_TFJS / "testdata").glob("golden.calling_examples.tfrecord.gz-*")
)
BAM = Path(
    "/tmp/dv-source/deepvariant/testdata/input/NA12878_S1.chr20.10_10p1mb.bam"
)
FASTA = Path(
    "/tmp/dv-source/deepvariant/testdata/input/ucsc.hg19.chr20.unittest.fasta.gz"
)

PILEUP_HALF_WIDTH = 110
PILEUP_WIDTH = 221
INPUT_SHAPE = (100, 221, 7)
CLASSES = ("hom_ref", "het", "hom_alt")

# DV's candidate-generation mapq filter (less strict than the pileup-image
# mapq>=10 — that one is applied inside our TS encoder).
MIN_MAPPING_QUALITY = 5

FEATURE_SPEC = {
    "image/encoded": tf.io.FixedLenFeature((), tf.string),
    "image/shape": tf.io.FixedLenFeature((3,), tf.int64),
    "variant/encoded": tf.io.FixedLenFeature((), tf.string),
    "alt_allele_indices/encoded": tf.io.FixedLenFeature((), tf.string),
}


# --- proto parsing ---------------------------------------------------------


def _parse_varint(buf, i):
    val = 0
    shift = 0
    while True:
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, i
        shift += 7


def parse_variant_proto(buf: bytes) -> dict:
    """Minimal nucleus.genomics.v1.Variant proto reader.

    Field numbers from r1.8 examples:
      6:  reference_bases (string)
      7:  alternate_bases (string, repeated)
     13:  end (int64, exclusive)
     14:  reference_name (string)
     16:  start (int64, 0-based)
    """
    out = {
        "reference_name": None,
        "start": None,
        "end": None,
        "reference_bases": None,
        "alternate_bases": [],
    }
    i = 0
    while i < len(buf):
        tag, i = _parse_varint(buf, i)
        field = tag >> 3
        wire = tag & 7
        if wire == 0:
            val, i = _parse_varint(buf, i)
            if field == 16:
                out["start"] = val
            elif field == 13:
                out["end"] = val
        elif wire == 2:
            ln, i = _parse_varint(buf, i)
            chunk = buf[i : i + ln]
            i += ln
            if field == 14:
                out["reference_name"] = chunk.decode("ascii", errors="replace")
            elif field == 6:
                out["reference_bases"] = chunk.decode("ascii", errors="replace")
            elif field == 7:
                out["alternate_bases"].append(
                    chunk.decode("ascii", errors="replace")
                )
        elif wire == 1:
            i += 8
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"unsupported wire type {wire} at offset {i}")
    return out


def parse_alt_indices_proto(buf: bytes) -> list[int]:
    """AltAlleleIndices { repeated int32 indices = 1; }
    Field 1 may be packed (wire=2 with concatenated varints) or unpacked
    (one wire=0 entry per index). Handle both.
    """
    out: list[int] = []
    i = 0
    while i < len(buf):
        tag, i = _parse_varint(buf, i)
        field = tag >> 3
        wire = tag & 7
        if field == 1:
            if wire == 0:
                v, i = _parse_varint(buf, i)
                out.append(v)
            elif wire == 2:
                ln, i = _parse_varint(buf, i)
                end = i + ln
                while i < end:
                    v, i = _parse_varint(buf, i)
                    out.append(v)
            else:
                raise ValueError(
                    f"unexpected wire type {wire} for AltAlleleIndices.indices"
                )
        else:
            # Skip unknown fields
            if wire == 0:
                _, i = _parse_varint(buf, i)
            elif wire == 2:
                ln, i = _parse_varint(buf, i)
                i += ln
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
            else:
                raise ValueError(f"unsupported wire type {wire}")
    return out


# --- read extraction --------------------------------------------------------


def extract_reads(bam: pysam.AlignmentFile, chrom: str, position: int) -> list[dict]:
    """Pull all qualifying reads spanning the candidate window. Convert each
    to our JS Read schema, with insertions captured as a sparse sidecar."""
    reads_out: list[dict] = []
    fetch_start = position - PILEUP_HALF_WIDTH
    fetch_end = position + PILEUP_HALF_WIDTH + 1
    for read in bam.fetch(chrom, fetch_start, fetch_end):
        if (
            read.is_unmapped
            or read.is_secondary
            or read.is_supplementary
            or read.is_duplicate
            or read.is_qcfail
        ):
            continue
        if read.mapping_quality < MIN_MAPPING_QUALITY:
            continue
        if read.query_sequence is None:
            continue

        seq = read.query_sequence
        qual = read.query_qualities
        ref_start = read.reference_start
        ref_end = read.reference_end
        if ref_end is None or ref_end <= ref_start:
            continue

        aligned = read.get_aligned_pairs(matches_only=False)

        bases = ["N"] * (ref_end - ref_start)
        qualities = [0] * (ref_end - ref_start)
        for read_pos, ref_pos in aligned:
            if ref_pos is None or ref_pos < ref_start or ref_pos >= ref_end:
                continue
            offset = ref_pos - ref_start
            if read_pos is None:
                bases[offset] = "-"
                qualities[offset] = 0
            else:
                bases[offset] = seq[read_pos]
                qualities[offset] = int(qual[read_pos]) if qual is not None else 0

        # Insertions: contiguous runs of (read_pos != None, ref_pos None) that
        # are inside the alignment span (anchor exists). Skip leading/trailing
        # soft clips.
        insertions = []
        i = 0
        while i < len(aligned):
            read_pos, ref_pos = aligned[i]
            if ref_pos is None and read_pos is not None:
                anchor_offset = None
                for j in range(i - 1, -1, -1):
                    rp_back = aligned[j][1]
                    if rp_back is not None:
                        anchor_offset = rp_back - ref_start
                        break
                if anchor_offset is None:
                    while i < len(aligned) and aligned[i][1] is None:
                        i += 1
                    continue
                ins_bases = []
                ins_quals = []
                while (
                    i < len(aligned)
                    and aligned[i][1] is None
                    and aligned[i][0] is not None
                ):
                    rp_a = aligned[i][0]
                    ins_bases.append(seq[rp_a])
                    ins_quals.append(int(qual[rp_a]) if qual is not None else 0)
                    i += 1
                if ins_bases:
                    insertions.append(
                        {
                            "offset": anchor_offset,
                            "bases": ins_bases,
                            "qualities": ins_quals,
                        }
                    )
                continue
            i += 1

        reads_out.append(
            {
                "id": read.query_name,
                "startCol": ref_start,
                "bases": bases,
                "qualities": qualities,
                "strand": "reverse" if read.is_reverse else "forward",
                "mapq": int(read.mapping_quality),
                "insertSize": abs(int(read.template_length or 0)),
                "row": 0,
                "insertions": insertions if insertions else None,
            }
        )
    return reads_out


# --- main -------------------------------------------------------------------


def kind_from_alleles(ref_bases: str, alt_bases: str) -> str:
    if len(ref_bases) == 1 and len(alt_bases) == 1:
        return "snv"
    if len(ref_bases) > len(alt_bases) and alt_bases and ref_bases.startswith(alt_bases):
        return "del"
    if len(alt_bases) > len(ref_bases) and ref_bases and alt_bases.startswith(ref_bases):
        return "ins"
    return "complex"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] checkpoint {CKPT}")
    model = inceptionv3(INPUT_SHAPE)
    model.load_weights(str(CKPT)).expect_partial()

    print(f"[open] BAM {BAM.name}")
    bam = pysam.AlignmentFile(str(BAM), "rb")
    fasta = pysam.FastaFile(str(FASTA))

    print(f"[load] {len(TFRECORDS)} TFRecord shards (no take cap)")
    ds = tf.data.TFRecordDataset(
        [str(p) for p in TFRECORDS], compression_type="GZIP"
    ).map(lambda r: tf.io.parse_single_example(r, FEATURE_SPEC))

    images: list[np.ndarray] = []
    samples_meta: list[dict] = []
    sample_files: list[tuple[int, dict]] = []

    for idx, feats in enumerate(ds):
        img_bytes = feats["image/encoded"].numpy()
        img_shape = list(feats["image/shape"].numpy())
        var_buf = bytes(feats["variant/encoded"].numpy())
        alt_buf = bytes(feats["alt_allele_indices/encoded"].numpy())

        image = np.frombuffer(img_bytes, dtype=np.uint8).reshape(img_shape).astype(
            np.float32
        )

        variant = parse_variant_proto(var_buf)
        alt_indices = parse_alt_indices_proto(alt_buf)
        chrom = variant["reference_name"]
        position = variant["start"]
        ref_alleles = variant["reference_bases"] or ""
        alt_alleles = variant["alternate_bases"]

        # The "alt this example is for" — primary alt for level-1 scoring.
        emitted_alts = (
            [alt_alleles[k] for k in alt_indices if 0 <= k < len(alt_alleles)]
            if alt_alleles
            else []
        )
        primary_alt = emitted_alts[0] if emitted_alts else None
        primary_kind = (
            kind_from_alleles(ref_alleles, primary_alt) if primary_alt else "unknown"
        )

        # Reference window
        ref_start = position - PILEUP_HALF_WIDTH
        ref_end = position + PILEUP_HALF_WIDTH + 1
        ref_seq = fasta.fetch(chrom, ref_start, ref_end).upper()

        # Reads + window-relative startCol
        reads = extract_reads(bam, chrom, position)
        for r in reads:
            r["startCol"] = r["startCol"] - ref_start

        sample = {
            "index": idx,
            "chrom": chrom,
            "position_genomic": position,
            "position_in_window": PILEUP_HALF_WIDTH,
            "ref_window_start": ref_start,
            "ref_window": ref_seq,
            "ref_alleles": ref_alleles,
            "alt_alleles": alt_alleles,
            "alt_indices": alt_indices,
            "emitted_alts": emitted_alts,
            "primary_kind": primary_kind,
            "n_reads": len(reads),
            "reads": reads,
        }

        sample_files.append((idx, sample))
        samples_meta.append(
            {
                "index": idx,
                "chrom": chrom,
                "position_genomic": position,
                "position_1_based": position + 1,
                "ref_alleles": ref_alleles,
                "alt_alleles": alt_alleles,
                "alt_indices": alt_indices,
                "emitted_alts": emitted_alts,
                "primary_kind": primary_kind,
                "n_reads": len(reads),
            }
        )
        images.append(image)

        kind_label = (
            primary_kind if len(emitted_alts) <= 1 else f"{primary_kind}+multi"
        )
        print(
            f"[{idx:>3}] {chrom}:{position + 1} "
            f"{ref_alleles}>{','.join(alt_alleles)} idx={alt_indices} "
            f"kind={kind_label} reads={len(reads)}"
        )

    if not images:
        print("[fail] no examples found", file=sys.stderr)
        sys.exit(1)

    stacked = np.stack(images, axis=0)
    print(
        f"\n[parse] {stacked.shape[0]} examples, "
        f"range=[{stacked.min():.0f}, {stacked.max():.0f}], dtype={stacked.dtype}"
    )

    print("[pred ] running upstream Keras model on all pileups…")
    probs = model.predict(stacked, verbose=0)
    if isinstance(probs, list):
        probs = probs[0]
    probs = probs.astype(np.float32)

    # Write per-sample JSON (large: reads list); manifest carries metadata.
    for idx, sample in sample_files:
        argmax_idx = int(np.argmax(probs[idx]))
        sample["dv_argmax"] = CLASSES[argmax_idx]
        sample["dv_probs"] = {
            "hom_ref": float(probs[idx][0]),
            "het": float(probs[idx][1]),
            "hom_alt": float(probs[idx][2]),
        }
        (out_dir / f"sample_{idx}.json").write_text(json.dumps(sample))

    for i, meta in enumerate(samples_meta):
        meta["dv_argmax"] = CLASSES[int(np.argmax(probs[i]))]

    # Pileups are all in [0, 254] (DV's uint8 internal). Save as gzipped
    # uint8 .npy.gz — shrinks from ~50 MB float32 to ~0.6 MB. The TS side
    # reads, gunzips, casts to float32 for model input.
    pileups_u8 = stacked.astype(np.uint8)
    with gzip.open(out_dir / "pileups.npy.gz", "wb") as gz:
        np.save(gz, pileups_u8)  # type: ignore[arg-type]
    np.save(out_dir / "outputs.npy", probs)

    manifest = {
        "n": stacked.shape[0],
        "bam": BAM.name,
        "fasta": FASTA.name,
        "tfrecords": [p.name for p in TFRECORDS],
        "shape": list(stacked.shape),
        "samples": samples_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    kind_counts: dict[str, int] = {}
    for m in samples_meta:
        kind_counts[m["primary_kind"]] = kind_counts.get(m["primary_kind"], 0) + 1
    print("\n[done ] kind breakdown: " + ", ".join(f"{k}={v}" for k, v in kind_counts.items()))
    print(f"[done ] wrote {stacked.shape[0]} samples → {out_dir}/")


if __name__ == "__main__":
    main()
