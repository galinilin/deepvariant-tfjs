"""Extract a contiguous chr20 region (reads + reference + DV-emitted
candidates) into the static-asset fixture bundle that v5.1's
real-bam world-builder loads at startup.

Output bundle in public/fixtures/browser-region/:
  meta.json         { chrom, region_start, region_end, n_reads, n_candidates }
  reference.txt     plain ASCII reference sequence (one line, region length chars)
  reads.json        JSON array of Read objects in our TS schema (~1.8 MB
                    plain, but the dev server + GH Pages both
                    transport-compress, so wire size is ~200 KB)
  candidates.json   DV-emitted candidates with positions in region-relative coords

Inputs (hardcoded — DV r1.8 testdata):
  /tmp/dv-source/deepvariant/testdata/input/NA12878_S1.chr20.10_10p1mb.bam
  /tmp/dv-source/deepvariant/testdata/input/ucsc.hg19.chr20.unittest.fasta.gz
  /root/dv-tfjs/testdata/golden.calling_examples.tfrecord.gz-*

Usage (from /root/deepvariant-tf-js):
  npm run extract-browser-region

Or directly:
  cd /root/dv-tfjs
  uv run python /root/deepvariant-tf-js/scripts/extract_browser_region.py \
      [chrom=chr20] [start=10000000] [end=10005000]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pysam
import tensorflow as tf

OUT_DIR = Path("/root/deepvariant-tf-js/public/fixtures/browser-region")

DV_TFJS = Path("/root/dv-tfjs")
TFRECORDS = sorted(
    (DV_TFJS / "testdata").glob("golden.calling_examples.tfrecord.gz-*")
)
BAM = Path(
    "/tmp/dv-source/deepvariant/testdata/input/NA12878_S1.chr20.10_10p1mb.bam"
)
FASTA = Path(
    "/tmp/dv-source/deepvariant/testdata/input/ucsc.hg19.chr20.unittest.fasta.gz"
)

DEFAULT_CHROM = "chr20"
DEFAULT_START = 10_000_000
DEFAULT_END = 10_005_000  # 5 kb keeps the static asset around 250 KB gzipped

# DV's candidate-generation mapq filter (less strict than the pileup-image
# mapq>=10 — that one is applied inside our TS encoder).
MIN_MAPPING_QUALITY = 5

FEATURE_SPEC = {
    "image/encoded": tf.io.FixedLenFeature((), tf.string),
    "image/shape": tf.io.FixedLenFeature((3,), tf.int64),
    "variant/encoded": tf.io.FixedLenFeature((), tf.string),
    "alt_allele_indices/encoded": tf.io.FixedLenFeature((), tf.string),
}


# --- proto parsers --------------------------------------------------------


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
            raise ValueError(f"unsupported wire type {wire}")
    return out


def parse_alt_indices_proto(buf: bytes) -> list[int]:
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
                raise ValueError(f"unexpected wire type {wire}")
        else:
            if wire == 0:
                _, i = _parse_varint(buf, i)
            elif wire == 2:
                ln, i = _parse_varint(buf, i)
                i += ln
            elif wire == 1:
                i += 8
            elif wire == 5:
                i += 4
    return out


# --- read extraction -----------------------------------------------------


def extract_reads(
    bam: pysam.AlignmentFile, chrom: str, region_start: int, region_end: int
) -> list[dict]:
    """Pull every qualifying read overlapping the region. Convert to our JS
    Read schema; startCol becomes region-relative."""
    reads_out: list[dict] = []
    for read in bam.fetch(chrom, region_start, region_end):
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
        ref_start = read.reference_start  # absolute genomic
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

        # Insertions
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

        # Clip read to region: drop bases outside [region_start, region_end)
        # and adjust startCol. Reads spanning the boundary keep only their
        # in-region portion.
        clipped_start = max(ref_start, region_start)
        clipped_end = min(ref_end, region_end)
        if clipped_end <= clipped_start:
            continue
        slice_lo = clipped_start - ref_start
        slice_hi = clipped_end - ref_start
        clipped_bases = bases[slice_lo:slice_hi]
        clipped_qualities = qualities[slice_lo:slice_hi]
        # Adjust insertion offsets — keep insertions whose anchor sits inside
        # the clipped range, with offsets re-indexed to clipped startCol.
        clipped_insertions = []
        for ins in insertions:
            new_offset = ins["offset"] - slice_lo
            if 0 <= new_offset < (slice_hi - slice_lo):
                clipped_insertions.append(
                    {
                        "offset": new_offset,
                        "bases": ins["bases"],
                        "qualities": ins["qualities"],
                    }
                )

        reads_out.append(
            {
                "id": read.query_name,
                # startCol relative to region_start (so the encoder sees the
                # same coordinate space as the reference array).
                "startCol": clipped_start - region_start,
                "bases": clipped_bases,
                "qualities": clipped_qualities,
                "strand": "reverse" if read.is_reverse else "forward",
                "mapq": int(read.mapping_quality),
                "insertSize": abs(int(read.template_length or 0)),
                "row": 0,
                "insertions": clipped_insertions if clipped_insertions else None,
            }
        )
    return reads_out


# --- candidate extraction ------------------------------------------------


def kind_from_alleles(ref_bases: str, alt_bases: str) -> str:
    if len(ref_bases) == 1 and len(alt_bases) == 1:
        return "snv"
    if len(ref_bases) > len(alt_bases) and alt_bases and ref_bases.startswith(alt_bases):
        return "del"
    if len(alt_bases) > len(ref_bases) and ref_bases and alt_bases.startswith(ref_bases):
        return "ins"
    return "complex"


def extract_candidates(
    chrom: str, region_start: int, region_end: int
) -> list[dict]:
    """Pull DV's emitted candidates from the calling examples TFRecord and
    keep just those falling inside [region_start, region_end). Multi-allelic
    examples surface as separate entries (one per alt_indices combo)."""
    out: list[dict] = []
    ds = tf.data.TFRecordDataset(
        [str(p) for p in TFRECORDS], compression_type="GZIP"
    ).map(lambda r: tf.io.parse_single_example(r, FEATURE_SPEC))

    for feats in ds:
        var_buf = bytes(feats["variant/encoded"].numpy())
        alt_buf = bytes(feats["alt_allele_indices/encoded"].numpy())
        v = parse_variant_proto(var_buf)
        if v["reference_name"] != chrom:
            continue
        pos = v["start"]
        if pos is None or pos < region_start or pos >= region_end:
            continue
        alt_indices = parse_alt_indices_proto(alt_buf)
        emitted_alts = (
            [v["alternate_bases"][k] for k in alt_indices if 0 <= k < len(v["alternate_bases"])]
            if v["alternate_bases"]
            else []
        )
        primary_alt = emitted_alts[0] if emitted_alts else None
        primary_kind = (
            kind_from_alleles(v["reference_bases"] or "", primary_alt) if primary_alt else "unknown"
        )
        out.append(
            {
                "position": pos - region_start,  # region-relative
                "position_genomic": pos,
                "ref_alleles": v["reference_bases"],
                "alt_alleles": v["alternate_bases"],
                "alt_indices": alt_indices,
                "emitted_alts": emitted_alts,
                "primary_kind": primary_kind,
            }
        )

    out.sort(key=lambda c: c["position"])
    return out


# --- main ---------------------------------------------------------------


def main() -> None:
    chrom = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHROM
    start = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_START
    end = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_END

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] BAM {BAM.name}")
    bam = pysam.AlignmentFile(str(BAM), "rb")
    print(f"[load] FASTA {FASTA.name}")
    fasta = pysam.FastaFile(str(FASTA))

    print(f"[fetch] {chrom}:{start}-{end} ({end - start} bp)")

    ref_seq = fasta.fetch(chrom, start, end).upper()
    assert len(ref_seq) == end - start, f"reference length mismatch: {len(ref_seq)}"
    (OUT_DIR / "reference.txt").write_text(ref_seq)
    print(f"[ref ] wrote {len(ref_seq)} bp → reference.txt")

    reads = extract_reads(bam, chrom, start, end)
    print(f"[reads] extracted {len(reads)} reads")

    candidates = extract_candidates(chrom, start, end)
    print(
        f"[cand ] {len(candidates)} DV-emitted candidates: kinds = "
        + ", ".join(
            sorted({c["primary_kind"] for c in candidates})
        )
    )

    # Store uncompressed: Vite auto-decompresses .gz files which collides
    # with our DecompressionStream. Both dev server and GH Pages
    # transport-compress on the wire, so wire size still small.
    (OUT_DIR / "reads.json").write_text(json.dumps(reads))
    reads_size = (OUT_DIR / "reads.json").stat().st_size
    print(f"[reads] wrote reads.json ({reads_size/1024:.1f} KB)")

    (OUT_DIR / "candidates.json").write_text(json.dumps(candidates, indent=2))
    cand_size = (OUT_DIR / "candidates.json").stat().st_size
    print(f"[cand ] wrote candidates.json ({cand_size/1024:.1f} KB)")

    meta = {
        "chrom": chrom,
        "region_start": start,
        "region_end": end,
        "region_length": end - start,
        "bam": BAM.name,
        "fasta": FASTA.name,
        "n_reads": len(reads),
        "n_candidates": len(candidates),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[done ] bundle ready → {OUT_DIR}/")


if __name__ == "__main__":
    main()
