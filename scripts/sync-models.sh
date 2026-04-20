#!/usr/bin/env bash
# Hydrate public/ from the sibling dv-tfjs build outputs.
#
# Usage:
#   scripts/sync-models.sh [full|uint8-only]   (default: full)
#
# full        Copies both float32 (tfjs_dv_wgs) and uint8 (tfjs_dv_wgs_uint8)
#             models, plus parity fixtures. Use for local dev so both
#             precisions are available via DeepVariantModel.load().
#
# uint8-only  Copies only the uint8 model + fixtures, AND removes any
#             existing public/models/tfjs_dv_wgs/ directory. Used by
#             `npm run predeploy` so the Pages build stays lean (~24 MB
#             of artifacts instead of ~110 MB).
#
# Override source dir with DV_TFJS_OUT=/path/to/out. Re-runnable;
# existing files are overwritten.
set -euo pipefail

MODE="${1:-full}"
case "$MODE" in
  full|uint8-only) ;;
  *) echo "usage: $0 [full|uint8-only]" >&2; exit 1 ;;
esac

HERE="$(cd "$(dirname "$0")/.." && pwd)"
SRC="${DV_TFJS_OUT:-/root/dv-tfjs/out}"

if [[ ! -d "$SRC" ]]; then
  echo "error: dv-tfjs build dir not found at $SRC" >&2
  echo "       set DV_TFJS_OUT to override, or build dv-tfjs first." >&2
  exit 1
fi

mkdir -p "$HERE/public/models" "$HERE/public/fixtures"

# uint8 model: always synced
if [[ ! -d "$SRC/tfjs_dv_wgs_uint8" ]]; then
  echo "error: missing $SRC/tfjs_dv_wgs_uint8" >&2
  exit 1
fi
echo "  sync  $SRC/tfjs_dv_wgs_uint8 -> public/models/tfjs_dv_wgs_uint8"
rm -rf "$HERE/public/models/tfjs_dv_wgs_uint8"
cp -r "$SRC/tfjs_dv_wgs_uint8" "$HERE/public/models/tfjs_dv_wgs_uint8"

# float32 model: synced only in full mode; removed in uint8-only mode
if [[ "$MODE" == "full" ]]; then
  if [[ ! -d "$SRC/tfjs_dv_wgs" ]]; then
    echo "error: missing $SRC/tfjs_dv_wgs" >&2
    exit 1
  fi
  echo "  sync  $SRC/tfjs_dv_wgs -> public/models/tfjs_dv_wgs"
  rm -rf "$HERE/public/models/tfjs_dv_wgs"
  cp -r "$SRC/tfjs_dv_wgs" "$HERE/public/models/tfjs_dv_wgs"
else
  if [[ -d "$HERE/public/models/tfjs_dv_wgs" ]]; then
    echo "  drop  public/models/tfjs_dv_wgs  (uint8-only deploy)"
    rm -rf "$HERE/public/models/tfjs_dv_wgs"
  fi
fi

# Parity fixtures: always synced
for f in parity_inputs.npy parity_outputs.npy; do
  if [[ ! -f "$SRC/$f" ]]; then
    echo "error: missing $SRC/$f" >&2
    exit 1
  fi
  echo "  sync  $SRC/$f -> public/fixtures/$f"
  cp "$SRC/$f" "$HERE/public/fixtures/$f"
done

du -sh "$HERE/public/models" "$HERE/public/fixtures"
echo "done. ($MODE)"
