#!/usr/bin/env bash
#
# Derive the validation preset copies from the LIVE production ini files.
#
# Deriving instead of keeping hand-written copies matters: the fleet config changes often, and a
# stale copy would validate a configuration nobody runs. This reads /opt, writes into the repo,
# and never writes back to /opt.
#
# It also strips keys that the current build no longer understands. The preset parser does NOT
# ignore unknown keys - common/preset.cpp throws "option '<k>' not recognized in preset '<name>'"
# and server-models.cpp does not catch it, so a single stale key stops the router from starting.
#
# Stripping here is a TEST-TIME accommodation only. The real fix is to remove the key from the
# production ini, which is a deployment step requiring approval - not something this script does.
#
set -euo pipefail

SRC_DIR=${SRC_DIR:-/opt}
DST_DIR=${DST_DIR:-$(cd "$(dirname "$0")/.." && pwd)/validation/presets}

# Keys removed upstream that still appear in production presets.
# Add to this list as future rebases drop more options; each entry needs a reason.
STRIP_KEYS=(
    "rocwmma-fattn"   # upstream removed rocWMMA FA entirely (fa72aeccb, #26046)
)

mkdir -p "$DST_DIR"

for src in "$SRC_DIR"/modelsBOTH.ini "$SRC_DIR"/modelsUNIFIED.ini; do
    [ -f "$src" ] || { echo "missing $src - run this on the server" >&2; exit 1; }
    dst="$DST_DIR/$(basename "$src")"
    cp "$src" "$dst"

    for key in "${STRIP_KEYS[@]}"; do
        n=$(grep -cE "^[[:space:]]*${key}[[:space:]]*=" "$dst" || true)
        if [ "$n" -gt 0 ]; then
            sed -i -E "/^[[:space:]]*${key}[[:space:]]*=/d" "$dst"
            echo "  stripped $n x '$key' from $(basename "$src")"
        fi
    done

    echo "  wrote $dst"
done

echo ""
echo "Sanity: any key in the copies that the build does not define will abort the server."
echo "The parse check in validate.sh (T0.2) is what proves this worked - it is not optional."
