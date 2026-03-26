#!/usr/bin/env bash
# Convenience wrapper — sets LD_LIBRARY_PATH from the libtorch directory.
#
# Searches for libtorch in these locations (first match wins):
#   1. $LIBTORCH environment variable
#   2. /opt/libtorch
#   3. <script_dir>/libtorch
#
# Override with the LIBTORCH environment variable:
#   LIBTORCH=/path/to/libtorch ./transcribe.sh --model-dir ... audio.wav
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ -n "${LIBTORCH:-}" ]]; then
    LT="$LIBTORCH"
elif [[ -d /opt/libtorch ]]; then
    LT=/opt/libtorch
elif [[ -d "$SCRIPT_DIR/libtorch" ]]; then
    LT="$SCRIPT_DIR/libtorch"
else
    echo "libtorch not found. Set the LIBTORCH env var or place libtorch at /opt/libtorch." >&2
    echo "" >&2
    echo "Download for Linux x86_64:" >&2
    echo "  curl -Lo libtorch.zip \\" >&2
    echo "    'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.7.0%2Bcpu.zip'" >&2
    echo "  unzip libtorch.zip -d /opt" >&2
    echo "" >&2
    echo "Download for Linux ARM64:" >&2
    echo "  curl -Lo libtorch.tar.gz \\" >&2
    echo "    'https://github.com/second-state/libtorch-releases/releases/download/v2.7.1/libtorch-cxx11-abi-aarch64-2.7.1.tar.gz'" >&2
    echo "  tar xzf libtorch.tar.gz -C /opt" >&2
    exit 1
fi

BINARY="$SCRIPT_DIR/target/release/transcribe"
if [[ ! -x "$BINARY" ]]; then
    echo "Binary not found. Build first:" >&2
    echo "  LIBTORCH=$LT RUSTFLAGS=\"-C link-arg=-Wl,-rpath-link,$LT/lib\" cargo build --release -j 1" >&2
    exit 1
fi

exec env LD_LIBRARY_PATH="$LT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$BINARY" "$@"
