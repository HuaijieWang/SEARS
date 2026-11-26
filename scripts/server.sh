#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
run="${1:?Usage: bash scripts/server.sh RUN train|eval|run|check}"
operation="${2:-run}"
case "$operation" in
    train|run)
        python -m sears train "$run"
        if [[ "$operation" == run ]]; then
            python -m sears eval "$run"
        fi
        ;;
    eval|check)
        python -m sears "$operation" "$run"
        ;;
    *)
        exit 2
        ;;
esac
