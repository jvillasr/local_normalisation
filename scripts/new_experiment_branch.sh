#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
  echo "Usage: $0 <experiment_name>"
  echo "Example: $0 wing_anchor_renorm"
  exit 1
fi

NAME="$1"
DATE_UTC="$(date -u +%Y%m%d)"
BRANCH="exp/${DATE_UTC}_${NAME}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git switch "$BRANCH"
else
  git switch -c "$BRANCH"
fi

EXP_DIR="experiments/${DATE_UTC}_${NAME}"
mkdir -p "$EXP_DIR"

if [ ! -f "$EXP_DIR/README.md" ]; then
  cat > "$EXP_DIR/README.md" <<EOT
# ${DATE_UTC}_${NAME}

## Hypothesis
- Describe the method change being tested.

## Config
- Reference config file(s) in \`configs/\`.

## Metrics against baseline
- Balmer wing bias:
- Continuum kinks near edges:
- Number of failed fits:

## Artefacts
- Plots:
- Tables:
- Logs:
EOT
fi

echo "Switched to: $BRANCH"
echo "Experiment folder: $EXP_DIR"
