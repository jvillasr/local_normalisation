#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper"
FIG_DIR="/nexus/posix0/MIA-astro-env/hxr/shared/FEROS_data/BOSS_data/boss_v6_2_1_local_norm/figures"
LINES_FILE="${REPO_ROOT}/scripts/line_lists/fastwind_full_26.txt"
LOG="${FIG_DIR}/fwhm_mask_test_20260205.log"
UV_CACHE_DIR="${REPO_ROOT}/.uv-cache"

mkdir -p "${FIG_DIR}"
mkdir -p "${UV_CACHE_DIR}"

shopt -s nullglob
files=("${FIG_DIR}"/*_balmer_local_vs_full_bluepad20_hdelta10.png)
if [ ${#files[@]} -eq 0 ]; then
  echo "No bluepad20_hdelta10 plots found in ${FIG_DIR}" > "${LOG}"
  exit 1
fi

{
  echo "Start $(date -Is)"
  echo "Using ${#files[@]} pilot spectra"
} > "${LOG}"

for png in "${files[@]}"; do
  base=$(basename "${png}")
  spec_name=${base%_balmer_local_vs_full_bluepad20_hdelta10.png}
  field=$(echo "${spec_name}" | cut -d'-' -f2)
  for k in 1.0 1.5 2.0; do
    for stage in fit p98; do
      echo "[$(date -Is)] ${spec_name} k=${k} stage=${stage}" >> "${LOG}"
      UV_CACHE_DIR="${UV_CACHE_DIR}" uv run python "${REPO_ROOT}/scripts/plot_balmer_local_vs_full.py" \
        --merge-windows \
        --lines-file "${LINES_FILE}" \
        --field "${field}" \
        --spec-file "${spec_name}.fits" \
        --balmer-blue-pad 20 \
        --hdelta-blue-pad 10 \
        --balmer-mask-k "${k}" \
        --balmer-mask-stage "${stage}" \
        --balmer-mask-smooth 2.0 \
        --lorentzian-plots >> "${LOG}" 2>&1
    done
  done
done

echo "Done $(date -Is)" >> "${LOG}"
