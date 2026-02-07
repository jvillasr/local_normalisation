#!/usr/bin/env bash

python /nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/physparams_paper/scripts/compare_local_continuum_variants.py \
    --manifest /nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/local_normalised_data/p98_merge_comparison/manifest_first10.csv \
    --out-dir /nexus/posix0/MIA-astro-env/hxr/jvillasr/SDSS/local_normalised_data/p98_merge_comparison \
    --smooth-sigma 2.0 \
    --lines HDELTA,HEI4144,HEI4471 \
    --start-index 0 \
    --max-stars 10