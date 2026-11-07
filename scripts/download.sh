#!/usr/bin/env bash
# Download CropHarvest (labels + monthly satellite features, ~170 MB) and country borders.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw
Z="https://zenodo.org/records/10251170/files"
[ -f data/raw/labels.geojson ] || curl -L -o data/raw/labels.geojson "$Z/labels.geojson?download=1"
if [ ! -d data/raw/features/arrays ]; then
  curl -L -o data/raw/features.tar.gz "$Z/features.tar.gz?download=1"
  tar -xzf data/raw/features.tar.gz -C data/raw && rm data/raw/features.tar.gz
fi
[ -f data/ne_50m_admin_0_countries.geojson ] || curl -L -o data/ne_50m_admin_0_countries.geojson \
  https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_50m_admin_0_countries.geojson
echo "done: data/raw has $(ls data/raw/features/arrays | wc -l) point files"
