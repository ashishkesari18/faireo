#!/usr/bin/env bash
# Reproduce every number in the README. About 2-3 hours on a laptop CPU (no GPU needed).
set -euo pipefail
cd "$(dirname "$0")"
bash scripts/download.sh
python scripts/prepare.py
python scripts/quality.py
python scripts/embed_presto.py
python scripts/embed_prithvi.py
for m in lgbm rf presto_lr presto_lgbm presto_noloc_lr prithvi_lr prithvi_lgbm; do
  for s in country random; do python scripts/audit.py --model $m --split $s; done
done
for b in 0.5 1; do
  python scripts/audit.py --model lgbm --split country --balance $b
  python scripts/audit.py --model presto_lgbm --split country --balance $b
done
python scripts/report.py
python scripts/label_control.py --run lgbm_country_s0
python scripts/label_control.py --run presto_lgbm_country_s0
python scripts/sample_worldcover.py
python scripts/worldcover_audit.py
python scripts/novelty.py
python scripts/explain.py --run lgbm_country_s0
python scripts/trust.py --run lgbm_country_s0
python scripts/trust.py --run presto_lgbm_country_s0
python scripts/recalibrate.py --run lgbm_country_s0
python scripts/recalibrate.py --run presto_lgbm_country_s0
python scripts/label_budget.py
python scripts/figures.py
python scripts/build_trustmap.py
