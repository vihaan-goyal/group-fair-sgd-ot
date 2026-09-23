#!/usr/bin/env bash
# Imbalance baselines of Table 1 on the four headline cells, decaying stepsize:
# fixed multiplier m/K (fedavg_mk), upsampling (ipw) and downsampling on all four, LDS on the two IMDb-Wiki cells (regression only).
# Writes into the same cell folders as severity_sweep.sh (--no-clobber keeps existing runs).
# Usage (repo root):  bash experiments/baselines.sh
set -u
P=${PYTHON:-python}
ROOT=results/runs
C="--seeds 0 1 2 3 4 --sweep --gzip --user-log-every 10 --progress-every 0 --no-clobber --lr-decay 1000"
mkdir -p "$ROOT/logs"
$P src/run_experiments.py --datasets adult    --regimes infeasible --r-power 0 --models fedavg_mk ipw downsample     $C --outdir $ROOT/adult_beta0.00_decay1000 > $ROOT/logs/baselines_adult_beta0.00.log 2>&1
$P src/run_experiments.py --datasets adult    --regimes infeasible --r-power 1 --models fedavg_mk ipw downsample     $C --outdir $ROOT/adult_beta1.00_decay1000 > $ROOT/logs/baselines_adult_beta1.00.log 2>&1
$P src/run_experiments.py --datasets imdbwiki --regimes infeasible --r-power 0 --models fedavg_mk ipw downsample lds $C --outdir $ROOT/imdb_beta0.00_decay1000  > $ROOT/logs/baselines_imdb_beta0.00.log 2>&1
$P src/run_experiments.py --datasets imdbwiki --regimes feasible               --models fedavg_mk ipw downsample lds $C --outdir $ROOT/imdb_feasible_decay1000  > $ROOT/logs/baselines_imdb_feasible.log 2>&1
echo BASELINES_DONE
