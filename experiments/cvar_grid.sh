#!/usr/bin/env bash
# Mean-CVaR layer (Sec. 3, results in Sec. 4): full 10x10 (alpha, gamma) grid for fedavot_cvar and fedcvar on the four
# headline cells, decaying stepsize, 5 seeds. Writes into the severity-sweep cell folders.
# Usage (repo root):  bash experiments/cvar_grid.sh
set -u
P=${PYTHON:-python}
R=results/runs
C="--models fedavot_cvar fedcvar --seeds 0 1 2 3 4 --sweep --gzip --user-log-every 10 --progress-every 0 --no-clobber --lr-decay 1000"
mkdir -p "$R/logs"
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
$P src/run_experiments.py --datasets imdbwiki --regimes infeasible --r-power 0 $C --outdir $R/imdb_beta0.00_decay1000  > $R/logs/cvar_imdb_beta0.00.log 2>&1 &
$P src/run_experiments.py --datasets imdbwiki --regimes feasible               $C --outdir $R/imdb_feasible_decay1000  > $R/logs/cvar_imdb_feasible.log 2>&1 &
$P src/run_experiments.py --datasets adult    --regimes infeasible --r-power 0 $C --outdir $R/adult_beta0.00_decay1000 > $R/logs/cvar_adult_beta0.00.log 2>&1 &
$P src/run_experiments.py --datasets adult    --regimes infeasible --r-power 1 $C --outdir $R/adult_beta1.00_decay1000 > $R/logs/cvar_adult_beta1.00.log 2>&1 &
wait
echo CVAR_GRID_DONE
