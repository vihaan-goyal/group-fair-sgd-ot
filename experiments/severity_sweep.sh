#!/usr/bin/env bash
# Severity sweep behind Table 1(a,b), Fig. 2 and the repository severity figures: the grid-free rules, 5 seeds, 4000 steps,
# constant and decaying stepsize (eta_t = eta / (1 + t/1000)), one output folder per cell.
#   imdbwiki: sigma_i ~ i^beta,   beta in {0, .5, 1, 1.5, 2, 3}  (nu = 0.31 ... 0.88)
#             plus the aligned instance sigma_i ~ 101 - i        (--regimes feasible, nu = 0)
#   adult:    sigma_i ~ p_i^beta, beta in {0, .25, .5, .75, 1}  (nu = 0.60 ... 0)
# Usage (repo root):  bash experiments/severity_sweep.sh [parallel_jobs]     (PYTHON=... to override)
set -u
P=${PYTHON:-python}
ROOT=results/runs
JOBS=${1:-6}
COMMON="--models fedavot fedavg full --seeds 0 1 2 3 4 --sweep --gzip --user-log-every 10 --progress-every 0 --no-clobber"
mkdir -p "$ROOT/logs"
jobs=()
for decay in 1000 none; do
  d=""; tag="const"; [ "$decay" != none ] && { d="--lr-decay $decay"; tag="decay$decay"; }
  for b in 0.00 0.50 1.00 1.50 2.00 3.00; do
    jobs+=("$P src/run_experiments.py --datasets imdbwiki --regimes infeasible --r-power $b $d $COMMON --outdir $ROOT/imdb_beta${b}_$tag > $ROOT/logs/imdb_beta${b}_$tag.log 2>&1")
  done
  for b in 0.00 0.25 0.50 0.75 1.00; do
    jobs+=("$P src/run_experiments.py --datasets adult --regimes infeasible --r-power $b $d $COMMON --outdir $ROOT/adult_beta${b}_$tag > $ROOT/logs/adult_beta${b}_$tag.log 2>&1")
  done
  jobs+=("$P src/run_experiments.py --datasets imdbwiki --regimes feasible $d $COMMON --outdir $ROOT/imdb_feasible_$tag > $ROOT/logs/imdb_feasible_$tag.log 2>&1")
done
echo "${#jobs[@]} cells, $JOBS parallel"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
printf '%s\n' "${jobs[@]}" | xargs -P "$JOBS" -I{} bash -c '{}'
echo SWEEP_DONE
