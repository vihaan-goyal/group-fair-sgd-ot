# Group-Fair SGD via Masked Optimal Transport

Code and results for the experiments in

> H. Rahimi, A. Sodhi, V. Goyal, D. Kalogerias, "Group-Fair SGD via Masked Optimal Transport," submitted to ICASSP 2027.

**GAVOT** trains a model on a target mixture of groups $F_p(\theta)=\sum_i p_i f_i(\theta)$ when each SGD step
only sees a random batch of $K$ groups drawn by a sampler we do not control. It reweights the drawn groups with
the weights of a masked optimal-transport plan (fit by IPFP / Sinkhorn) between the target $p$ and the batch
distribution. When the target is reachable (every $p_i \le r_i$, $r_i$ = inclusion probability) this recovers
$F_p$ exactly. When it is not, transport converges to the closest reachable surrogate $\hat p$. The *infeasible mass*
$\nu=\sum_{p_i>r_i}p_i$ measures how far the target is out of reach.

Everything in Sec. 4 of the paper (Table 1, Fig. 2, every quoted number and $t$ statistic) comes from this
repository, which also holds what the paper had no room for: the per-skew severity curves, the constant-stepsize
runs and the full mean–CVaR $(\alpha,\gamma)$ grid.

## Quick start: tables and figures without training

The per-run summaries of every experiment are committed in `results/runs/`, and the tables behind the figures are
in `results/tables/`. So the paper's numbers and plots regenerate in seconds:

```bash
pip install -r requirements.txt
python src/paper_tables.py      # Table 1(a) and 1(b), all Welch t, the mean-CVaR grid results
python src/plot_rare_group.py   # figures/rare_group_fit.pdf (Fig. 2)
python src/plot_severity_col.py # figures/severity_col.pdf (severity sweeps, repository only)
```

## Where each result comes from

| Paper | What | Produced by | Reads |
|---|---|---|---|
| Table 1(a) | overall $F_p$ on the four headline instances, all rules | `src/paper_tables.py` | `results/runs/*/summary.csv` |
| Table 1(b) | IMDb-Wiki sweep: $\nu$, alignment gain $\Delta_{\mathrm{floor}}$, measured gain over group-blind averaging and its Welch $t$ | `src/paper_tables.py` | same, plus `results/tables/severity_sweep_table.csv` |
| Sec. 4 text | $t$ statistics, mean-CVaR grid optimum, worst-race losses | `src/paper_tables.py` | same |
| severity curves (repo only) | both severity sweeps, four stacked panels: IMDb-Wiki vs $\nu$, Adult vs $\beta$, constant and decaying stepsize, with floors $\Phi$ | `src/plot_severity_col.py` | `results/tables/severity_sweep_table.csv` |
| (same, split) | the same sweeps as two full-width figures | `src/plot_figures.py` | same |
| Fig. 2 | loss on the least represented group (Adult race Other, IMDb-Wiki top tier) | `src/plot_rare_group.py` | `results/tables/rare_group_table.csv` |
| floors $\Phi$, $\nu$, $\hat p$ | exact loss floor each rule converges toward | `src/build_tables.py` + `src/transport_floors.py` | raw runs + data |
| IPFP cap | 1000 IPFP sweeps already give the limiting $\hat p$ | `src/ipfp_limit_check.py` | data |

## Setup of the experiments

- **Groups.** $m=100$ groups. **IMDb-Wiki**: age regression (MSE, linear model on 128-d face embeddings), one group
  per identity, target $p_i \propto (101-i)^3$. **Adult**: income classification (cross-entropy, logistic regression),
  groups split by race with group counts proportional to prevalence, target uniform over the 5 races.
- **Sampler.** Each step draws $K=3$ groups without replacement with weights $\sigma_i$ (`--r-power BETA`). $r_i$ is the
  resulting inclusion probability.
  - IMDb-Wiki: $\sigma_i \propto i^\beta$, $\beta\in\{0,\tfrac12,1,\tfrac32,2,3\}$. This favors the least important
    identities, so $\nu$ grows from 0.31 to 0.88. There is also an aligned instance $\sigma_i\propto 101-i$ (`--regimes feasible`, $\nu=0$).
  - Adult: $\sigma_i \propto p_i^\beta$, $\beta\in\{0,\tfrac14,\tfrac12,\tfrac34,1\}$, from uniform sampling
    (prevalence availability, $\nu=0.60$) to aligned ($\nu=0$).
- **Step.** Each drawn group takes $H=5$ full-batch gradient steps on its data from the current iterate. The
  displacements are combined with the rule's weights. All rules share the same sequence of draws.
- **Stepsize.** Constant, or decaying $\eta_t=\eta/(1+t/1000)$ (`--lr-decay 1000`, the paper's headline setting).
- **Protocol.** 4000 steps and 5 seeds. We report the mean training loss $F_p$ over the last 500 iterates.
  Error bars are $\pm 1$ std over seeds, and Welch $t$ uses the sample std ($n-1$).
- **Floors.** $\Phi(w)=F_p(\theta^\star_w)$ is the target loss at the minimizer of the $w$-weighted objective. It is
  computed exactly (closed-form least squares / Newton) for $w=\hat p$ (GAVOT), $w\propto r$ (group-blind) and $w=p$ (full).

## Rule names in the code

The code grew out of a federated-learning codebase, so rule names in the code and in every CSV differ from the paper:

| code / CSV | paper |
|---|---|
| `fedavot` | GAVOT (transport weights) |
| `fedavg` | group-blind averaging (uniform $1/K$ over the drawn groups) |
| `fedavg_mk` | fixed multiplier $(m/K)\,p_i$ |
| `ipw` | upsampling (self-normalized $p_i/r_i$) |
| `downsample` | downsampling ($\min(p_i/r_i,1)$, renormalized) |
| `lds` | LDS, label distribution smoothing (Yang et al., ICML 2021) |
| `fedavot_cvar` / `fedcvar` | mean–CVaR layer on GAVOT / on group-blind averaging, $(\alpha,\gamma)$ grid (see the note below) |
| `full` | full access (every group every step, weighted by $p$) |

**CVaR convention.** The paper uses the FedeRage convention, $F^{\alpha,\gamma}_w=(1-\gamma)F_w+\gamma\,\mathrm{CVaR}^w_\alpha$,
so $\gamma=0$ is risk-neutral. The code and every CSV use the opposite weighting: the column `gamma` multiplies the mean,
and `gamma=1` is risk-neutral. So **paper $\gamma$ = 1 − code `gamma`** (same $\alpha$). For example, the paper's
$(\alpha,\gamma)=(0.2,0.3)$ is the code's `alpha=0.2, gamma=0.7`.

Cell folders are named `<dataset>_beta<β>_<stepsize>`, e.g. `imdb_beta0.00_decay1000`. `imdb_feasible_*` is the aligned
IMDb-Wiki instance.

## Full reproduction from scratch

```bash
bash experiments/severity_sweep.sh 6   # 24 cells x 5 seeds (GAVOT, group-blind, full), 6 in parallel
bash experiments/baselines.sh          # m/K, upsampling, downsampling, LDS on the four headline cells
bash experiments/cvar_grid.sh          # 10x10 (alpha, gamma) mean-CVaR grid on the four headline cells
python src/build_tables.py             # results/tables/*.csv (floors, nu, rare-group losses)
python src/paper_tables.py
python src/plot_severity_col.py
python src/plot_rare_group.py
```

Everything runs on a CPU. One cell (5 seeds, 4000 steps, GAVOT / group-blind / full) takes about 3 minutes on Adult
and 6 minutes on IMDb-Wiki on a laptop. `src/run_experiments.py --help` lists all options; `--smoke` gives a one-minute IMDb-Wiki check and
`--selftest` checks the vectorized engine. Raw per-step CSVs (`*.csv.gz`, about 1.2 GB for the full sweep) and
cached transport plans are git-ignored. Only `summary.csv` per cell is committed. Set `PYTHON=...` to choose the
interpreter used by the shell scripts.

## Data

See [`data/README.md`](data/README.md). Both datasets are included; the IMDb-Wiki files fall under IMDb-Wiki's
non-commercial research terms.

## Layout

```
src/run_experiments.py     training runner (all rules, vectorized over configs)
src/transport_floors.py    p_hat, p_tilde, nu and exact floors from a cached transport plan
src/build_tables.py        raw runs -> results/tables/*.csv
src/paper_tables.py        Table 1(a,b), Welch t, CVaR grid results
src/plot_rare_group.py     Fig. 2
src/plot_severity_col.py   severity sweeps, one column (repository figure)
src/plot_severity_1x4.py   severity sweeps, 1x4 row
src/plot_severity_2x2.py   severity sweeps, 2x2 grid
src/plot_figures.py        severity sweeps, two full-width figures
src/ipfp_limit_check.py    IPFP cap check
experiments/*.sh           the runs behind the paper
results/runs/<cell>/       summary.csv per cell (tail-500 statistics per rule and seed)
results/tables/            severity_sweep_table.csv, rare_group_table.csv
figures/                   the paper figures
```

## License

Code: MIT (see `LICENSE`). Adult data: UCI Machine Learning Repository, CC BY 4.0.
