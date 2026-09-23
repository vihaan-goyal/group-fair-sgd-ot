# Experiment runner for "Group-Fair SGD via Masked Optimal Transport" (GAVOT).
#
# Trains a linear model on m = 100 groups (IMDb-Wiki: age regression on 128-d face
# embeddings, one group per identity; Adult: logistic regression, groups split by race).
# Each step a sampler draws K = 3 groups without replacement with weights sigma_i (knob
# --r-power BETA); every drawn group takes H local full-batch gradient steps from the current
# iterate and the displacements are combined by the rule's weights. All rules are trained
# from the SAME sequence of draws, so their curves are directly comparable. Writes per-step
# CSVs (overall / per-critical-group / per-group train loss) and <outdir>/summary.csv
# (tail-500 statistics per run).
#
# Rule names in the code (and in every CSV) vs the paper:
#   fedavot      GAVOT: masked-OT (IPFP) transport weights over the drawn batch
#   fedavg       group-blind averaging: uniform 1/K over the drawn batch
#   fedavg_mk    fixed multiplier (m/K) p_i (unnormalized, not convex)
#   ipw          upsampling, as self-normalized p_i / pi_i reweighting
#   downsample   w_i ~ min(p_i / pi_i, 1), renormalized
#   lds          label distribution smoothing (Yang et al., ICML 2021), IMDb-Wiki only
#   fedavot_cvar mean-CVaR layer on top of GAVOT, swept over the (alpha, gamma) grid
#   fedcvar      the same CVaR layer on group-blind averaging
#   full         full access: every group every step, weighted by p (reference)
#
# Engine: every subset-trained model is a ROW of a config table pushed through ONE
# vectorized local-step function. GAVOT and group-blind averaging are the CVaR step at
# (alpha=1, gamma=1), where the gradient multiplier is exactly 1.0, so gamma=1.0 grid rows
# reproduce the grid-free models up to <=1e-15 relative (batched-BLAS summation order);
# --selftest enforces 1e-12. A diverged config's weights are zeroed and its logged losses
# are pinned at --cap.
#
# Filenames carry (dataset, regime, model, alpha, gamma, seed) but NOT rounds, --r-power or
# --lr-decay: use one --outdir per cell (the scripts in experiments/ do).

import argparse
import os
import re
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd

GRID_MODELS = ("fedavot_cvar", "fedcvar")     # swept over the (alpha, gamma) grid
FREE_MODELS = ("fedavot", "fedavg", "fedavg_mk", "ipw", "downsample", "lds")   # single config each (no alpha/gamma)
ALL_MODELS = GRID_MODELS + FREE_MODELS + ("full",)
UNIF_AGG = {"fedcvar", "fedavg"}              # rows aggregated uniformly (1/K)
MK_AGG = {"fedavg_mk"}                        # the paper's fixed multiplier: (N/K) * p_i (not convex)
# Imbalance baselines. Both are convex (self-normalized over the K observed
# users), so they cannot diverge the way fedavg_mk does:
IPW_AGG = {"ipw"}                             # upsampling as reweighting: w_i ~ p_i / pi_i, renormalized
DOWN_AGG = {"downsample"}                     # downsampling: w_i ~ min(p_i / pi_i, 1), renormalized
                                              #   (over-observed groups are down-weighted, starved ones untouched)
LDS_AGG = {"lds"}                             # imbalanced regression (Yang et al., ICML 2021, dir.csail.mit.edu):
                                              #   label distribution smoothing, group weight = mean over its samples of
                                              #   1 / (Gaussian-smoothed label density), renormalized over the batch.
                                              #   Regression only (IMDb-Wiki ages); FDS needs deep features, not applicable.


def lds_group_weights(y_all, kernel_size=5, sigma=2.0):
    """LDS (Yang et al. 2021) at group level: bin the labels at unit width, smooth the histogram with a
    Gaussian kernel (their defaults ks=5, sigma=2), weight every sample by the inverse smoothed density
    (their 'INV' weighting, normalized to mean 1), and average within each group -> (N,) weights."""
    y = y_all.ravel()
    lo = np.floor(y.min()); bins = (np.floor(y) - lo).astype(int)
    counts = np.bincount(bins, minlength=int(np.floor(y.max()) - lo) + 1).astype(float)
    half = kernel_size // 2
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2); k /= k.sum()
    dens = np.convolve(counts, k, mode="same")
    w = 1.0 / np.maximum(dens[bins], 1e-12)
    w /= w.mean()
    return w.reshape(y_all.shape).mean(axis=1)

DATASET_LR = {"imdbwiki": 0.01, "adult": 0.1}
DATASET_ETA_T = {"imdbwiki": 0.05, "adult": 0.005}   # adult = LR/20 heuristic (untested
                                                      # territory: CVaR never ran on Adult)

FNAME_RE = re.compile(
    r"^(?P<dataset>adult|imdbwiki)_(?P<regime>feasible|infeasible)"
    r"_(?P<model>fedavot_cvar|fedcvar|fedavot|fedavg|full)"
    r"(?:_a(?P<alpha>\d+\.\d{2})_g(?P<gamma>\d+\.\d{2}))?"
    r"_seed(?P<seed>\d+)\.csv(?:\.gz)?$")

SUMMARY_KEY = ["dataset", "regime", "model", "alpha", "gamma", "seed"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Run FedAVOT/CVaR aggregation experiments and write per-round CSVs "
                    "(overall loss, loss per critical group, loss per user).",
        epilog="fedavg = plain local training + uniform average over the K drawn users "
               "(the m/K-scaled FedAvg(K) is intentionally absent). Multiple concurrent "
               "runner processes should use separate --outdir (summary.csv merging is "
               "last-writer-wins); reconcile with --rebuild-summary.")
    ap.add_argument("--datasets", nargs="+", default=["imdbwiki", "adult"],
                    choices=["imdbwiki", "adult"], help="datasets to run")
    ap.add_argument("--regimes", nargs="+", default=["infeasible", "feasible"],
                    choices=["infeasible", "feasible"],
                    help="availability regime (adult: infeasible=uniform r / "
                         "feasible=aligned r=p; imdbwiki: mirrored cubic / aligned linear)")
    ap.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=ALL_MODELS,
                    help="models to run; alpha/gamma apply only to fedavot_cvar and fedcvar")
    ap.add_argument("--alphas", nargs="+", type=float, default=None,
                    help="CVaR alpha grid (default: 0.1 0.2 ... 1.0)")
    ap.add_argument("--gammas", nargs="+", type=float, default=None,
                    help="CVaR gamma grid (default: 0.1 0.2 ... 1.0); gamma=1.0 rows "
                         "reproduce fedavot/fedavg exactly")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="RandomState seeds (drive only the subset-draw sequence)")
    ap.add_argument("--rounds", type=int, default=4000, help="communication rounds")
    ap.add_argument("--sweep", action="store_true",
                    help="vectorize all configs of each (dataset, regime, seed) into one "
                         "training loop (STRONGLY recommended for grids)")
    ap.add_argument("--local-epochs", type=int, default=5, help="local full-batch GD steps")
    ap.add_argument("--k", type=int, default=3, help="users drawn per round")
    ap.add_argument("--num-users", type=int, default=100, help="population size")
    ap.add_argument("--samples-per-user", type=int, default=30, help="samples per user")
    ap.add_argument("--r-power", type=float, default=None,
                    help="override the regime's availability to sweep infeasibility severity. "
                         "imdbwiki: r_i ~ i**BETA (3 = mirrored cubic = the infeasible anchor, "
                         "0 = uniform; importance p stays reversed-cubic). adult: r_i ~ p_i**BETA "
                         "(0 = uniform = the infeasible anchor, 1 = aligned = the feasible anchor). "
                         "Requires a non-default --outdir; filenames do not encode it.")
    ap.add_argument("--lr-decay", type=float, default=None,
                    help="learning-rate decay horizon T0: lr_t = lr / (1 + t / T0). "
                         "Default None = constant lr (the anchors). Requires non-default --outdir.")
    ap.add_argument("--lr-schedule", default="hyper", choices=["hyper", "sqrt", "step", "cosine"],
                    help="shape of the decay (all rules share it): hyper = lr/(1+t/T0) [default]; "
                         "sqrt = lr/sqrt(1+t/T0); step = lr*0.5**floor(t/T0); cosine = lr*(1+cos(pi t/R))/2 "
                         "(T0 ignored). T0 from --lr-decay. Non-default requires non-default --outdir.")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default per dataset: imdbwiki 0.01, adult 0.1)")
    ap.add_argument("--eta-t", type=float, default=None,
                    help="CVaR threshold step (default: imdbwiki 0.05, adult 0.005=LR/20)")
    ap.add_argument("--t0", type=float, default=0.0, help="initial CVaR threshold")
    ap.add_argument("--cap", type=float, default=1e12,
                    help="divergence cap; dead configs log this sentinel forever")
    ap.add_argument("--tail", type=int, default=500, help="rounds averaged for summary stats")
    ap.add_argument("--q-samples", type=int, default=1_000_000, help="MC samples for q")
    ap.add_argument("--pi-samples", type=int, default=500_000,
                    help="MC samples for inclusion probabilities")
    ap.add_argument("--ipfp-tol", type=float, default=1e-12, help="IPFP row-error tolerance")
    ap.add_argument("--ipfp-iters", type=int, default=1000, help="IPFP max iterations")
    ap.add_argument("--imdb-groups", default="tiers", choices=["tiers", "decades", "feasibility"],
                    help="IMDb-Wiki critical-group definition (tiers = importance quintiles)")
    ap.add_argument("--outdir", default="results/runs/default",
                    help="output directory (git-ignored); use a new dated folder per run")
    ap.add_argument("--user-log-every", type=int, default=1,
                    help="log per-user columns every N rounds (overall/groups always "
                         "every round; the final round is always logged)")
    ap.add_argument("--gzip", action="store_true", help="write .csv.gz")
    ap.add_argument("--no-clobber", action="store_true",
                    help="skip configs whose CSV already exists (cheap sweep resume)")
    ap.add_argument("--transport-cache", action=argparse.BooleanOptionalAction, default=True,
                    help="cache q/Wcols/pi per (dataset, regime) in the output dir")
    ap.add_argument("--rebuild-summary", action="store_true",
                    help="regenerate summary.csv from the CSVs on disk, then exit")
    ap.add_argument("--selftest", action="store_true",
                    help="run engine identity checks (small N, fast), then exit")
    ap.add_argument("--smoke", action="store_true",
                    help="1-unit shape check (imdbwiki/infeasible, seed 0, 50 rounds) "
                         "into <outdir>/smoke; prints full-sweep time/disk projections")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned units, file counts and estimates; run nothing")
    ap.add_argument("--progress-every", type=int, default=1000,
                    help="round-progress print interval")
    args = ap.parse_args(argv)
    if args.alphas is None:
        args.alphas = [round(float(v), 2) for v in np.linspace(0.1, 1.0, 10)]
    if args.gammas is None:
        args.gammas = [round(float(v), 2) for v in np.linspace(0.1, 1.0, 10)]
    return args


# ================================================================
# Transport machinery (transcribed from the anchor scripts)
# ================================================================
def all_K_subsets_1based(N, K):
    return list(combinations(range(1, N + 1), K))


def build_mask(n, subsets):
    M = np.zeros((n, len(subsets)), dtype=bool)
    for j, s in enumerate(subsets):
        for i in s:
            M[i - 1, j] = True
    return M


def ipfp_masked(p, q, M, tol, max_iter):
    Y = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        rows = np.where(M[:, j])[0]
        if rows.size:
            Y[rows, j] = q[j] / rows.size
    row_err = np.inf
    for _ in range(max_iter):
        Y *= (p / np.maximum(Y.sum(axis=1), 1e-12))[:, None]
        Y *= (q / np.maximum(Y.sum(axis=0), 1e-12))[None, :]
        row_err = np.max(np.abs(Y.sum(axis=1) - p))
        if row_err < tol:
            break
    return Y, row_err


def solve_T(p, q, subsets, tol, max_iter):
    M = build_mask(len(p), subsets)
    Y, row_err = ipfp_masked(p, q, M, tol, max_iter)
    print(f"  IPFP row_err = {row_err:.2e}")
    T = np.zeros_like(Y)
    pos = q > 0
    T[:, pos] = Y[:, pos] / q[pos]
    T[~M] = 0
    s = T.sum(axis=0, keepdims=True)
    s[s == 0] = 1.0
    T /= s
    return T, row_err


def column_users_and_weights(T, subsets, j):
    # verbatim ancestor lookup (imdbwiki_cvar_feasible.py) -- kept for --selftest
    rows = np.array([i - 1 for i in subsets[j]])
    w = T[rows, j]
    ss = w.sum()
    return rows, (w / ss if ss > 0 else np.full(len(rows), 1.0 / len(rows)))


def compact_col_weights(T, subs0):
    # per-column normalized weights (M, K); semantics of column_users_and_weights
    # including the uniform fallback for zero-sum columns. Lets the 129 MB T be freed.
    M, K = subs0.shape
    w = T[subs0, np.arange(M)[:, None]]
    s = w.sum(axis=1, keepdims=True)
    return np.where(s > 0, w / np.where(s > 0, s, 1.0), np.full((1, K), 1.0 / K))


def estimate_q(subsets, r, N, K, samples, rng):
    # fast variant (adult_fairness.py): 200k batches + np.unique pre-aggregation;
    # stream-identical to the 50k-batch IMDb version (gumbel is elementwise on the
    # uniform stream, so chunk boundaries do not change the draws)
    lookup = {s: i for i, s in enumerate(subsets)}
    counts = np.zeros(len(subsets))
    logr = np.log(r)
    done = 0
    while done < samples:
        b = min(200_000, samples - done)
        g = rng.gumbel(size=(b, N)) + logr[None, :]
        tk = np.argpartition(-g, K, axis=1)[:, :K]
        tk.sort(axis=1)
        uniq, cnt = np.unique(tk + 1, axis=0, return_counts=True)
        for row, c in zip(uniq, cnt):
            counts[lookup[tuple(row.tolist())]] += c
        done += b
    return counts / counts.sum()


def inclusion_probs(r, N, K, samples, rng, batch=50_000):
    # batched (caps the ancestors' ~400 MB transient at ~40 MB; identical stream)
    incl = np.zeros(N)
    logr = np.log(r)
    done = 0
    while done < samples:
        b = min(batch, samples - done)
        g = rng.gumbel(size=(b, N)) + logr[None, :]
        tk = np.argpartition(-g, K, axis=1)[:, :K]
        np.add.at(incl, tk.ravel(), 1)
        done += b
    return incl / samples


def get_transport(dataset, regime, p, r, args, verify=False):
    N, K = args.num_users, args.k
    subsets = all_K_subsets_1based(N, K)
    subs0 = np.array(subsets, dtype=np.int64) - 1
    cache_path = os.path.join(args.outdir, f"transport_{dataset}_{regime}_K{K}.npz")
    if args.transport_cache and not verify and os.path.exists(cache_path):
        z = np.load(cache_path)
        if (int(z["N"]) == N and int(z["K"]) == K
                and int(z["q_samples"]) == args.q_samples
                and int(z["pi_samples"]) == args.pi_samples
                and float(z["ipfp_tol"]) == args.ipfp_tol
                and int(z["ipfp_iters"]) == args.ipfp_iters
                and np.array_equal(z["p"], p) and np.array_equal(z["r"], r)):
            print(f"  transport cache hit: {cache_path} (row_err {float(z['row_err']):.2e})")
            q = z["q"]
            return dict(q=q, q_cum=np.cumsum(q), Wcols=z["Wcols"], pi=z["pi"],
                        subs0=subs0, subsets=subsets, row_err=float(z["row_err"]))
        print("  transport cache stale (params changed), recomputing")

    t0 = time.time()
    pi = inclusion_probs(r, N, K, args.pi_samples, np.random.RandomState(0))
    infeas = p > pi
    print(f"  infeasible users: {infeas.sum()}/{N}, p-mass {p[infeas].sum() * 100:.1f}%, "
          f"max p_i/pi_i = {np.max(p / np.maximum(pi, 1e-12)):.2f}")
    q = estimate_q(subsets, r, N, K, args.q_samples, np.random.RandomState(0))
    q = (q + 1e-12) / q.sum()   # ancestor smoothing: divisor is the PRE-smoothing sum
    print(f"  estimate_q done in {time.time() - t0:.0f}s")
    T, row_err = solve_T(p, q, subsets, args.ipfp_tol, args.ipfp_iters)
    Wcols = compact_col_weights(T, subs0)
    if verify:   # --selftest check (iv): compaction == verbatim per-column lookup
        rngv = np.random.RandomState(123)
        for j in rngv.choice(len(subsets), size=min(100, len(subsets)), replace=False):
            rows_ref, w_ref = column_users_and_weights(T, subsets, j)
            if not (np.array_equal(rows_ref, subs0[j])
                    and np.max(np.abs(w_ref - Wcols[j])) < 1e-15):
                raise SystemExit(f"SELFTEST FAIL: Wcols mismatch at column {j}")
        print("  [selftest] Wcols == column_users_and_weights on 100 random columns")
    del T
    if args.transport_cache:
        os.makedirs(args.outdir, exist_ok=True)
        np.savez_compressed(cache_path, q=q, Wcols=Wcols, pi=pi, p=p, r=r,
                            N=N, K=K, q_samples=args.q_samples, pi_samples=args.pi_samples,
                            ipfp_tol=args.ipfp_tol, ipfp_iters=args.ipfp_iters,
                            row_err=row_err)
        print(f"  transport cached -> {cache_path}")
    return dict(q=q, q_cum=np.cumsum(q), Wcols=Wcols, pi=pi,
                subs0=subs0, subsets=subsets, row_err=row_err)


# ================================================================
# Data pipelines (transcribed; deterministic, seed-independent)
# ================================================================
def make_imdb_distributions(N, regime):
    idx = np.arange(1, N + 1)
    p = idx[::-1] ** 3
    if regime == "infeasible":
        r = (idx ** 3).astype(float)          # mirrored cubic
    else:
        r = idx[::-1].astype(float)           # aligned linear
    return p / p.sum(), r / r.sum()


def load_imdbwiki(args):
    N, S = args.num_users, args.samples_per_user
    df = pd.read_csv("data/imdb_wiki.csv")
    df = df[df["split"] == "train"]
    df["client_id"] = df["path"].str.extract(r"(nm\d+)")
    groups = [g for _, g in df.groupby("client_id") if len(g) >= S]
    assert len(groups) >= N, f"only {len(groups)} clients with >= {S} samples"
    groups = groups[:N]
    EMB = np.load("data/imdb_embeddings.npy", allow_pickle=True).item()
    DIM = next(iter(EMB.values())).shape[0]
    X_full = np.concatenate([np.stack([EMB[pth] for pth in g["path"].values[:S]])
                             for g in groups])
    y_raw = np.concatenate([g["age"].values[:S] for g in groups]).astype(float)
    y_full = y_raw - y_raw.mean()
    X_all = X_full.reshape(N, S, DIM)
    y_all = y_full.reshape(N, S)
    user_mean_age = y_raw.reshape(N, S).mean(axis=1)   # pre-centering, for --imdb-groups decades
    return dict(X_all=X_all, y_all=y_all, X2=X_all.reshape(N * S, DIM),
                D=DIM, N=N, S=S, task="mse", user_mean_age=user_mean_age)


def load_adult(args):
    N, S = args.num_users, args.samples_per_user
    df = pd.read_csv("data/adult.csv")
    y_bin = (df["class"].str.strip() == ">50K").astype(float).values
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
                "hours-per-week"]
    cat_cols = ["workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "native-country"]
    Xnum = df[num_cols].astype(float).values
    Xnum = (Xnum - Xnum.mean(axis=0)) / np.maximum(Xnum.std(axis=0), 1e-9)
    Xcat = pd.get_dummies(df[cat_cols].fillna("Unknown"), dtype=float).values
    X_feat = np.concatenate([Xnum, Xcat, np.ones((len(df), 1))], axis=1)  # +intercept
    DIM = X_feat.shape[1]

    gser = df["race"].str.strip()
    GROUP_NAMES = sorted(gser.unique(), key=lambda g: -(gser == g).sum())
    NG = len(GROUP_NAMES)
    counts = np.array([(gser == g).sum() for g in GROUP_NAMES], dtype=float)
    users_per_group = np.maximum(1, np.round(counts / counts.sum() * N)).astype(int)
    users_per_group[0] += N - users_per_group.sum()
    assert users_per_group.sum() == N

    rng_part = np.random.RandomState(42)   # fixed partition, independent of --seeds
    X_users, y_users, group_of_user = [], [], []
    for gi, g in enumerate(GROUP_NAMES):
        idx = np.where(gser.values == g)[0]
        rng_part.shuffle(idx)
        need = users_per_group[gi] * S
        assert len(idx) >= need, f"group {g} too small: {len(idx)} < {need}"
        take = idx[:need].reshape(users_per_group[gi], S)
        for rows in take:
            X_users.append(X_feat[rows]); y_users.append(y_bin[rows]); group_of_user.append(gi)
    X_all = np.stack(X_users)
    y_all = np.stack(y_users)
    group_of_user = np.array(group_of_user)
    group_slices = [np.where(group_of_user == gi)[0] for gi in range(NG)]

    p = np.zeros(N)
    for gi in range(NG):
        p[group_slices[gi]] = (1.0 / NG) / users_per_group[gi]
    assert abs(p.sum() - 1) < 1e-12
    return dict(X_all=X_all, y_all=y_all, X2=X_all.reshape(N * S, DIM),
                D=DIM, N=N, S=S, task="ce", p=p,
                group_slices=group_slices, group_names=GROUP_NAMES)


def dataset_distributions(dataset, regime, data, args):
    if dataset == "imdbwiki":
        p, r = make_imdb_distributions(args.num_users, regime)
    else:
        p = data["p"]
        r = np.ones(args.num_users) / args.num_users if regime == "infeasible" else p.copy()
    if args.r_power is not None:
        if dataset == "imdbwiki":
            r = np.arange(1, args.num_users + 1, dtype=float) ** args.r_power   # 3 == mirrored cubic
            what = "r ~ i^beta"
        else:
            r = p ** args.r_power                                              # 0 == uniform, 1 == aligned
            what = "r ~ p^beta"
        r = r / r.sum()
        print(f"  --r-power {args.r_power}: availability {what} overrides regime '{regime}'")
    return p, r


def build_groups(dataset, data, p, pi, args):
    # returns an ordered {name: index array}; CSV columns derive from it
    if dataset == "adult":
        return {name: data["group_slices"][gi] for gi, name in enumerate(data["group_names"])}
    if args.imdb_groups == "tiers":
        order = np.argsort(-p, kind="stable")
        chunks = np.array_split(order, 5)
        return {f"tier{i + 1}": np.sort(c) for i, c in enumerate(chunks)}
    if args.imdb_groups == "decades":
        age = data["user_mean_age"]
        bins = np.digitize(age, [30, 40, 50, 60])
        labels = ["20s", "30s", "40s", "50s", "60s"]
        return {labels[b]: np.where(bins == b)[0] for b in range(5)
                if (bins == b).sum() > 0}
    infeas = p > pi
    out = {}
    if infeas.sum():
        out["infeasible"] = np.where(infeas)[0]
    if (~infeas).sum():
        out["feasible"] = np.where(~infeas)[0]
    return out


# ================================================================
# Engine
# ================================================================
def grid_local_step(Xu, yu, W0, t0, A, G, epochs, lr, eta_t, task):
    # vectorized generalization of batched_local_train_cvar (imdbwiki_cvar_grid.py)
    # over a config axis C. At (alpha, gamma) = (1, 1) the multiplier coef == 1.0
    # exactly, making this equal to plain batched_local_train up to batched-BLAS
    # summation-order effects (<=1e-15 relative, row-position dependent).
    m = Xu.shape[0]
    S = yu.shape[1]
    W = np.repeat(W0[:, None, :], m, axis=1)          # (C, m, D)
    Tv = np.repeat(np.asarray(t0, dtype=float)[:, None], m, axis=1)   # (C, m)
    amp = (1.0 - G) / A                                # (C,)
    for _ in range(epochs):
        z = np.einsum('msd,cmd->cms', Xu, W, optimize=True)
        if task == "mse":
            resid = z - yu[None, :, :]
            f = (resid ** 2).mean(axis=2)
            gsrc = resid
        else:
            f = (np.logaddexp(0.0, z) - yu[None, :, :] * z).mean(axis=2)
            gsrc = 1.0 / (1.0 + np.exp(-z)) - yu[None, :, :]
        active = (f > Tv).astype(float)
        coef = amp[:, None] * active + G[:, None]
        grad = np.einsum('msd,cms->cmd', Xu, gsrc, optimize=True) / S
        W = W - lr * coef[:, :, None] * grad
        Tv = Tv - eta_t * (1.0 - G)[:, None] * (1.0 - active / A[:, None])
    return W, Tv


def eval_losses(W_stack, X2, y_all, task):
    # per-user TRAIN losses for all models in one matmul: (Cs, N)
    Cs = W_stack.shape[0]
    N, S = y_all.shape
    Z = (X2 @ W_stack.T).T.reshape(Cs, N, S)
    if task == "mse":
        return ((Z - y_all[None]) ** 2).mean(axis=2)
    return (np.logaddexp(0.0, Z) - y_all[None] * Z).mean(axis=2)


def make_config_table(models, alphas, gammas):
    rows = []
    for m in GRID_MODELS:
        if m in models:
            for a in alphas:
                for g in gammas:
                    rows.append(dict(model=m, alpha=float(a), gamma=float(g)))
    for m in FREE_MODELS:
        if m in models:
            rows.append(dict(model=m, alpha=None, gamma=None))
    return rows


def run_unit(dataset, regime, seed, table, has_full, data, tp, groups, args, lr, eta_t):
    C = len(table)
    Cs = C + (1 if has_full else 0)
    if Cs == 0:
        return None
    X_all, y_all, X2, p = data["X_all"], data["y_all"], data["X2"], data["p"]
    task, D, N = data["task"], data["D"], data["N"]
    R, Kk, H, cap = args.rounds, args.k, args.local_epochs, args.cap

    A = np.array([row["alpha"] if row["alpha"] is not None else 1.0 for row in table])
    G = np.array([row["gamma"] if row["gamma"] is not None else 1.0 for row in table])
    unif = np.array([row["model"] in UNIF_AGG for row in table], dtype=bool)
    mk = np.array([row["model"] in MK_AGG for row in table], dtype=bool)
    ipw = np.array([row["model"] in IPW_AGG for row in table], dtype=bool)
    down = np.array([row["model"] in DOWN_AGG for row in table], dtype=bool)
    ratio_all = p / np.maximum(tp["pi"], 1e-300)          # p_i / pi_i, the under-observation ratio
    lds = np.array([row["model"] in LDS_AGG for row in table], dtype=bool)
    if lds.any():
        if task != "mse":
            raise SystemExit("model 'lds' (label distribution smoothing) is defined for regression only (imdbwiki)")
        lds_w = lds_group_weights(y_all)

    draws = np.searchsorted(tp["q_cum"], np.random.RandomState(seed).rand(R))
    user_every = max(1, args.user_log_every)
    user_rounds = np.array(sorted(set(range(0, R, user_every)) | {R - 1}))
    user_row_of = np.full(R, -1)
    user_row_of[user_rounds] = np.arange(len(user_rounds))

    gnames = list(groups)
    gidx = [groups[n] for n in gnames]
    NG = len(gnames)

    W_grid = np.zeros((C, D))
    t_grid = np.full(C, args.t0, dtype=float)
    w_full = np.zeros(D)
    alive = np.ones(Cs, dtype=bool)
    death_round = np.full(Cs, -1, dtype=int)
    overall_buf = np.empty((Cs, R))
    group_buf = np.empty((Cs, R, NG), dtype=np.float32)
    user_buf = np.empty((Cs, len(user_rounds), N), dtype=np.float32)
    ONE1, ZERO1 = np.ones(1), np.zeros(1)
    adult_cvar_note_done = False

    t_start = time.time()
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(R):
            j = draws[t]
            users = tp["subs0"][j]
            wj = tp["Wcols"][j]
            if args.lr_schedule == "cosine":
                lr_t = lr * 0.5 * (1.0 + np.cos(np.pi * t / R))
            elif args.lr_decay is None:
                lr_t = lr
            elif args.lr_schedule == "sqrt":
                lr_t = lr / np.sqrt(1.0 + t / args.lr_decay)
            elif args.lr_schedule == "step":
                lr_t = lr * 0.5 ** (t // args.lr_decay)
            else:
                lr_t = lr / (1.0 + t / args.lr_decay)
            if C:
                Wl, Tvl = grid_local_step(X_all[users], y_all[users], W_grid, t_grid,
                                          A, G, H, lr_t, eta_t, task)
                Agg = np.where(unif[:, None], 1.0 / Kk, wj[None, :])      # (C, K)
                if mk.any():
                    Agg = np.where(mk[:, None], (N / Kk) * p[users][None, :], Agg)
                if ipw.any():
                    rw = ratio_all[users]; rw = rw / rw.sum()
                    Agg = np.where(ipw[:, None], rw[None, :], Agg)
                if down.any():
                    dw = np.minimum(ratio_all[users], 1.0); dw = dw / dw.sum()
                    Agg = np.where(down[:, None], dw[None, :], Agg)
                if lds.any():
                    lw = lds_w[users]; lw = lw / lw.sum()
                    Agg = np.where(lds[:, None], lw[None, :], Agg)
                W_new = np.einsum('ck,ckd->cd', Agg, Wl)
                t_new = (Agg * Tvl).sum(axis=1)
                W_grid = np.where(alive[:C, None], W_new, W_grid)
                t_grid = np.where(alive[:C], t_new, t_grid)
            if has_full and alive[C]:
                Wf, _ = grid_local_step(X_all, y_all, w_full[None, :], ZERO1, ONE1, ONE1,
                                        H, lr_t, eta_t, task)
                w_full = p @ Wf[0]
            W_stack = np.vstack([W_grid, w_full[None, :]]) if has_full else W_grid
            ul = eval_losses(W_stack, X2, y_all, task)
            overall = ul @ p
            gm = np.stack([ul[:, ix].mean(axis=1) for ix in gidx], axis=1)   # (Cs, NG)

            bad = (~np.isfinite(overall)) | (overall > cap)
            lm = alive & ~bad
            overall_buf[:, t] = np.where(lm, overall, cap)
            group_buf[:, t, :] = np.where(lm[:, None], gm, cap)
            ur = user_row_of[t]
            if ur >= 0:
                user_buf[:, ur, :] = np.where(lm[:, None], ul, cap)
            newly = alive & bad
            if newly.any():
                death_round[newly] = t
                alive &= ~bad
                ng = newly[:C]
                W_grid[ng] = 0.0
                t_grid[ng] = 0.0
                if has_full and newly[C]:
                    w_full = np.zeros(D)
                for ci in np.where(newly)[0]:
                    name = table[ci]["model"] if ci < C else "full"
                    ag = (f"(a={table[ci]['alpha']:.2f}, g={table[ci]['gamma']:.2f}) "
                          if ci < C and table[ci]["alpha"] is not None else "")
                    print(f"    [death] {name} {ag}diverged at round {t}, pinned at cap")
                    if (dataset == "adult" and ci < C and table[ci]["model"] in GRID_MODELS
                            and not adult_cvar_note_done):
                        adult_cvar_note_done = True
                        print("    note: Adult+CVaR is untested territory (eta_t=LR/20 "
                              "heuristic); divergence regions in the heatmap are data, "
                              "not bugs. --eta-t is the tuning valve.")
            if args.progress_every and (t + 1) % args.progress_every == 0:
                print(f"    round {t + 1}/{R} ({time.time() - t_start:.0f}s)")

    elapsed = time.time() - t_start
    line = " ".join(f"{table[ci]['model']}={overall_buf[ci, -args.tail:].mean():.4g}"
                    for ci in range(C) if table[ci]["alpha"] is None)
    if has_full:
        line += f" full={overall_buf[C, -args.tail:].mean():.4g}"
    print(f"    seed {seed}: {elapsed:.0f}s, {Cs} models, "
          f"{(death_round >= 0).sum()} diverged; tail-{args.tail} {line}")
    return dict(table=table, has_full=has_full, overall=overall_buf, group=group_buf,
                user=user_buf, user_rounds=user_rounds, death_round=death_round,
                gnames=gnames, elapsed=elapsed)


# ================================================================
# Output
# ================================================================
def result_filename(outdir, dataset, regime, model, alpha, gamma, seed, gz):
    ag = f"_a{alpha:.2f}_g{gamma:.2f}" if alpha is not None else ""
    return os.path.join(outdir, f"{dataset}_{regime}_{model}{ag}_seed{seed}.csv"
                        + (".gz" if gz else ""))


def write_result_csv(path, R, overall, group_rows, users_rows, user_rounds, gnames, N, gz):
    users_full = np.full((R, N), np.nan)
    users_full[user_rounds] = users_rows
    cols = {"round": np.arange(R), "overall": overall}
    for gi, name in enumerate(gnames):
        cols[f"group_{name}"] = group_rows[:, gi].astype(float)
    for u in range(N):
        cols[f"user_{u}"] = users_full[:, u]
    df = pd.DataFrame(cols)
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, float_format="%.7g",
              compression="gzip" if gz else None)
    os.replace(tmp, path)


def unit_summary_rows(res, dataset, regime, seed, args, outdir_rel):
    rows = []
    C = len(res["table"])
    Cs = C + (1 if res["has_full"] else 0)
    for ci in range(Cs):
        if ci < C:
            model = res["table"][ci]["model"]
            alpha, gamma = res["table"][ci]["alpha"], res["table"][ci]["gamma"]
        else:
            model, alpha, gamma = "full", None, None
        gt = res["group"][ci, -args.tail:, :].mean(axis=0)   # (NG,)
        wi = int(np.argmax(gt))
        row = {
            "dataset": dataset, "regime": regime, "model": model,
            "alpha": f"{alpha:.2f}" if alpha is not None else "",
            "gamma": f"{gamma:.2f}" if gamma is not None else "",
            "seed": str(seed), "rounds": str(args.rounds), "tail": str(args.tail),
            "diverged": str(int(res["death_round"][ci] >= 0)),
            "death_round": str(res["death_round"][ci]) if res["death_round"][ci] >= 0 else "",
            "overall_tail": f"{res['overall'][ci, -args.tail:].mean():.6g}",
            "worst_group": res["gnames"][wi],
            "worst_group_tail": f"{gt[wi]:.6g}",
        }
        for gi, name in enumerate(res["gnames"]):
            row[f"group_tail_{name}"] = f"{gt[gi]:.6g}"
        row["file"] = os.path.basename(
            result_filename(outdir_rel, dataset, regime, model, alpha, gamma, seed, args.gzip))
        rows.append(row)
    return rows


def update_summary(outdir, new_rows):
    if not new_rows:
        return
    spath = os.path.join(outdir, "summary.csv")
    new = pd.DataFrame(new_rows).astype(str)
    if os.path.exists(spath):
        old = pd.read_csv(spath, dtype=str, keep_default_na=False)
        allcols = list(dict.fromkeys(list(old.columns) + list(new.columns)))
        old = old.reindex(columns=allcols, fill_value="")
        new = new.reindex(columns=allcols, fill_value="")
        newkeys = set(map(tuple, new[SUMMARY_KEY].values))
        keep = [tuple(k) not in newkeys for k in old[SUMMARY_KEY].values]
        combined = pd.concat([old[keep], new], ignore_index=True)
    else:
        combined = new
    combined = combined.sort_values(SUMMARY_KEY, kind="stable")
    tmp = spath + ".tmp"
    combined.to_csv(tmp, index=False)
    os.replace(tmp, spath)


def rebuild_summary(args):
    files = [f for f in os.listdir(args.outdir) if FNAME_RE.match(f)]
    print(f"rebuilding summary from {len(files)} CSVs in {args.outdir}")
    rows = []
    for i, fname in enumerate(sorted(files)):
        m = FNAME_RE.match(fname).groupdict()
        head = pd.read_csv(os.path.join(args.outdir, fname), nrows=1)
        gcols = [c for c in head.columns if c.startswith("group_")]
        df = pd.read_csv(os.path.join(args.outdir, fname),
                         usecols=["round", "overall"] + gcols)
        tail = df.iloc[-args.tail:]
        gt = tail[gcols].mean().values
        wi = int(np.argmax(gt))
        overall = df["overall"].values
        dead = overall.max() >= args.cap
        death = int(np.argmax(overall >= args.cap)) if dead else -1
        row = {
            "dataset": m["dataset"], "regime": m["regime"], "model": m["model"],
            "alpha": m["alpha"] or "", "gamma": m["gamma"] or "",
            "seed": m["seed"], "rounds": str(len(df)), "tail": str(min(args.tail, len(df))),
            "diverged": str(int(dead)),
            "death_round": str(death) if dead else "",
            "overall_tail": f"{tail['overall'].mean():.6g}",
            "worst_group": gcols[wi][len('group_'):],
            "worst_group_tail": f"{gt[wi]:.6g}",
        }
        for gi, c in enumerate(gcols):
            row[f"group_tail_{c[len('group_'):]}"] = f"{gt[gi]:.6g}"
        row["file"] = fname
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(files)}")
    spath = os.path.join(args.outdir, "summary.csv")
    if os.path.exists(spath):
        os.remove(spath)
    update_summary(args.outdir, rows)
    print(f"wrote {spath} ({len(rows)} rows)")


# ================================================================
# Selftest / smoke
# ================================================================
def _check(cond, msg):
    if not cond:
        raise SystemExit(f"SELFTEST FAIL: {msg}")
    print(f"  [selftest] OK: {msg}")


def run_selftest(args):
    print("=== selftest (imdbwiki, N=20, K=3, 200 rounds, infeasible) ===")
    args.num_users, args.k, args.rounds = 20, 3, 200
    args.q_samples, args.pi_samples = 200_000, 100_000
    args.user_log_every = 1
    args.outdir = os.path.join(args.outdir, "selftest")
    os.makedirs(args.outdir, exist_ok=True)
    lr, eta_t = DATASET_LR["imdbwiki"], DATASET_ETA_T["imdbwiki"]

    data = load_imdbwiki(args)
    p, r = make_imdb_distributions(args.num_users, "infeasible")
    data["p"] = p
    tp = get_transport("imdbwiki", "infeasible", p, r, args, verify=True)
    groups = build_groups("imdbwiki", data, p, tp["pi"], args)

    # (iii) draw-sequence equivalence: vector rand == per-round rand loop
    rng_a = np.random.RandomState(0)
    vec = rng_a.rand(args.rounds)
    rng_b = np.random.RandomState(0)
    loop = np.array([rng_b.rand() for _ in range(args.rounds)])
    _check(np.array_equal(vec, loop), "pre-generated draws == per-round rand() loop")

    table = make_config_table(list(ALL_MODELS), [0.3, 1.0], [0.3, 1.0])
    res = run_unit("imdbwiki", "infeasible", 0, table, True, data, tp, groups, args, lr, eta_t)

    idx_of = {(row["model"], row["alpha"], row["gamma"]): i
              for i, row in enumerate(table)}
    # (i) gamma=1.0 collapse: exact in exact arithmetic; in floats, batched-BLAS
    # summation order differs by row position (measured ~6e-16 rel), so enforce
    # exact-or-tolerance instead of bitwise
    for a in (0.3, 1.0):
        for grid_m, base_m in (("fedavot_cvar", "fedavot"), ("fedcvar", "fedavg")):
            ic = idx_of[(grid_m, a, 1.0)]
            ib = idx_of[(base_m, None, None)]
            rels = []
            for buf, tol in (("overall", 1e-12), ("group", 1e-6), ("user", 1e-6)):
                x = res[buf][ic].astype(float)
                y = res[buf][ib].astype(float)
                rel = float(np.max(np.abs(x - y) / np.maximum(np.abs(y), 1e-12)))
                rels.append(rel)
                if not rel < tol:
                    raise SystemExit(f"SELFTEST FAIL: {grid_m}(a={a}, g=1.0) vs "
                                     f"{base_m} {buf} rel diff {rel:.2e} >= {tol}")
            print(f"  [selftest] OK: {grid_m}(a={a}, g=1.0) == {base_m} "
                  f"(max rel {max(rels):.1e}, last-ulp BLAS row-position effects only)")

    # (ii) sweep row == single-config rerun (same engine; cross-call BLAS batching may
    # differ at the last ulp, so exact first, tolerance fallback with report)
    single = [dict(model="fedavot_cvar", alpha=0.3, gamma=0.3)]
    res1 = run_unit("imdbwiki", "infeasible", 0, single, False, data, tp, groups, args, lr, eta_t)
    ic = idx_of[("fedavot_cvar", 0.3, 0.3)]
    if np.array_equal(res["overall"][ic], res1["overall"][0]):
        print("  [selftest] OK: sweep row == single-config rerun (bitwise)")
    else:
        rel = np.max(np.abs(res["overall"][ic] - res1["overall"][0])
                     / np.maximum(np.abs(res["overall"][ic]), 1e-12))
        _check(rel < 1e-9, f"sweep row == single-config rerun (max rel diff {rel:.2e})")

    # (v) CSV round-trip at format precision
    path = result_filename(args.outdir, "imdbwiki", "infeasible", "fedavot_cvar",
                           0.3, 0.3, 0, False)
    write_result_csv(path, args.rounds, res["overall"][ic], res["group"][ic],
                     res["user"][ic], res["user_rounds"], res["gnames"], data["N"], False)
    back = pd.read_csv(path)
    rel_o = np.max(np.abs(back["overall"].values - res["overall"][ic])
                   / np.maximum(np.abs(res["overall"][ic]), 1e-12))
    ucols = [f"user_{u}" for u in range(data["N"])]
    rel_u = np.max(np.abs(back[ucols].values - res["user"][ic].astype(float))
                   / np.maximum(np.abs(res["user"][ic]), 1e-12))
    _check(rel_o < 5e-7 and rel_u < 5e-6 and len(back) == args.rounds,
           f"CSV round-trip at format precision (overall {rel_o:.1e}, user {rel_u:.1e})")
    print("=== SELFTEST PASSED ===")


# ================================================================
# Main
# ================================================================
def main(argv=None):
    try:   # keep progress visible when stdout is redirected (background runs)
        sys.stdout.reconfigure(line_buffering=True)
    except AttributeError:
        pass
    args = parse_args(argv)

    if args.rebuild_summary:
        rebuild_summary(args)
        return
    if args.selftest:
        run_selftest(args)
        return
    if args.smoke:
        print("=== smoke: imdbwiki/infeasible, seed 0, 50 rounds, alphas [0.3], "
              "gammas [0.3, 1.0], sweep ===")
        args.datasets, args.regimes = ["imdbwiki"], ["infeasible"]
        args.seeds, args.rounds, args.sweep = [0], 50, True
        args.alphas, args.gammas = [0.3], [0.3, 1.0]
        args.progress_every = 0
        args.outdir = os.path.join(args.outdir, "smoke")

    os.makedirs(args.outdir, exist_ok=True)

    # plan the units
    grid_n = sum(1 for m in GRID_MODELS if m in args.models) * len(args.alphas) * len(args.gammas)
    free_n = sum(1 for m in FREE_MODELS if m in args.models)
    per_unit_files = grid_n + free_n + (1 if "full" in args.models else 0)
    n_units = len(args.datasets) * len(args.regimes) * len(args.seeds)
    total_files = per_unit_files * n_units
    est_mb = total_files * (args.rounds / 4000) * 3.5
    print(f"planned: {n_units} units x {per_unit_files} configs = {total_files} CSVs, "
          f"~{est_mb / 1024:.1f} GB at full per-user logging"
          + (" (gzip ~4x smaller)" if not args.gzip else " (gzip on)"))
    if not args.sweep and total_files > 20:
        print(f"WARNING: {total_files} configs WITHOUT --sweep run as sequential "
              f"single-config loops (days, not hours). Use --sweep.")
    if not args.gzip and est_mb > 5000:
        print("note: consider --gzip (plotter reads .csv.gz transparently)")
    if args.dry_run:
        for ds in args.datasets:
            for regime in args.regimes:
                for seed in args.seeds:
                    print(f"  unit: {ds}/{regime}/seed{seed} "
                          f"({per_unit_files} configs{' vectorized' if args.sweep else ''})")
        return

    if (args.r_power is not None or args.lr_decay is not None or args.lr_schedule != "hyper") and            os.path.abspath(args.outdir) == os.path.abspath("results/runs/default"):
        raise SystemExit("--r-power / --lr-decay change the experiment but not the filenames: "
                         "pass a dedicated --outdir (results/runs/<cell>).")
    t_all = time.time()
    smoke_stats = {}
    for ds in args.datasets:
        print(f"\n=== dataset: {ds} ===")
        data = load_imdbwiki(args) if ds == "imdbwiki" else load_adult(args)
        lr = args.lr if args.lr is not None else DATASET_LR[ds]
        eta_t = args.eta_t if args.eta_t is not None else DATASET_ETA_T[ds]
        for regime in args.regimes:
            print(f"--- regime: {regime} ---")
            p, r = dataset_distributions(ds, regime, data, args)
            data["p"] = p
            tp = get_transport(ds, regime, p, r, args)
            groups = build_groups(ds, data, p, tp["pi"], args)
            print(f"  critical groups: " + ", ".join(f"{n}({len(ix)})"
                                                     for n, ix in groups.items()))
            full_table = make_config_table(args.models, args.alphas, args.gammas)
            for seed in args.seeds:
                if args.sweep:
                    unit_tables = [(full_table, "full" in args.models)]
                else:
                    unit_tables = [([row], False) for row in full_table]
                    if "full" in args.models:
                        unit_tables.append(([], True))
                for table, has_full in unit_tables:
                    if args.no_clobber:
                        table = [row for row in table if not os.path.exists(
                            result_filename(args.outdir, ds, regime, row["model"],
                                            row["alpha"], row["gamma"], seed, args.gzip))]
                        if has_full and os.path.exists(result_filename(
                                args.outdir, ds, regime, "full", None, None, seed, args.gzip)):
                            has_full = False
                    if not table and not has_full:
                        continue
                    res = run_unit(ds, regime, seed, table, has_full, data, tp,
                                   groups, args, lr, eta_t)
                    if res is None:
                        continue
                    C = len(table)
                    for ci in range(C + (1 if has_full else 0)):
                        if ci < C:
                            model = table[ci]["model"]
                            alpha, gamma = table[ci]["alpha"], table[ci]["gamma"]
                        else:
                            model, alpha, gamma = "full", None, None
                        path = result_filename(args.outdir, ds, regime, model,
                                               alpha, gamma, seed, args.gzip)
                        write_result_csv(path, args.rounds, res["overall"][ci],
                                         res["group"][ci], res["user"][ci],
                                         res["user_rounds"], res["gnames"],
                                         data["N"], args.gzip)
                    update_summary(args.outdir,
                                   unit_summary_rows(res, ds, regime, seed, args, args.outdir))
                    smoke_stats = dict(elapsed=res["elapsed"],
                                       Cs=C + (1 if has_full else 0),
                                       one_file=path)
    print(f"\nall done in {(time.time() - t_all) / 60:.1f} min; "
          f"results + summary.csv in {args.outdir}")

    if args.smoke and smoke_stats:
        ms_round = smoke_stats["elapsed"] / args.rounds * 1000
        full_cs = 2 * 100 + 2 + 1        # default sweep: 202 grid+free rows + full
        proj_unit = ms_round * (full_cs / smoke_stats["Cs"]) * 4000 / 1000
        size = os.path.getsize(smoke_stats["one_file"])
        n_files_full = full_cs * 20      # 2 datasets x 2 regimes x 5 seeds
        proj_disk = size / (args.rounds + 1) * 4001 * n_files_full / 1e9
        print(f"\nsmoke projection (rough, linear in configs):")
        print(f"  measured {ms_round:.1f} ms/round at {smoke_stats['Cs']} models")
        print(f"  full default sweep (~{full_cs} models, 4000 rounds, 20 units): "
              f"~{proj_unit * 20 / 60:.0f} min training + transport setup")
        print(f"  disk at full per-user logging: ~{proj_disk:.1f} GB "
              f"({n_files_full} files); --gzip ~4x smaller")


if __name__ == "__main__":
    main()
