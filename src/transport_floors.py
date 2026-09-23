# Marginals and exact loss floors from a cached transport plan (no training).
#
# For each cell the runner caches results/runs/<cell>/transport_<dataset>_<regime>_K3.npz
# (q, Wcols, pi, p, r). From it:
#   p_hat_i   = sum_j q_j W[i, j]   expected per-step weight GAVOT gives group i
#                                   (= p when IPFP converges; the surrogate marginal otherwise)
#   p_tilde_i = pi_i / K            expected per-step weight of group-blind averaging
# and the floor Phi(w) = F_p(theta*_w), the target loss at the minimizer of the w-weighted
# objective each rule converges toward (closed-form least squares on IMDb-Wiki, Newton
# weighted logistic regression on Adult).
import numpy as np
import run_experiments as RE


def floor_mse(data, w):
    X, y, S = data["X_all"], data["y_all"], data["S"]
    A = np.einsum('n,nsd,nse->de', w, X, X) / S
    b = np.einsum('n,nsd,ns->d', w, X, y) / S
    th = np.linalg.solve(A + 1e-9 * np.eye(data["D"]), b)
    return ((np.einsum('nsd,d->ns', X, th) - y) ** 2).mean(1)


def floor_ce(data, w, iters=60):
    X, y, S, D = data["X_all"], data["y_all"], data["S"], data["D"]
    th = np.zeros(D)
    for _ in range(iters):
        z = np.einsum('nsd,d->ns', X, th); s = 1 / (1 + np.exp(-z))
        g = np.einsum('n,nsd,ns->d', w, X, s - y) / S + 1e-6 * th
        H = np.einsum('n,ns,nsd,nse->de', w, s * (1 - s), X, X) / S + 1e-6 * np.eye(D)
        step = np.linalg.solve(H, g); th -= step
        if np.abs(step).max() < 1e-9: break
    z = np.einsum('nsd,d->ns', X, th)
    return (np.logaddexp(0, z) - y * z).mean(1)


def cell_marginals(transport_npz, N, K):
    """(p, pi, p_hat, p_tilde, infeasible mass nu) from a cached transport plan."""
    z = np.load(transport_npz)
    p, pi, q, W = z["p"], z["pi"], z["q"], z["Wcols"]
    subs0 = np.array(RE.all_K_subsets_1based(N, K), dtype=np.int64) - 1
    p_hat = np.bincount(subs0.ravel(), weights=(q[:, None] * W).ravel(), minlength=N)
    return p, pi, p_hat, pi / K, float(p[p > pi].sum())


def cell_floors(data, p, p_hat, p_tilde):
    fl = floor_mse if data["task"] == "mse" else floor_ce
    return {name: float(p @ fl(data, w)) for name, w in
            [("fedavot", p_hat), ("fedavg", p_tilde / p_tilde.sum()), ("full", p)]}
