# Sec. 5 check: the 1000-sweep IPFP cap used by run_experiments.py (--ipfp-iters) already gives the
# limiting surrogate marginal p_hat. Runs IPFP to 20000 sweeps at beta = 0 and beta = 3 on IMDb-Wiki and
# reports KL(p_hat || p), the L1 drift of p_hat from its 1000-sweep value, and the floor Phi(p_hat).
# Usage (repo root): python src/ipfp_limit_check.py
import sys, types, time, numpy as np
sys.path.insert(0, "src")
import run_experiments as R
args = types.SimpleNamespace(num_users=100, samples_per_user=30, q_samples=1_000_000, pi_samples=1_000_000)
data = R.load_imdbwiki(args)
X, y = data["X_all"], data["y_all"]; S = y.shape[1]
XtX = np.einsum('nsd,nse->nde', X, X) / S; Xty = np.einsum('nsd,ns->nd', X, y) / S
def f_all(th): return ((np.einsum('nsd,d->ns', X, th) - y) ** 2).mean(axis=1)
def Phi(w, p):
    th = np.linalg.solve(np.tensordot(w, XtX, 1), w @ Xty); return p @ f_all(th)
N, K = 100, 3
subs0 = np.array(R.all_K_subsets_1based(N, K)) - 1
for beta in (0.0, 3.0):
    idx = np.arange(1, N + 1); p = idx[::-1] ** 3.0; p /= p.sum()
    r = idx ** beta; r = r / r.sum()
    q = R.estimate_q([tuple(s + 1) for s in subs0], r, N, K, args.q_samples, np.random.RandomState(0))
    q = (q + 1e-12) / q.sum()
    Yc = np.repeat(q[:, None] / K, K, axis=1)            # (J,K) compact plan
    flat = subs0.ravel(); t0 = time.time()
    snaps = {}
    for it in range(1, 20001):
        rs = np.bincount(flat, Yc.ravel(), minlength=N)
        Yc *= (p / np.maximum(rs, 1e-12))[subs0]
        Yc *= (q / Yc.sum(axis=1))[:, None]
        if it in (1000, 3000, 10000, 20000):
            ph = np.bincount(flat, Yc.ravel(), minlength=N); snaps[it] = ph
            kl = np.sum(ph * np.log(np.maximum(ph, 1e-300) / p))
            print(f"beta={beta} it={it:5d} ({time.time()-t0:.0f}s) KL(ph||p)={kl:.6f} "
                  f"|ph-ph1000|_1={np.abs(ph - snaps[1000]).sum():.2e} Phi(ph)={Phi(ph, p):.3f}", flush=True)
