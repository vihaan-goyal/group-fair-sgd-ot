# Tables 1 and 2 of the paper, and every Welch t quoted in Sec. 5, from the per-cell
# results/runs/<cell>/summary.csv files (tail-500 mean training loss per seed). No training.
# Welch t uses the sample standard deviation (ddof = 1) over the 5 seeds.
# Usage (repo root): python src/paper_tables.py
import csv, os, statistics as st

RUNS = "results/runs"
NAME = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "ipw": "upsampling",
        "lds": "LDS", "fedavg_mk": "fixed multiplier m/K", "full": "full access"}

def tails(cell, model):
    rows = csv.DictReader(open(os.path.join(RUNS, cell, "summary.csv")))
    return [float(r["overall_tail"]) for r in rows if r["model"] == model and not r["alpha"]]

def welch_t(a, b):
    """t of mean(a) - mean(b), sample variances."""
    return (st.mean(a) - st.mean(b)) / (st.variance(a) / len(a) + st.variance(b) / len(b)) ** 0.5

# ---- Table 1: overall F_p, decaying stepsize, four headline instances
T1 = [("Adult, beta=0", "adult_beta0.00_decay1000", 4), ("Adult, beta=1", "adult_beta1.00_decay1000", 4),
      ("IMDb, beta=0", "imdb_beta0.00_decay1000", 2), ("IMDb, aligned", "imdb_feasible_decay1000", 2)]
print("Table 1  mean tail-500 F_p over 5 seeds (decaying stepsize)")
print(f"{'instance':15s}" + "".join(f"{NAME[m]:>18s}" for m in ["fedavot", "fedavg", "ipw", "lds", "full"])
      + f"{'t vs avg':>10s}{'t vs ups':>10s}")
for label, cell, d in T1:
    vals = {m: tails(cell, m) for m in ["fedavot", "fedavg", "ipw", "lds", "full"]}
    line = f"{label:15s}" + "".join(f"{st.mean(v):>18.{d}f}" if v else f"{'--':>18s}" for v in vals.values())
    t_avg = welch_t(vals["fedavg"], vals["fedavot"])
    t_ups = welch_t(vals["ipw"], vals["fedavot"]) if vals["ipw"] else float("nan")
    print(line + f"{t_avg:>10.1f}{t_ups:>10.1f}")

# ---- Table 2: IMDb-Wiki severity sweep, decaying stepsize, gain = group-blind - GAVOT
print("\nTable 2  IMDb-Wiki, decaying stepsize: Welch t of the gain (group-blind avg. - GAVOT)")
for b in ["0.00", "0.50", "1.00", "1.50", "2.00", "3.00"]:
    cell = f"imdb_beta{b}_decay1000"
    ot, avg = tails(cell, "fedavot"), tails(cell, "fedavg")
    print(f"beta={float(b):<4g} GAVOT {st.mean(ot):6.2f}  avg {st.mean(avg):6.2f}  "
          f"gain {st.mean(avg) - st.mean(ot):+5.2f}  t {welch_t(avg, ot):+5.1f}")

# ---- Sec. 5, mean-CVaR layer: best (alpha, gamma) grid point per headline cell, and the worst race on Adult
print("\nMean-CVaR grid (fedavot_cvar), decaying stepsize, PAPER convention (paper gamma = 1 - code gamma):")
pg = lambda k: f"(alpha={k[0]:g}, gamma={1 - k[1]:.1f})"
for label, cell, d in T1:
    rows = [r for r in csv.DictReader(open(os.path.join(RUNS, cell, "summary.csv"))) if r["model"] == "fedavot_cvar"]
    by = {}
    for r in rows:
        by.setdefault((float(r["alpha"]), float(r["gamma"])), []).append(r)
    mean = lambda key, col: st.mean(float(r[col]) for r in by[key])
    best = min(by, key=lambda k: mean(k, "overall_tail"))
    g1 = min((k for k in by if k[1] == 1.0), key=lambda k: mean(k, "overall_tail"))
    print(f"{label:15s} best {pg(best)} F_p={mean(best, 'overall_tail'):.{d}f}   "
          f"best risk-neutral (gamma=0) {pg(g1)} F_p={mean(g1, 'overall_tail'):.{d}f}")
    if cell == "adult_beta0.00_decay1000":
        wb = min(by, key=lambda k: mean(k, "worst_group_tail"))
        w = lambda m: st.mean(float(r["worst_group_tail"]) for r in csv.DictReader(open(os.path.join(RUNS, cell, "summary.csv")))
                              if r["model"] == m and not r["alpha"])
        print(f"{'':15s} worst race: CVaR best {pg(wb)} {mean(wb, 'worst_group_tail'):.4f}, "
              f"GAVOT {w('fedavot'):.4f}, group-blind {w('fedavg'):.4f}")
