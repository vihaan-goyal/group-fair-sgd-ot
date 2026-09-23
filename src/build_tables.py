# Rebuild results/tables/{severity_sweep_table,rare_group_table}.csv from the raw runs in
# results/runs/<cell>/ (per-seed CSVs + cached transport plans written by run_experiments.py).
#   severity_sweep_table.csv  per cell: infeasible mass nu, ||p - p_hat||_1, ||p - p_tilde||_1,
#                             exact floors Phi per rule, measured tail-500 F_p mean / std
#   rare_group_table.csv      per cell and rule: tail-500 loss of the least represented group
#                             (Adult: race Other and Amer-Indian-Eskimo; IMDb-Wiki: tier 1 = the 20
#                             highest-importance identities) and of the single most starved group
# Needs the data (the floors refit each rule's weighted objective) and the raw per-seed CSVs,
# i.e. a completed experiments/severity_sweep.sh. The committed tables are its output; the
# figure script reads only them.
# Usage (repo root): python src/build_tables.py
import os, sys, csv, glob, gzip, re, statistics as st
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE
from transport_floors import cell_marginals, cell_floors

ROOT = "results/runs"
OUT = "results/tables"
RULES = ["fedavot", "fedavg", "full"]
RARE = {"adult": ["Other", "Amer-Indian-Eskimo"], "imdbwiki": ["tier1"]}
RARE_MODELS = ["fedavot", "fedavg", "fedavg_mk", "full"]
TAIL_USER_ROWS = 50   # per-group losses are logged every 10th step: last 500 steps = 50 rows

args = RE.parse_args([])
N, K = args.num_users, args.k
data, sev, rare = {}, [], []
for cell in sorted(glob.glob(os.path.join(ROOT, "*_*"))):
    name = os.path.basename(cell)
    m = re.match(r"(imdb|adult)_(beta([\d.]+)|feasible)_(const|decay\d+)$", name)
    if not m or not os.path.exists(os.path.join(cell, "summary.csv")):
        continue
    ds = "imdbwiki" if m.group(1) == "imdb" else "adult"
    regime = "feasible" if m.group(2) == "feasible" else "infeasible"
    beta = float(m.group(3)) if m.group(3) else None
    tag = m.group(4)
    if ds not in data:
        data[ds] = RE.load_imdbwiki(args) if ds == "imdbwiki" else RE.load_adult(args)
    tpath = os.path.join(cell, f"transport_{ds}_{regime}_K{K}.npz")
    p, pi, p_hat, p_til, nu = cell_marginals(tpath, N, K)
    floors = cell_floors(data[ds], p, p_hat, p_til)
    srows = [r for r in csv.DictReader(open(os.path.join(cell, "summary.csv"))) if not r["alpha"]]
    meas = {}
    for mdl in RULES:
        v = [float(r["overall_tail"]) for r in srows if r["model"] == mdl]
        meas[mdl] = (st.mean(v), st.pstdev(v), len(v)) if v else (np.nan, np.nan, 0)
    sev.append(dict(dataset=ds, regime=regime, beta=beta, tag=tag, n_infeasible=int((p > pi).sum()),
                    infeasible_mass=nu, l1_p_phat=float(np.abs(p - p_hat).sum()),
                    l1_p_ptilde=float(np.abs(p - p_til).sum()),
                    **{f"floor_{k}": v for k, v in floors.items()},
                    **{f"meas_{k}": meas[k][0] for k in RULES}, **{f"std_{k}": meas[k][1] for k in RULES},
                    n_seeds=meas["fedavot"][2]))

    for grp in RARE[ds]:
        for mdl in RARE_MODELS:
            v = [float(r[f"group_tail_{grp}"]) for r in srows if r["model"] == mdl]
            ov = [float(r["overall_tail"]) for r in srows if r["model"] == mdl]
            if v:
                rare.append(dict(cell=name, dataset=ds, regime=regime, beta=beta, tag=tag, nu=nu, group=grp,
                                 model=mdl, n_seeds=len(v), rare_mean=st.mean(v), rare_std=st.pstdev(v),
                                 overall_mean=st.mean(ov), overall_std=st.pstdev(ov)))
    # most starved single group: largest p_i / pi_i in the cell (ties averaged)
    ratio = p / np.maximum(pi, 1e-300)
    top = [int(i) for i in np.where(np.isclose(ratio, ratio.max(), rtol=1e-6))[0]]
    for mdl in RARE_MODELS:
        vals = []
        for f in sorted(glob.glob(os.path.join(cell, f"{ds}_{regime}_{mdl}_seed*.csv.gz"))):
            recs = list(csv.DictReader(gzip.open(f, "rt")))
            per_u = [st.mean([float(r[f"user_{u}"]) for r in recs if r[f"user_{u}"] != ""][-TAIL_USER_ROWS:])
                     for u in top]
            vals.append(st.mean(per_u))
        if vals:
            rare.append(dict(cell=name, dataset=ds, regime=regime, beta=beta, tag=tag, nu=nu, group="most_biased",
                             model=mdl, n_seeds=len(vals), rare_mean=st.mean(vals), rare_std=st.pstdev(vals),
                             overall_mean=float("nan"), overall_std=float("nan"),
                             most_biased_users=" ".join(map(str, top)), p_over_r=float(ratio.max())))

os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "severity_sweep_table.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(sev[0])); wr.writeheader(); wr.writerows(sev)
with open(os.path.join(OUT, "rare_group_table.csv"), "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rare[0]) + ["most_biased_users", "p_over_r"], restval="")
    wr.writeheader(); wr.writerows(rare)
print(f"wrote {OUT}/severity_sweep_table.csv ({len(sev)} cells) and rare_group_table.csv ({len(rare)} rows)")
