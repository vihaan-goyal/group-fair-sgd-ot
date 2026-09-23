# Paper Fig. 2, 2x2 variant: rows = dataset (IMDb-Wiki vs infeasible mass nu, Adult vs skew beta),
# columns = constant | decaying stepsize. Solid = tail-500 F_p mean over 5 seeds (+-1 std),
# dashed = exact floor Phi of each rule. 7 in wide for \textwidth, text at 9.3 pt, TrueType fonts.
# Usage (repo root): python src/plot_severity_2x2.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["pdf.fonttype"] = 42

FS = 9.3
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
STEP = {"const": r"constant stepsize $\eta$", "decay1000": r"decaying stepsize $\eta_t=\eta/(1+t/1000)$"}
rows = list(csv.DictReader(open("results/tables/severity_sweep_table.csv")))

fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharey="row")
for i, ds in enumerate(["imdbwiki", "adult"]):
    for j, tag in enumerate(["const", "decay1000"]):
        ax = axes[i, j]
        pts = [r for r in rows if r["dataset"] == ds and r["regime"] == "infeasible" and r["tag"] == tag]
        if ds == "imdbwiki":
            pts.sort(key=lambda r: float(r["infeasible_mass"])); x = [100 * float(r["infeasible_mass"]) for r in pts]
        else:
            pts.sort(key=lambda r: -float(r["beta"])); x = [float(r["beta"]) for r in pts]
        for m in ["full", "fedavg", "fedavot"]:
            y = [float(r[f"meas_{m}"]) for r in pts]; s = [float(r[f"std_{m}"]) for r in pts]
            ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=3.5, lw=1.4, capsize=2, zorder=3)
            if m != "full":
                ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.1, zorder=2)
        if ds == "imdbwiki":
            ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS, labelpad=1)
            ax.margins(y=0.12)
        else:
            ax.invert_xaxis(); ax.set_xticks(x)
            ax.set_xticklabels([rf"$\beta$={float(r['beta']):g}" "\n" rf"$\nu$={float(r['infeasible_mass']):.2f}" for r in pts])
            ax.set_xlabel(r"sampling weights $\propto p^{\beta}$", fontsize=FS, labelpad=1)
        ax.set_title(("IMDb-Wiki" if ds == "imdbwiki" else "Adult") + ", " + STEP[tag], fontsize=FS, pad=3)
        ax.tick_params(labelsize=FS, pad=2, length=2.5)
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
axes[0, 0].set_ylabel(r"$F_p$ (MSE)", fontsize=FS); axes[1, 0].set_ylabel(r"$F_p$ (cross-entropy)", fontsize=FS)
handles = [Line2D([], [], color=COL[m], marker=MK[m], ms=3.5, lw=1.4, label=LBL[m]) for m in LBL] + \
          [Line2D([], [], color="0.35", lw=1.4, label="measured"),
           Line2D([], [], color="0.35", ls="--", lw=1.1, label=r"floor $\Phi$")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=2.0, columnspacing=1.5, borderaxespad=0.1)
fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=1.0, rect=(0, 0, 1, 0.93))
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/severity_2x2.pdf")
fig.savefig("figures/severity_2x2.png", dpi=200)
print("wrote figures/severity_2x2.{pdf,png}")
