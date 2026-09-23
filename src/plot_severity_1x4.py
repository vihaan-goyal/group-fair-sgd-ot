# Paper Fig. 2: both severity sweeps in one 1x4 row (IMDb-Wiki constant / decaying stepsize,
# Adult constant / decaying stepsize), from results/tables/severity_sweep_table.csv.
# Solid = tail-500 F_p mean over 5 seeds (+-1 std), dashed = exact floor Phi of each rule.
# 7 in wide for \textwidth; text at 9.3 pt so it prints at >= 9 pt (ICASSP kit); TrueType fonts.
# Usage (repo root): python src/plot_severity_1x4.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams["pdf.fonttype"] = 42

FS = 9.3
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
rows = list(csv.DictReader(open("results/tables/severity_sweep_table.csv")))

fig = plt.figure(figsize=(7.0, 2.05))
ax0 = fig.add_subplot(1, 4, 1); ax2 = fig.add_subplot(1, 4, 3)
axes = [ax0, fig.add_subplot(1, 4, 2, sharey=ax0), ax2, fig.add_subplot(1, 4, 4, sharey=ax2)]
panels = [("imdbwiki", "const", r"IMDb-Wiki, constant $\eta$"), ("imdbwiki", "decay1000", r"IMDb-Wiki, decaying $\eta_t$"),
          ("adult", "const", r"Adult, constant $\eta$"), ("adult", "decay1000", r"Adult, decaying $\eta_t$")]
for ax, (ds, tag, title) in zip(axes, panels):
    pts = [r for r in rows if r["dataset"] == ds and r["regime"] == "infeasible" and r["tag"] == tag]
    if ds == "imdbwiki":
        pts.sort(key=lambda r: float(r["infeasible_mass"])); x = [100 * float(r["infeasible_mass"]) for r in pts]
    else:
        pts.sort(key=lambda r: -float(r["beta"])); x = [float(r["beta"]) for r in pts]
    for m in ["fedavot", "fedavg", "full"]:
        y = [float(r[f"meas_{m}"]) for r in pts]; s = [float(r[f"std_{m}"]) for r in pts]
        ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=2.5, lw=1.2, capsize=1.5)
        if m != "full":
            ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.0)
    if ds == "imdbwiki":
        ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS); ax.set_xticks([40, 60, 80])
    else:
        ax.invert_xaxis(); ax.set_xticks([1, 0.5, 0]); ax.set_xticklabels(["1", "0.5", "0"])
        ax.set_xlabel(r"$\beta$ (weights $\propto p^{\beta}$)", fontsize=FS)
    ax.set_title(title, fontsize=FS, pad=4)
    ax.tick_params(labelsize=FS, pad=2); ax.grid(alpha=0.3)
axes[1].tick_params(labelleft=False); axes[3].tick_params(labelleft=False)
axes[0].set_ylabel("$F_p$ (MSE)", fontsize=FS, labelpad=2); axes[2].set_ylabel("$F_p$ (CE)", fontsize=FS, labelpad=2)
handles = [Line2D([], [], color=COL[m], marker=MK[m], ms=3, lw=1.2, label=LBL[m]) for m in LBL] + \
          [Line2D([], [], color="0.3", lw=1.2, label="measured"), Line2D([], [], color="0.3", ls="--", lw=1.0, label=r"floor $\Phi$")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=FS, frameon=False, bbox_to_anchor=(0.5, 1.0),
           handlelength=1.8, columnspacing=1.2)
fig.tight_layout(pad=0.3, w_pad=0.5, rect=(0, 0, 1, 0.86))
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/severity_1x4.pdf", bbox_inches="tight")
fig.savefig("figures/severity_1x4.png", dpi=200, bbox_inches="tight")
print("wrote figures/severity_1x4.{pdf,png}")
