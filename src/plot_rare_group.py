# Paper Fig. 3: loss on the least represented groups, decaying stepsize, from results/tables/rare_group_table.csv.
#   left:  Adult, cross-entropy on race Other vs skew beta (aligned beta = 1 on the left)
#   right: IMDb-Wiki, MSE on the 20 top-importance identities vs infeasible mass nu
# Mean over 5 seeds, +-1 std. 5.95 in wide = .85\textwidth, so it is placed unscaled; 7.5 pt text, TrueType fonts.
# Usage (repo root): python src/plot_rare_group.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42

FS = 7.5
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
rare = list(csv.DictReader(open("results/tables/rare_group_table.csv")))

def pick(ds, grp, mdl):
    return [r for r in rare if r["dataset"] == ds and r["group"] == grp and r["tag"] == "decay1000"
            and r["model"] == mdl and r["regime"] == "infeasible"]

fig, axes = plt.subplots(1, 2, figsize=(5.95, 2.0))
for ax, ds, grp in [(axes[0], "adult", "Other"), (axes[1], "imdbwiki", "tier1")]:
    for mdl in ["full", "fedavg", "fedavot"]:
        pts = pick(ds, grp, mdl)
        if ds == "adult":
            pts.sort(key=lambda r: -float(r["beta"])); x = [float(r["beta"]) for r in pts]
        else:
            pts.sort(key=lambda r: float(r["nu"])); x = [100 * float(r["nu"]) for r in pts]
        y = [float(r["rare_mean"]) for r in pts]; s = [float(r["rare_std"]) for r in pts]
        ax.errorbar(x, y, yerr=s, color=COL[mdl], marker=MK[mdl], ms=2.5, lw=1.1, capsize=1.2,
                    label=LBL[mdl], zorder=3)
    if ds == "adult":
        ax.invert_xaxis(); ax.set_xticks(x)
        ax.set_xticklabels([rf"{float(r['beta']):g}" "\n" rf"{float(r['nu']):.2f}" for r in pts])
        ax.set_xlabel(r"skew $\beta$ (upper), infeasible mass $\nu$ (lower)", fontsize=FS, labelpad=2)
        ax.set_title("Adult, race Other", fontsize=FS, pad=3)
        ax.set_ylabel("cross-entropy", fontsize=FS, labelpad=2)
        ax.set_ylim(0.0, 0.16)
        h, l = ax.get_legend_handles_labels()
        ax.legend(h[::-1], l[::-1], loc="upper left", fontsize=FS, framealpha=0.9, handlelength=1.6, handletextpad=0.4,
                  borderpad=0.3, labelspacing=0.2)
    else:
        ax.set_xticks([30, 40, 50, 60, 70, 80, 90]); ax.set_xlim(28, 91)
        ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS, labelpad=2)
        ax.set_title("IMDb-Wiki, 20 top-importance identities", fontsize=FS, pad=3)
        ax.set_ylabel("MSE", fontsize=FS, labelpad=2)
    ax.tick_params(labelsize=FS, pad=2, length=3)
    ax.grid(True, ls="--", lw=0.6, alpha=0.45); ax.set_axisbelow(True)
fig.tight_layout(pad=0.3, w_pad=1.5)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/rare_group_fit.pdf")
fig.savefig("figures/rare_group_fit.png", dpi=200)
print("wrote figures/rare_group_fit.{pdf,png}")
