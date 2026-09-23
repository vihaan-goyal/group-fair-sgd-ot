# Paper Fig. 2, single-column variant: the four severity panels stacked (IMDb-Wiki constant / decaying
# stepsize vs infeasible mass nu, then Adult constant / decaying vs skew beta), each pair sharing its x axis.
# Solid = tail-500 F_p mean over 5 seeds (+-1 std), dashed = exact floor Phi of each rule.
# 3.4 in wide for \columnwidth, text at 9.3 pt (ICASSP kit: >= 9 pt as printed), TrueType fonts.
# Usage (repo root): python src/plot_severity_col.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
plt.rcParams["pdf.fonttype"] = 42

FS = 9.3
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
STEP = {"const": r"constant $\eta$", "decay1000": r"decaying $\eta_t$"}
rows = list(csv.DictReader(open("results/tables/severity_sweep_table.csv")))

fig = plt.figure(figsize=(3.4, 5.6))
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 0.42, 1, 1], hspace=0.12,
                      left=0.185, right=0.975, top=0.905, bottom=0.105)
a0 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[3])
axes = [a0, fig.add_subplot(gs[1], sharex=a0, sharey=a0), a2, fig.add_subplot(gs[4], sharex=a2, sharey=a2)]
panels = [("imdbwiki", "const"), ("imdbwiki", "decay1000"), ("adult", "const"), ("adult", "decay1000")]
for k, (ax, (ds, tag)) in enumerate(zip(axes, panels)):
    pts = [r for r in rows if r["dataset"] == ds and r["regime"] == "infeasible" and r["tag"] == tag]
    if ds == "imdbwiki":
        pts.sort(key=lambda r: float(r["infeasible_mass"])); x = [100 * float(r["infeasible_mass"]) for r in pts]
    else:
        pts.sort(key=lambda r: -float(r["beta"])); x = [float(r["beta"]) for r in pts]
    for m in ["full", "fedavg", "fedavot"]:
        y = [float(r[f"meas_{m}"]) for r in pts]; s = [float(r[f"std_{m}"]) for r in pts]
        ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=3, lw=1.3, capsize=1.5, zorder=3)
        if m != "full":
            ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.0, zorder=2)
    ax.text(0.02, 0.95, ("IMDb-Wiki, " if ds == "imdbwiki" else "Adult, ") + STEP[tag], transform=ax.transAxes,
            ha="left", va="top", fontsize=FS)
    ax.set_ylabel(r"$F_p$ (MSE)" if ds == "imdbwiki" else r"$F_p$ (CE)", fontsize=FS, labelpad=2)
    ax.tick_params(labelsize=FS, pad=2, length=2.5)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.margins(y=0.2)
    ax.set_yticks([80, 100, 120] if ds == "imdbwiki" else [0.20, 0.22, 0.24])
    if ds == "adult":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if k in (0, 2):
        ax.tick_params(labelbottom=False)
    if ds == "adult" and k == 3:
        ax.invert_xaxis(); ax.set_xticks(x)
        ax.set_xticklabels([rf"{float(r['beta']):g}" "\n" rf"{float(r['infeasible_mass']):.2f}" for r in pts])
        ax.set_xlabel(r"skew $\beta$ (upper), infeasible mass $\nu$ (lower)", fontsize=FS, labelpad=2)
    if k == 1:
        ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS, labelpad=2)

handles = [Line2D([], [], color=COL[m], marker=MK[m], ms=3.5, lw=1.3, label=LBL[m]) for m in LBL] + \
          [Line2D([], [], color="0.35", lw=1.3, label="measured"),
           Line2D([], [], color="0.35", ls="--", lw=1.0, label=r"floor $\Phi$")]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=FS, frameon=False, bbox_to_anchor=(0.54, 1.0),
           handlelength=1.6, columnspacing=0.8, handletextpad=0.4, borderaxespad=0.1)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/severity_col.pdf")
fig.savefig("figures/severity_col.png", dpi=200)
print("wrote figures/severity_col.{pdf,png}")
