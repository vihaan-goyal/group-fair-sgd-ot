# Paper Fig. 2, single-column variant: the four severity panels stacked (IMDb-Wiki constant / decaying
# stepsize vs infeasible mass nu, then Adult constant / decaying vs skew beta), each pair sharing its x axis.
# Solid = tail-500 F_p mean over 5 seeds (+-1 std), dashed = exact floor Phi of each rule.
# 3.4 in wide for \columnwidth, 7.5 pt text as printed, TrueType fonts.
# Usage (repo root): python src/plot_severity_col.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
plt.rcParams["pdf.fonttype"] = 42

FS = 7.5
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
STEP = {"const": r"constant $\eta$", "decay1000": r"decaying $\eta_t$"}
rows = list(csv.DictReader(open("results/tables/severity_sweep_table.csv")))

fig = plt.figure(figsize=(3.4, 5.0))
gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 0.2, 1, 1], hspace=0.34,
                      left=0.15, right=0.98, top=0.96, bottom=0.10)
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
        ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=2.5, lw=1.1, capsize=1.2, zorder=3)
        if m != "full":
            ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.0, zorder=2)
    ax.set_title(("IMDb-Wiki, " if ds == "imdbwiki" else "Adult, ") + STEP[tag], fontsize=FS, pad=3)
    ax.set_ylabel(r"$F_p$ (MSE)" if ds == "imdbwiki" else r"$F_p$ (CE)", fontsize=FS, labelpad=2)
    ax.tick_params(labelsize=FS, pad=2, length=3)
    ax.grid(True, ls="--", lw=0.6, alpha=0.45); ax.set_axisbelow(True)
    ax.margins(y=0.15)
    ax.set_yticks([80, 90, 100, 110, 120] if ds == "imdbwiki" else [0.20, 0.21, 0.22, 0.23, 0.24, 0.25])
    if ds == "adult":
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    if k in (0, 2):
        ax.tick_params(labelbottom=False)
    if ds == "adult" and k == 3:
        ax.invert_xaxis(); ax.set_xticks(x)
        ax.set_xticklabels([rf"{float(r['beta']):g}" "\n" rf"{float(r['infeasible_mass']):.2f}" for r in pts])
        ax.set_xlabel(r"skew $\beta$ (upper), infeasible mass $\nu$ (lower)", fontsize=FS, labelpad=2)
    if ds == "imdbwiki":
        ax.set_xticks([30, 40, 50, 60, 70, 80, 90]); ax.set_xlim(28, 91)
    if k == 1:
        ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS, labelpad=2)

# legends inside the panels, in the empty upper-left corners: the rules on IMDb-Wiki, the line styles on Adult
axes[0].set_ylim(76, 128)
axes[1].legend(handles=[Line2D([], [], color=COL[m], marker=MK[m], ms=3, lw=1.1, label=LBL[m]) for m in LBL],
               loc="upper center", ncol=3, fontsize=FS, framealpha=0.9, handlelength=1.4, handletextpad=0.3,
               columnspacing=0.7,
               borderpad=0.3, labelspacing=0.2)
axes[2].legend(handles=[Line2D([], [], color="0.35", lw=1.3, label="measured"),
                        Line2D([], [], color="0.35", ls="--", lw=1.0, label=r"floor $\Phi$")],
               loc="upper left", fontsize=FS, framealpha=0.9, handlelength=1.6, handletextpad=0.4,
               borderpad=0.3, labelspacing=0.2)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/severity_col.pdf")
fig.savefig("figures/severity_col.png", dpi=200)
print("wrote figures/severity_col.{pdf,png}")
