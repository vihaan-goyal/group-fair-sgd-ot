# Paper Fig. 2: both severity sweeps in one 1x4 row, grouped by dataset:
#   IMDb-Wiki (constant | decaying stepsize) vs infeasible mass nu, Adult (constant | decaying) vs beta.
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
STEP = {"const": r"constant $\eta$", "decay1000": r"decaying $\eta_t$"}
rows = list(csv.DictReader(open("results/tables/severity_sweep_table.csv")))

fig = plt.figure(figsize=(7.0, 2.25))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.34, 1, 1], wspace=0.08,
                      left=0.075, right=0.995, bottom=0.22, top=0.70)
a0 = fig.add_subplot(gs[0]); a2 = fig.add_subplot(gs[3])
axes = [a0, fig.add_subplot(gs[1], sharey=a0), a2, fig.add_subplot(gs[4], sharey=a2)]
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
    if ds == "imdbwiki":
        ax.set_xticks([40, 60, 80]); ax.set_xlim(26, 93)
    else:
        ax.invert_xaxis(); ax.set_xticks([1, 0.5, 0]); ax.set_xticklabels(["1", "0.5", "0"]); ax.set_xlim(1.1, -0.1)
    ax.set_title(STEP[tag], fontsize=FS, pad=3)
    ax.tick_params(labelsize=FS, pad=2, length=2.5)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if k in (1, 3):
        ax.tick_params(labelleft=False)
axes[0].set_ylabel(r"$F_p$ (MSE)", fontsize=FS, labelpad=2)
axes[2].set_ylabel(r"$F_p$ (CE)", fontsize=FS, labelpad=2)

# dataset headers and shared x labels, centered over each pair
def pair_center(i, j):
    b0, b1 = axes[i].get_position(), axes[j].get_position()
    return (b0.x0 + b1.x1) / 2, b0.y1, b0.y0
for (i, j), head, xl in [((0, 1), "IMDb-Wiki", r"infeasible mass $\nu$ (%)"),
                         ((2, 3), "Adult", r"skew $\beta$, sampling weights $\propto p^{\beta}$")]:
    xc, top, bot = pair_center(i, j)
    fig.text(xc, top + 0.105, head, ha="center", va="bottom", fontsize=FS, fontweight="bold")
    fig.text(xc, 0.035, xl, ha="center", va="baseline", fontsize=FS)

handles = [Line2D([], [], color=COL[m], marker=MK[m], ms=3.5, lw=1.3, label=LBL[m]) for m in LBL] + \
          [Line2D([], [], color="0.35", lw=1.3, label="measured"),
           Line2D([], [], color="0.35", ls="--", lw=1.0, label=r"floor $\Phi$")]
fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=FS, frameon=False,
           bbox_to_anchor=(0.535, 1.0), handlelength=2.0, columnspacing=1.4, borderaxespad=0.1)
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/severity_1x4.pdf")
fig.savefig("figures/severity_1x4.png", dpi=200)
print("wrote figures/severity_1x4.{pdf,png}")
