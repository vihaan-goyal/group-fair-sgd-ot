# Paper Figs. 2-4 from the committed tables in results/tables/ (no data, no training):
#   figures/severity_imdb.pdf   Fig. 2  IMDb-Wiki: overall loss vs infeasible mass nu
#   figures/severity_adult.pdf  Fig. 3  Adult: overall loss vs sampling skew beta
# Solid = tail-500 mean over 5 seeds (+-1 std), dashed = exact floor Phi of each rule.
# Text is set at 9.3 pt so it prints at >= 9 pt at \textwidth (ICASSP kit); TrueType fonts.
# Usage (repo root): python src/plot_figures.py
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42

SEV = "results/tables/severity_sweep_table.csv"
RARE = "results/tables/rare_group_table.csv"
OUT = "figures"
FS = 9.3
COL = {"fedavot": "tab:blue", "fedavg": "tab:orange", "full": "tab:red"}
MK = {"fedavot": "o", "fedavg": "s", "full": "^"}
LBL = {"fedavot": "GAVOT", "fedavg": "group-blind avg.", "full": "full"}
PANEL = {"const": r"constant stepsize $\eta$", "decay1000": r"decaying stepsize $\eta_t=\eta/(1+t/1000)$"}
os.makedirs(OUT, exist_ok=True)
sev = list(csv.DictReader(open(SEV)))


def save(fig, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.join(OUT, name + ".{pdf,png}"))


# ---- Fig. 2: IMDb-Wiki severity sweep
rows = [r for r in sev if r["dataset"] == "imdbwiki" and r["regime"] == "infeasible"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.8), sharey=True)
for ax, tag in zip(axes, ["const", "decay1000"]):
    pts = sorted([r for r in rows if r["tag"] == tag], key=lambda r: float(r["infeasible_mass"]))
    x = [100 * float(r["infeasible_mass"]) for r in pts]
    for m in ["fedavot", "fedavg", "full"]:
        y = [float(r[f"meas_{m}"]) for r in pts]; s = [float(r[f"std_{m}"]) for r in pts]
        ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=3, lw=1.3, capsize=2, label=LBL[m])
        if m != "full":
            ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.0)
    for r, xi in zip(pts, x):
        b = float(r["beta"])
        ax.annotate(rf"$\beta$={b:g}", (xi, float(r["meas_fedavot"])), textcoords="offset points",
                    xytext=((0, -11) if b == 2 else (0, 5)), ha=("right" if b == 3 else "center"),
                    fontsize=FS, color="0.3")
    ax.set_title(PANEL[tag], fontsize=FS, pad=6)
    ax.set_xlabel(r"infeasible mass $\nu$ (%)", fontsize=FS)
    ax.tick_params(labelsize=FS); ax.grid(alpha=0.3)
axes[0].set_ylabel(r"overall loss $F_p$ (MSE)", fontsize=FS)
axes[1].legend(fontsize=FS, frameon=False, loc="upper left", handlelength=2.2)
save(fig, "severity_imdb")

# ---- Fig. 3: Adult severity sweep (x = beta; two skews share nu = 0.60 and two share nu = 0)
rows = [r for r in sev if r["dataset"] == "adult" and r["regime"] == "infeasible"]
fig, axes = plt.subplots(1, 2, figsize=(7.0, 1.8), sharey=True)
for ax, tag in zip(axes, ["const", "decay1000"]):
    pts = sorted([r for r in rows if r["tag"] == tag], key=lambda r: -float(r["beta"]))
    x = [float(r["beta"]) for r in pts]
    for m in ["fedavot", "fedavg", "full"]:
        y = [float(r[f"meas_{m}"]) for r in pts]; s = [float(r[f"std_{m}"]) for r in pts]
        ax.errorbar(x, y, yerr=s, color=COL[m], marker=MK[m], ms=3, lw=1.3, capsize=2, label=LBL[m])
        if m != "full":
            ax.plot(x, [float(r[f"floor_{m}"]) for r in pts], color=COL[m], ls="--", lw=1.0)
    ax.invert_xaxis(); ax.set_xticks(x)
    ax.set_xticklabels([f"$\\beta$={float(r['beta']):g}\n$\\nu$={float(r['infeasible_mass']):.2f}" for r in pts],
                       fontsize=FS)
    ax.set_title(PANEL[tag], fontsize=FS, pad=6)
    ax.set_xlabel(r"sampling weights $\propto p^{\beta}$", fontsize=FS)
    ax.tick_params(labelsize=FS); ax.grid(alpha=0.3)
axes[0].set_ylabel(r"overall loss $F_p$ (CE)", fontsize=FS)
axes[1].legend(fontsize=FS, frameon=False, loc="upper left", handlelength=2.2)
save(fig, "severity_adult")
