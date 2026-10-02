"""Publication figures for IEEE Transactions on Quantum Engineering (TQE) from the hardware run.

    python tqe_figures.py [results_dir] [out_dir]
        results_dir  default ../Results/hardware_2026-10-01   (the CSV/JSON tree written by the study notebook)
        out_dir      default ../Figures/tqe

Every figure is a vector PDF (the file to submit) plus a 300-dpi PNG preview, at IEEE column widths
(single 3.5 in, double 7.16 in), Times-compatible serif text at 7-8 pt, with fonts embedded (Type 42).

Colour system (Okabe-Ito based, colour-blind safe, also legible in greyscale through markers/hatching):
    devices      ibm_kingston  #0072B2 (blue, circles)    ibm_marrakesh  #D55E00 (vermillion, squares)
    measured     filled markers / solid bars              noise model    hollow markers / dashed lines
    theory       black line                               ideal sim      light grey bars
    techniques   None grey, DD blue, Twirling orange, TREX pink, ZNE green, All purple
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "Results" / "hardware_2026-10-01"
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE.parent / "Figures" / "tqe"

# ----------------------------------------------------------------------------- style
SINGLE, DOUBLE = 3.5, 7.16
INK, INK2, MUTED, GRID = "#141414", "#4a4a4a", "#8a8a8a", "#e6e6e6"
IDEAL_FILL, SOL_BAND = "#d9d9d9", "#e8f4ee"
DEV = {"ibm_kingston": "#0072B2", "ibm_marrakesh": "#D55E00"}
DEV_MARK = {"ibm_kingston": "o", "ibm_marrakesh": "s"}
DEV_NAME = {"ibm_kingston": "ibm_kingston", "ibm_marrakesh": "ibm_marrakesh"}
TECHS = ["No mitigation", "DD", "Twirling", "TREX", "ZNE", "All combined"]
TECH_SHORT = {"No mitigation": "None", "DD": "DD", "Twirling": "Twirling", "TREX": "TREX", "ZNE": "ZNE",
              "All combined": "All"}
TECH_COL = {"No mitigation": "#8a8a8a", "DD": "#0072B2", "Twirling": "#E69F00", "TREX": "#CC79A7",
            "ZNE": "#009E73", "All combined": "#5E3C99"}
TECH_MARK = {"No mitigation": "o", "DD": "s", "Twirling": "^", "TREX": "D", "ZNE": "v", "All combined": "P"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "legend.frameon": False,
    "legend.handlelength": 1.5, "legend.handletextpad": 0.4, "legend.columnspacing": 1.0,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 1.2, "lines.markersize": 4, "errorbar.capsize": 1.6, "hatch.linewidth": 0.45,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.dpi": 150,
})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print("  wrote", name)


def tag(ax, letter, x=-0.16, y=1.0):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom", ha="left")


def head(ax, text):
    ax.set_title(text, loc="left", fontsize=7.6, pad=3, color=INK)


def read(rel):
    with open(RES / rel, newline="") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return x


TEXT_COLS = {"state", "target", "circuit", "config", "backend", "device", "step", "version", "mcx_synthesis",
             "job_id", "note", "mode", "oracle", "recorded_utc", "utc", "two_qubit_gate"}  # bitstrings stay strings


def rows(rel):
    return [{k: (v if k in TEXT_COLS else num(v)) for k, v in r.items()} for r in read(rel)]


def p_theory(n, m, k):
    th = math.asin(math.sqrt(m / 2**n))
    return math.sin((2 * k + 1) * th) ** 2


def dev_legend(include_model=False, include_theory=False, extra=()):
    h = [Line2D([], [], color=DEV[d], marker=DEV_MARK[d], ls="-", ms=4, label=DEV_NAME[d] + " (QPU)") for d in DEV]
    if include_model:
        h += [Line2D([], [], color=MUTED, marker="o", mfc="white", ls="--", ms=4, label="noise model")]
    if include_theory:
        h += [Line2D([], [], color=INK, lw=1, label="theory")]
    return h + list(extra)


# ============================================================================ Fig. 1 compilation
def fig_compilation():
    data = [r for r in rows("step01_circuits/compilation_variants.csv") if r["backend"] == "ibm_kingston"]
    synths = ["default", "noaux_v24", "noaux_hp24", "gray_code"]
    syn_lbl = {"default": "Default", "noaux_v24": "No-aux V24", "noaux_hp24": "No-aux HP24", "gray_code": "Gray code"}
    syn_col = {"default": "#1b3a5c", "noaux_v24": "#3f74a8", "noaux_hp24": "#86b3d9", "gray_code": "#c7dcef"}
    circuits = list(dict.fromkeys(r["circuit"] for r in data))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.25), sharey=True)
    w = 0.2
    for ax, lvl, letter in zip(axes, (1, 3), "ab"):
        for j, s in enumerate(synths):
            ys = [next(r["two_qubit_gates"] for r in data if r["opt_level"] == lvl and r["mcx_synthesis"] == s
                       and r["circuit"] == c) for c in circuits]
            xs = np.arange(len(circuits)) + (j - 1.5) * w
            ax.bar(xs, ys, w * 0.9, color=syn_col[s], edgecolor=INK2 if s == "gray_code" else "none", lw=0.3,
                   label=syn_lbl[s])
            for x, y in zip(xs, ys):
                ax.text(x, y + 8, f"{int(y)}", ha="center", va="bottom", fontsize=5.3, rotation=90, color=INK2)
        ax.set_xticks(range(len(circuits)))
        ax.set_xticklabels([c.replace(", k=1", "").replace("K-SAT 5, oracle ", "5-var K-SAT\nOracle ")
                            .replace("Grover n=5", "Grover\nn = 5") for c in circuits])
        head(ax, f"Optimization level {lvl}")
        tag(ax, letter, -0.12)
    axes[0].set_ylabel("CZ gate count ($k=1$)")
    axes[0].set_ylim(0, 760)
    axes[1].legend(loc="upper left", ncol=2, title="MCX synthesis", title_fontsize=6.8)
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig01_compilation_variants")


# ============================================================================ Fig. 2 simulator baseline
def fig_simulator():
    s = rows("step02_simulator/grover_simulator_summary.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1.55, 1]})
    nn = np.linspace(2, 10, 300)
    a.plot(nn, [p_theory(x, 1, 1) for x in nn], color=INK, lw=0.9, zorder=1)
    a.plot(nn, 2.0 ** -nn, ":", color=MUTED, lw=0.9)
    a.text(9.9, 0.03, r"random $2^{-n}$", ha="right", fontsize=6.5, color=MUTED)
    one = [r for r in s if r["k"] == 1]
    opt = [r for r in s if r["k"] != 1 or r["n"] == 2]
    opt = [r for r in opt if not (r["n"] == 2 and r["k"] == 1)] + [r for r in s if r["n"] == 2]
    opt = sorted({(r["n"], r["k"]): r for r in s if r["k"] == max(x["k"] for x in s if x["n"] == r["n"])}.values(),
                 key=lambda r: r["n"])
    a.plot([r["n"] for r in opt], [r["P_theory"] for r in opt], color=INK, lw=0.9, ls="--")
    for rr, col, mk, lbl in ((one, "#0072B2", "o", "$k=1$"), (opt, "#5E3C99", "s", r"$k^{*}$")):
        x = [r["n"] for r in rr]
        y = [r["P_mean"] for r in rr]
        err = [[r["P_mean"] - r["P_min"] for r in rr], [r["P_max"] - r["P_mean"] for r in rr]]
        a.errorbar(x, y, err, fmt=mk, color=col, mec="white", mew=0.5, ms=4.5, elinewidth=0.8, label=f"Aer, {lbl}")
    a.legend(handles=[*a.get_legend_handles_labels()[0],
                      Line2D([], [], color=INK, lw=0.9, label=r"theory $\sin^2((2k+1)\theta)$")],
             loc="center right")
    a.set(xlabel="Qubits $n$", ylabel="P(marked state)", ylim=(-0.03, 1.06), xticks=range(2, 11))
    tag(a, "a", -0.1)
    for rr, col, mk, lbl in ((one, "#0072B2", "o", "$k=1$"), (opt, "#5E3C99", "s", r"$k^{*}$")):
        b.plot([r["n"] for r in rr], [1e3 * r["time_mean_s"] for r in rr], "-" + mk, color=col, mec="white",
               mew=0.5, ms=4.5, label=lbl)
        for r in rr[-1:]:
            b.annotate(f"$k^*={int(r['k'])}$" if lbl != "$k=1$" else "", (r["n"], 1e3 * r["time_mean_s"]),
                       xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=6.3, color=col)
    b.set(xlabel="Qubits $n$", ylabel="Simulation time per circuit (ms)", xticks=range(2, 11, 2), ylim=(0, None))
    b.legend(loc="upper left")
    tag(b, "b", -0.17)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig02_simulator_baseline")


# ============================================================================ Fig. 3 devices
def fig_devices():
    hw = {r["device"]: r for r in rows("step04_devices/calibration_hardware.csv")}
    snap = {r["device"]: r for r in rows("step04_devices/calibration_local_test.csv")}
    cal = json.loads((RES / "step06_ksat/ksat_counts_hardware.json").read_text())["calibration"]
    fig = plt.figure(figsize=(DOUBLE, 3.4))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.62, wspace=0.55)
    metrics = [("median_2q_error", r"CZ error ($\times10^{-3}$)", 1e3), ("median_readout_error",
               r"Readout error ($\times10^{-2}$)", 1e2), ("median_T1_us", r"$T_1$ ($\mu$s)", 1),
               ("median_T2_us", r"$T_2$ ($\mu$s)", 1)]
    for i, (key, lbl, sc) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        for j, d in enumerate(DEV):
            ax.bar(j - 0.19, hw[d][key] * sc, 0.36, color=DEV[d])
            ax.bar(j + 0.19, snap[d][key] * sc, 0.36, color="white", edgecolor=DEV[d], hatch="//////", lw=0.7)
            ax.text(j - 0.19, hw[d][key] * sc, f"{hw[d][key] * sc:.2f}" if sc > 1 else f"{hw[d][key]:.0f}",
                    ha="center", va="bottom", fontsize=5.6, color=INK2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Kingston", "Marrakesh"])
        ax.set_ylabel(lbl, fontsize=7)
        ax.margins(y=0.18)
        tag(ax, "abcd"[i], -0.42)
    fig.legend(handles=[Patch(color=INK2, label="run day (2026-10-01)"),
                        Patch(facecolor="white", edgecolor=INK2, hatch="//////", label="noise-model snapshot")],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    for i, d in enumerate(DEV):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        c = cal[d]
        q = np.arange(len(c["physical_qubits"]))
        ax.bar(q - 0.19, np.array(c["p1_given0"]) * 100, 0.36, color=DEV[d], label=r"$P(1\,|\,0)$")
        ax.bar(q + 0.19, np.array(c["p0_given1"]) * 100, 0.36, color="white", edgecolor=DEV[d], hatch="//////",
               lw=0.7, label=r"$P(0\,|\,1)$")
        ax.set_xticks(q)
        ax.set_xticklabels([f"Q{p}" for p in c["physical_qubits"]])
        ax.set_ylabel("Readout flip (%)")
        head(ax, f"{d}: readout calibration of the K-SAT qubits")
        ax.legend(loc="upper right")
        tag(ax, "ef"[i], -0.12)
    save(fig, "fig03_device_calibration")


# ============================================================================ Fig. 4/5 mitigation
def mitigation_totals(rel):
    """P(solution) per (backend, n, config, repeat) = sum over the three solution states."""
    tot = defaultdict(float)
    for r in rows(rel):
        tot[(r["backend"], int(r["n"]), r["config"], int(r["repeat"]))] += r["probability"]
    out = defaultdict(list)
    for (b, n, c, _), v in tot.items():
        out[(b, n, c)].append(v)
    return out


def fig_mitigation():
    hw = mitigation_totals("step05_mitigation/mitigation_raw_hardware.csv")
    fake = mitigation_totals("step05_mitigation/mitigation_raw_local_test.csv")
    summ = rows("step05_mitigation/mitigation_summary_hardware.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.55), sharey=False)
    w = 0.38
    for ax, n, letter in zip(axes, (5, 6), "ab"):
        ideal = 3 * next(r["P_ideal"] for r in summ if int(r["n"]) == n)
        for j, d in enumerate(DEV):
            base = np.mean(hw[(d, n, "No mitigation")])
            for i, t in enumerate(TECHS):
                v = hw[(d, n, t)]
                x = i + (j - 0.5) * w
                m = float(np.mean(v))
                ax.bar(x, m, w * 0.88, color=DEV[d], alpha=1.0 if t != "No mitigation" else 0.55, lw=0)
                ax.plot([x] * len(v), v, "_", color=INK, ms=5, mew=0.8)  # individual repetitions
                if (d, n, t) in fake:
                    ax.plot(x, np.mean(fake[(d, n, t)]), marker="D", ms=3.2, mfc="white", mec=INK, mew=0.6, ls="")
                if t != "No mitigation":
                    ax.text(x, ideal * 0.955, f"×{m / base:.2f}", ha="center", va="top", rotation=90,
                            fontsize=5.6, color=DEV[d])
        ax.axhline(ideal, color=INK, lw=0.9)
        ax.text(len(TECHS) - 0.45, ideal, " ideal", va="center", fontsize=6.3)
        ax.text(-0.5, ideal * 0.955, "gain vs none:", ha="left", va="top", fontsize=5.6, color=INK2)
        ax.axhline(3 / 2**n, color=MUTED, lw=0.9, ls=":")
        ax.text(len(TECHS) - 0.45, 3 / 2**n, " random", va="center", fontsize=6.3, color=MUTED)
        ax.set_xticks(range(len(TECHS)))
        ax.set_xticklabels([TECH_SHORT[t] for t in TECHS])
        ax.set_xlim(-0.55, len(TECHS) - 0.45)
        ax.set_ylim(0, ideal * 1.12)
        ax.set_ylabel("P(solution)")
        head(ax, f"{n}-variable K-SAT, Oracle A, $k=1$ (ideal {ideal:.3f})")
        tag(ax, letter, -0.13)
    handles = [Patch(color=DEV[d], label=d) for d in DEV] + [
        Line2D([], [], marker="_", ls="", color=INK, ms=6, mew=0.8, label="single repetition"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=3.2, label="noise-model prediction")]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(w_pad=1.6, rect=(0, 0, 1, 0.95))
    save(fig, "fig04_mitigation")


def fig_mitigation_states():
    s = rows("step05_mitigation/mitigation_summary_hardware.csv")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 3.9))
    w = 0.13
    for (ri, d), (ci, n) in [((i, d), (j, n)) for i, d in enumerate(DEV) for j, n in enumerate((5, 6))]:
        ax = axes[ri][ci]
        rr = [r for r in s if r["backend"] == d and int(r["n"]) == n]
        states = list(dict.fromkeys(r["state"] for r in rr))
        for j, t in enumerate(TECHS):
            for i, st in enumerate(states):
                r = next(x for x in rr if x["config"] == t and x["state"] == st)
                x = i + (j - 2.5) * w
                ax.bar(x, r["P"], w * 0.9, color=TECH_COL[t], yerr=r["sd"], lw=0,
                       error_kw={"elinewidth": 0.6, "capsize": 1, "ecolor": INK})
        ideal = rr[0]["P_ideal"]
        ax.axhline(ideal, color=INK, lw=0.9)
        ax.axhline(1 / 2**n, color=MUTED, lw=0.9, ls=":")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels([f"|{st}⟩" for st in states], family="monospace", fontsize=6.6)
        ax.set_ylim(0, ideal * 1.15)
        ax.set_ylabel("P(state)")
        head(ax, f"{d}, {n}-variable K-SAT")
        tag(ax, "abcd"[2 * ri + ci], -0.13)
    handles = [Patch(color=TECH_COL[t], label=TECH_SHORT[t]) for t in TECHS] + [
        Line2D([], [], color=INK, lw=0.9, label="ideal"), Line2D([], [], color=MUTED, ls=":", lw=0.9, label="random")]
    fig.legend(handles=handles, loc="upper center", ncol=8, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(h_pad=1.2, rect=(0, 0, 1, 0.96))
    save(fig, "figS1_mitigation_per_state")


# ============================================================================ Fig. 6 K-SAT distributions
def fig_ksat(ks, name):
    dist = rows("step06_ksat/ksat_distributions_hardware.csv")
    ideal = rows("step02_simulator/ksat_ideal_distributions.csv")
    sols = {5: ["00000", "01010", "11111"], 6: ["000000", "101010", "111111"]}
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 3.9))
    for ax, n, letter in zip(axes, (5, 6), "ab"):
        k = ks[n]
        keys = [format(i, f"0{n}b") for i in range(2**n)]
        xs = np.arange(len(keys))
        for x, b in zip(xs, keys):
            if b in sols[n]:
                ax.axvspan(x - 0.5, x + 0.5, color=SOL_BAND, lw=0, zorder=0)
        idl = {r["state"]: r["probability"] for r in ideal if int(r["n"]) == n and r["oracle"] == "A"
               and int(r["k"]) == k}
        ax.bar(xs, [idl.get(b, 0) for b in keys], 0.8, color=IDEAL_FILL, label="ideal (statevector)", zorder=1)
        for j, d in enumerate(DEV):
            per = defaultdict(list)
            for r in dist:
                if r["backend"] == d and int(r["n"]) == n and int(r["k"]) == k:
                    per[r["state"]].append(r["probability"])
            reps = len({int(r["repeat"]) for r in dist if r["backend"] == d and int(r["n"]) == n
                        and int(r["k"]) == k})
            full = {b: per[b] + [0.0] * (reps - len(per[b])) for b in keys}  # unobserved states have P = 0
            m = [np.mean(full[b]) for b in keys]
            sd = [np.std(full[b], ddof=1) for b in keys]
            off = (j - 0.5) * 0.3
            ax.errorbar(xs + off, m, sd, fmt=DEV_MARK[d], color=DEV[d], ms=2.6 if n == 6 else 3.2, mew=0,
                        elinewidth=0.6, capsize=0, label=f"{d} (mean of 3)", zorder=3)
        ax.axhline(1 / 2**n, color=MUTED, lw=0.8, ls=":", zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, rotation=90, fontsize=4.6 if n == 6 else 5.6, family="monospace")
        for t, b in zip(ax.get_xticklabels(), keys):
            if b in sols[n]:
                t.set_color("#00734f"); t.set_fontweight("bold")
        ax.set_xlim(-0.7, len(keys) - 0.3)
        ax.set_ylabel("Probability")
        p_id = 3 * idl.get(sols[n][0], 0)
        head(ax, f"{n}-variable K-SAT, $k={k}$: ideal P(solution) = {p_id:.3f}")
        tag(ax, letter, -0.07)
    h, l = axes[0].get_legend_handles_labels()
    h.append(Patch(color=SOL_BAND, label="solution states")); l.append("solution states")
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(h_pad=0.6, rect=(0, 0, 1, 0.965))
    save(fig, name)


# ============================================================================ Fig. 7 iteration sweep
def fig_sweep():
    hw = rows("step07_iteration_sweep/sweep_summary_hardware.csv")
    fake = [r for r in rows("step03_fake_prediction/fake_prediction.csv") if r["step"] == "step07"]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), sharey=True)
    for ax, n, letter in zip(axes, (3, 4), "ab"):
        kk = np.linspace(0, 3.25, 300)
        ax.plot(kk, [p_theory(n, 1, k) for k in kk], color=INK, lw=0.9, alpha=0.9)
        ax.plot(range(4), [p_theory(n, 1, k) for k in range(4)], "o", color=INK, ms=3.5)
        for j, d in enumerate(DEV):
            rr = sorted([r for r in hw if r["backend"] == d and int(r["n"]) == n], key=lambda r: r["k"])
            ff = sorted([r for r in fake if r["backend"] == d and int(r["n"]) == n], key=lambda r: r["k"])
            off = (j - 0.5) * 0.08
            ax.plot([r["k"] + off for r in ff], [r["probability"] for r in ff], DEV_MARK[d] + "--", color=DEV[d],
                    mfc="white", mew=0.8, ms=4, lw=0.8, alpha=0.75)
            ax.errorbar([r["k"] + off for r in rr], [r["P"] for r in rr], [r["sd"] for r in rr],
                        fmt=DEV_MARK[d] + "-", color=DEV[d], ms=4.2, mec="white", mew=0.4)
        ax.axhline(2**-n, color=MUTED, lw=0.8, ls=":")
        kopt = max(1, math.floor(math.pi / (4 * math.asin(2 ** (-n / 2)))))
        ax.axvline(kopt, color=MUTED, lw=0.6, ls="-.", zorder=0)
        ax.text(kopt, 0.02, f" $k^*={kopt}$", fontsize=6.3, color=MUTED)
        ax.set(xlabel="Grover iterations $k$", xticks=range(4), ylim=(0, 1.08))
        head(ax, f"$n={n}$, one marked state")
        tag(ax, letter, -0.14)
    axes[0].set_ylabel("P(marked state)")
    fig.legend(handles=dev_legend(True, True), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.94))
    save(fig, "fig06_iteration_sweep")


# ============================================================================ Fig. 8 scaling + noise model
def fig_scaling():
    hw = rows("step08_scaling/scaling_summary_hardware.csv")
    sweep = rows("step07_iteration_sweep/sweep_summary_hardware.csv")
    fake = [r for r in rows("step03_fake_prediction/fake_prediction.csv") if r["step"] == "step08"]
    fit = {r["backend"]: r for r in rows("step10_scaling_model/scaling_fit_hardware.csv")}
    fit_fake = {r["backend"]: r for r in rows("step10_scaling_model/scaling_fit_local_test.csv")}
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.5))
    ns = list(range(2, 8))
    a.plot(ns, [p_theory(n, 1, 1) for n in ns], "-", color=INK, lw=0.9, marker="o", ms=3, label="theory")
    a.plot(ns, [2**-n for n in ns], ":", color=MUTED, lw=0.9)
    a.text(2.06, 0.27, r"random $2^{-n}$", ha="left", fontsize=6.3, color=MUTED)
    for j, d in enumerate(DEV):
        rr = sorted([r for r in hw if r["backend"] == d and int(r["k"]) == 1], key=lambda r: r["n"])
        off = (j - 0.5) * 0.1
        fm = [np.mean([r["probability"] for r in fake if r["backend"] == d and int(r["n"]) == n and int(r["k"]) == 1])
              for n in ns]
        a.plot(np.array(ns) + off, fm, DEV_MARK[d] + "--", color=DEV[d], mfc="white", mew=0.8, ms=4, lw=0.8,
               alpha=0.75)
        a.errorbar([r["n"] + off for r in rr], [r["P_mean"] for r in rr],
                   [[r["P_mean"] - r["P_min"] for r in rr], [r["P_max"] - r["P_mean"] for r in rr]],
                   fmt=DEV_MARK[d] + "-", color=DEV[d], ms=4.2, mec="white", mew=0.4)
    a.set(xlabel="Qubits $n$", ylabel="P(marked state), $k=1$", ylim=(0, 1.05), xticks=ns)
    a.legend(handles=dev_legend(True, True), loc="upper right")
    tag(a, "a", -0.14)
    gg = np.logspace(0, 3.5, 300)
    for d in DEV:
        pts = [(r["g"], r["P_mean"], r["P_theory"], r["n"]) for r in hw if r["backend"] == d]
        pts += [(r["g"], r["P"], r["P_theory"], r["n"]) for r in sweep if r["backend"] == d and r["k"] > 0]
        pts = [(g, (p - 2**-n) / (pt - 2**-n)) for g, p, pt, n in pts if pt - 2**-n > 0.05]
        b.plot([p[0] for p in pts], [p[1] for p in pts], DEV_MARK[d], color=DEV[d], ms=4, mec="white", mew=0.4,
               ls="")
        F, Fs = fit[d]["F_per_2q_gate"], fit[d]["F_std"]
        b.plot(gg, F**gg, "-", color=DEV[d], lw=1.1, label=f"{d.removeprefix('ibm_').capitalize()} fit, "
               f"$F$ = {F:.4f} ± {Fs:.4f}")
        b.fill_between(gg, (F - Fs) ** gg, np.minimum(1, (F + Fs) ** gg), color=DEV[d], alpha=0.12, lw=0)
        Ff = fit_fake[d]["F_per_2q_gate"]
        b.plot(gg, Ff**gg, "--", color=DEV[d], lw=0.8, alpha=0.6)
        g10 = fit[d]["g_10pct"]
        b.plot([g10, g10], [0, 0.1], color=DEV[d], lw=0.7)
        b.text(g10, -0.045, f"{g10:.0f}", ha="center", va="top", fontsize=6, color=DEV[d])
    b.axhline(0.1, color=MUTED, lw=0.7, ls=":")
    b.text(1.2, 0.115, "10% of ideal signal", fontsize=6.2, color=MUTED)
    b.set_xscale("log")
    b.set(xlabel="CZ gates $g$", ylabel=r"Normalised signal $\frac{P-2^{-n}}{P_\mathrm{ideal}-2^{-n}}$",
          xlim=(1, 3e3), ylim=(-0.12, 1.1))
    b.grid(True, which="major", axis="both")
    b.legend(handles=[*b.get_legend_handles_labels()[0],
                      Line2D([], [], color=MUTED, ls="--", lw=0.8, label="noise-model fit")], loc="upper center",
             bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.4)
    tag(b, "b", -0.16)
    fig.tight_layout(w_pad=1.6)
    save(fig, "fig07_scaling_and_noise_model")


# ============================================================================ Fig. 9 model vs hardware
def fig_fake_vs_real():
    data = rows("step09_fake_vs_real/fake_vs_real_hardware.csv")
    stats = {r["backend"]: r for r in rows("step09_fake_vs_real/fake_vs_real_stats_hardware.csv")}
    step_mark = {"step06": "D", "step07": "^", "step08": "o"}
    step_lbl = {"step06": "K-SAT (Step 6)", "step07": "sweep (Step 7)", "step08": "scaling (Step 8)"}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 3.25), sharey=True)
    for ax, d, letter in zip(axes, DEV, "ab"):
        ax.fill_between([0, 1], [-0.05, 0.95], [0.05, 1.05], color=GRID, lw=0, zorder=0)
        ax.plot([0, 1], [0, 1], color=INK2, lw=0.7, zorder=1)
        for st, mk in step_mark.items():
            rr = [r for r in data if r["backend"] == d and r["step"] == st]
            ax.plot([r["P_fake"] for r in rr], [r["P_real"] for r in rr], mk, color=DEV[d], ms=4.4, mec="white",
                    mew=0.4, ls="", alpha=0.95)
        s = stats[d]
        ax.text(0.04, 0.95, f"MAE = {s['mean_abs_difference']:.3f}\nbias = {s['mean_difference']:+.3f}\n"
                            f"$N$ = {int(s['points'])}", transform=ax.transAxes, va="top", fontsize=6.8,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": GRID, "lw": 0.6})
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", xlabel="Noise-model prediction P")
        ax.grid(True, axis="both")
        head(ax, d)
        tag(ax, letter, -0.14)
    axes[0].set_ylabel("Hardware P")
    handles = [Line2D([], [], marker=m, ls="", color=INK2, ms=4.4, label=step_lbl[s]) for s, m in step_mark.items()]
    handles.append(Patch(color=GRID, label="±0.05"))
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=1.5, rect=(0, 0, 1, 0.95))
    save(fig, "fig08_model_vs_hardware")


# ============================================================================ Fig. 10 time to solution
def fig_tts():
    s = rows("step11_time_to_solution/time_to_solution_summary_hardware.csv")
    cases = [(5, 1), (5, 2), (6, 1), (6, 3)]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for i, (n, k) in enumerate(cases):
        rnd = 2**n / 3
        ideal = next(r["ideal_shots"] for r in s if int(r["n"]) == n and int(r["k"]) == k)
        ax.plot([i - 0.42, i + 0.42], [rnd, rnd], ":", color=MUTED, lw=1)
        ax.plot([i - 0.42, i + 0.42], [ideal, ideal], "-", color=INK, lw=1)
        for j, d in enumerate(DEV):
            for v, fill, dx in (("No mitigation", False, -0.07), ("Readout corr.", True, 0.07)):
                r = next(x for x in s if x["backend"] == d and int(x["n"]) == n and int(x["k"]) == k
                         and x["version"] == v)
                x = i + (j - 0.5) * 0.36 + dx
                err = r["sd"] / r["P_valid"] ** 2
                ax.errorbar(x, r["shots_to_solution"], err, fmt=DEV_MARK[d], color=DEV[d], ms=4.3,
                            mfc=DEV[d] if fill else "white", mew=0.8, elinewidth=0.6)
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([f"$n={n}$\n$k={k}$" for n, k in cases])
    ax.set_ylabel("Expected shots to a solution, $1/P$")
    ax.set_ylim(0, 25)
    handles = [Line2D([], [], marker=DEV_MARK[d], color=DEV[d], ls="", ms=4.3, label=d) for d in DEV] + [
        Line2D([], [], marker="o", color=INK2, mfc="white", ls="", ms=4.3, label="raw counts"),
        Line2D([], [], marker="o", color=INK2, ls="", ms=4.3, label="readout-corrected"),
        Line2D([], [], color=INK, lw=1, label="ideal Grover"),
        Line2D([], [], color=MUTED, ls=":", lw=1, label=r"random, $2^n/3$")]
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=6.3)
    fig.tight_layout()
    save(fig, "fig09_time_to_solution")


# ============================================================================ Fig. 11 cost-benefit
def fig_cost_benefit():
    data = rows("step12_summary/cost_benefit_hardware.csv")
    fake = {(r["backend"], r["config"]): r for r in rows("step12_summary/cost_benefit_local_test.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), sharey=True)
    for ax, d, letter in zip(axes, DEV, "ab"):
        ax.axhline(1, color=INK2, lw=0.7)
        pts = [r for r in data if r["backend"] == d]
        for r in pts:
            t = r["config"]
            f = fake.get((d, t))
            if f:
                ax.plot([f["relative_cost"], r["relative_cost"]], [f["mean_gain"], r["mean_gain"]], color=TECH_COL[t],
                        lw=0.5, alpha=0.5, zorder=1)
                ax.plot(f["relative_cost"], f["mean_gain"], TECH_MARK[t], ms=4, mfc="white", mec=TECH_COL[t], mew=0.8,
                        zorder=2)
            ax.plot(r["relative_cost"], r["mean_gain"], TECH_MARK[t], ms=6, color=TECH_COL[t], mec="white", mew=0.5,
                    zorder=3)
        # direct labels, spread vertically where points share x
        lab = sorted(pts, key=lambda r: (round(r["relative_cost"]), r["mean_gain"]))
        last = {}
        for r in lab:
            key = round(r["relative_cost"])
            y = r["mean_gain"]
            if key in last and y - last[key] < 0.045:
                y = last[key] + 0.045
            last[key] = y
            ax.annotate(TECH_SHORT[r["config"]], (r["relative_cost"], r["mean_gain"]), xytext=(r["relative_cost"] + 0.2, y),
                        textcoords="data", fontsize=6.4, va="center", color=INK2)
        ax.set(xlabel="Relative QPU cost (no mitigation = 1)", xlim=(0.75, 5.6), xticks=range(1, 6))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.grid(True, axis="both")
        head(ax, d)
        tag(ax, letter, -0.14)
    axes[0].set_ylabel("Mean gain in P(state) (×)")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    handles = [Line2D([], [], marker="o", ls="", color=INK2, ms=5.5, label="QPU"),
               Line2D([], [], marker="o", ls="", color=INK2, mfc="white", ms=4, label="noise model")]
    axes[1].legend(handles=handles, loc="upper right")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig10_cost_benefit")


# ============================================================================ Fig. 12 bootstrap + fidelity
def fig_bootstrap():
    data = rows("step12_summary/bootstrap_and_fidelity_hardware.csv")
    cases = [(5, 1), (5, 2), (6, 1), (6, 3)]
    fig, (a, b) = plt.subplots(1, 2, figsize=(DOUBLE, 2.45), gridspec_kw={"width_ratios": [1.25, 1]})
    for i, (n, k) in enumerate(cases):
        ideal = p_theory(n, 3, k)
        a.plot([i - 0.42, i + 0.42], [ideal, ideal], color=INK, lw=1)
        a.plot([i - 0.42, i + 0.42], [3 / 2**n] * 2, ":", color=MUTED, lw=1)
        for j, d in enumerate(DEV):
            rr = sorted([r for r in data if r["backend"] == d and int(r["n"]) == n and int(r["k"]) == k],
                        key=lambda r: r["repeat"])
            for m, r in enumerate(rr):
                x = i + (j - 0.5) * 0.4 + (m - 1) * 0.08
                a.errorbar(x, r["P_solutions"], [[r["P_solutions"] - r["ci_low"]], [r["ci_high"] - r["P_solutions"]]],
                           fmt=DEV_MARK[d], color=DEV[d], ms=3.4, mec="white", mew=0.3, elinewidth=0.7, capsize=0)
            hf = [r["hellinger_fidelity"] for r in rr]
            x = i + (j - 0.5) * 0.38
            b.bar(x, np.mean(hf), 0.34, color=DEV[d], yerr=np.std(hf, ddof=1),
                  error_kw={"elinewidth": 0.6, "capsize": 1.5, "ecolor": INK})
            b.text(x, np.mean(hf) + 0.03, f"{np.mean(hf):.2f}", ha="center", fontsize=5.8, color=INK2)
    for ax in (a, b):
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([f"$n={n}$, $k={k}$" for n, k in cases])
    a.set(ylabel="P(solution) with 95% bootstrap CI", ylim=(0, 1.05))
    a.legend(handles=[Line2D([], [], marker=DEV_MARK[d], color=DEV[d], ls="", label=d) for d in DEV] +
             [Line2D([], [], color=INK, lw=1, label="ideal"), Line2D([], [], color=MUTED, ls=":", label="random")],
             loc="center right", ncol=2)
    b.set(ylabel="Hellinger fidelity to ideal", ylim=(0, 1.05))
    tag(a, "a", -0.13)
    tag(b, "b", -0.16)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig11_bootstrap_fidelity")


# ============================================================================ Fig. 13 QPU usage
def fig_usage():
    data = rows("step12_summary/qpu_usage_hardware.csv")
    steps = ["step05", "step06", "step07", "step08", "readout_cal"]
    lbl = {"step05": "Step 5 mitigation", "step06": "Step 6 K-SAT", "step07": "Step 7 sweep",
           "step08": "Step 8 scaling", "readout_cal": "readout cal."}
    col = {"step05": "#1b3a5c", "step06": "#3f74a8", "step07": "#86b3d9", "step08": "#c7dcef", "readout_cal": "#e9e9e9"}
    fig, ax = plt.subplots(figsize=(SINGLE, 1.55))
    for i, d in enumerate(DEV):
        left = 0
        for s in steps:
            v = sum(r["qpu_seconds"] for r in data if r["backend"] == d and r["step"] == s)
            ax.barh(i, v, 0.55, left=left, color=col[s], edgecolor="white", lw=0.5, label=lbl[s] if i == 0 else None)
            left += v
        ax.text(left + 4, i, f"{left:.0f} s", va="center", fontsize=6.6)
    total = sum(r["qpu_seconds"] for r in data)
    ax.set_yticks(range(len(DEV)))
    ax.set_yticklabels(list(DEV))
    ax.invert_yaxis()
    ax.set_xlabel(f"QPU time (s); total {total:.0f} s of the 600 s monthly allowance")
    ax.set_xlim(0, 200)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.2)
    save(fig, "fig12_qpu_usage")


def main():
    print(f"reading {RES}\nwriting {OUT}")
    fig_compilation()
    fig_simulator()
    fig_devices()
    fig_mitigation()
    fig_mitigation_states()
    fig_ksat({5: 1, 6: 1}, "fig05_ksat_distributions_k1")
    fig_ksat({5: 2, 6: 3}, "figS2_ksat_distributions_kopt")
    fig_sweep()
    fig_scaling()
    fig_fake_vs_real()
    fig_tts()
    fig_cost_benefit()
    fig_bootstrap()
    fig_usage()


if __name__ == "__main__":
    main()
