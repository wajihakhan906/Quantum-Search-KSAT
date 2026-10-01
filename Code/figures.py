"""Figures for pipeline.py results (one PNG per pipeline stage)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED = "#0b0b0b", "#52514e"
COLORS = {"theory": MUTED, "qasm": "#2a78d6", "sim": "#eb6834", "hw": "#1baf7a"}
METHOD_LABELS = {"none": "None", "dd": "DD", "trex": "TREX", "twirl": "Twirling", "zne": "ZNE", "all": "All"}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})


def _is_hw(name):
    return not name.startswith("fake_")


def _series(results, sim):
    """[(label, color, rows)] for the device run and, if given, the noisy-simulator run it is compared with."""
    out = []
    if sim is not None:
        out.append((f"Noisy sim ({sim['device']})", COLORS["sim"], sim["rows"]))
    dev = results["device"]
    out.append((f"QPU ({dev})" if _is_hw(dev) else f"Noisy sim ({dev})",
                COLORS["hw"] if _is_hw(dev) else COLORS["sim"], results["rows"]))
    return out


def _bars(ax, xs, groups, width=0.8):
    w = width / len(groups)
    for j, (label, color, ys) in enumerate(groups):
        ax.bar([x + (j - (len(groups) - 1) / 2) * w for x in xs], ys, w * 0.92, color=color, label=label)


def grover_figure(results, tag, sim=None):
    rows = [r for r in results["rows"] if r["step"] == "grover"]
    ns = [r["n"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    groups = [("QASM simulator", COLORS["qasm"], [r["p_success_qasm"] for r in rows])]
    for label, color, rr in _series(results, sim):
        by_n = {r["n"]: r for r in rr if r["step"] == "grover"}
        groups.append((label, color, [by_n[n]["p_success_device"] if n in by_n else 0 for n in ns]))
    _bars(a, ns, groups)
    a.plot(ns, [r["p_theory"] for r in rows], "-", color=COLORS["theory"], lw=2, label="Theory")
    a.plot(ns, [1 / 2**n for n in ns], ":", color=COLORS["theory"], lw=1.5, label="Random guess (1/2ⁿ)")
    a.set_xlabel("Qubits n (search space 2ⁿ)"); a.set_ylabel("P(marked state)"); a.set_ylim(0, 1.05)
    a.set_xticks(ns); a.set_title("Probability of the marked state", loc="left"); a.legend(fontsize=7)

    b.plot(ns, [r["qasm_time_s"] for r in rows], "o-", color=COLORS["qasm"], lw=2, ms=6, label="QASM simulator (wall)")
    for label, color, rr in _series(results, sim):
        by_n = {r["n"]: r for r in rr if r["step"] == "grover"}
        meas = all(by_n[n]["qpu_time_s"] is not None for n in ns if n in by_n)
        key = "qpu_time_s" if meas else "est_qpu_time_s"
        lbl = label.replace("Noisy sim", "Est. QPU time") if not meas else label + " (measured)"
        b.plot([n for n in ns if n in by_n], [by_n[n][key] for n in ns if n in by_n], "s--", color=color, lw=2,
               ms=6, label=lbl)
    b.set_yscale("log"); b.set_xticks(ns); b.set_xlabel("Qubits n"); b.set_ylabel("Search time (s)")
    b.set_title("Computational search time", loc="left"); b.legend(fontsize=7)
    fig.suptitle(f"Grover search, one marked state, optimal iterations — {results['device']}", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / f"steps1-3_grover_{tag}.png", dpi=200); plt.close(fig)


def ksat_figure(results, tag, sim=None):
    rows = [r for r in results["rows"] if r["step"] == "ksat"]
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(len(ks), 2, figsize=(11, 3.6 * len(ks)), gridspec_kw={"width_ratios": [2.4, 1]},
                             squeeze=False)
    for (h, c), k in zip(axes, ks):
        rk = sorted([r for r in rows if r["k"] == k], key=lambda r: r["iterations"])
        r1 = next((r for r in rk if r["iterations"] == 1), rk[-1])
        keys = [format(i, f"0{k}b") for i in range(2**k)]
        groups = [("Ideal", COLORS["qasm"], [r1["exact_dist"].get(b, 0) for b in keys])]
        for label, color, rr in _series(results, sim):
            m = next(x for x in rr if x["id"] == r1["id"])
            groups.append((label, color, [m["device_dist"].get(b, 0) for b in keys]))
        _bars(h, range(len(keys)), groups)
        h.set_xticks(range(len(keys))); h.set_xticklabels(keys, rotation=90, fontsize=5.5)
        for t, b in zip(h.get_xticklabels(), keys):
            if b not in r1["marked"]:
                t.set_color("#e34948"); t.set_weight("bold")
        h.set_ylabel("Probability"); h.legend(fontsize=7)
        h.set_title(f"K={k}: {len(r1['marked'])}/{2**k} assignments satisfy; {r1['iterations']} iteration "
                    f"(non-solutions in red)", loc="left")
        its = [r["iterations"] for r in rk]
        c.plot(its, [r["p_theory"] for r in rk], "-", color=COLORS["theory"], lw=2, label="Theory")
        c.plot(its, [r["p_success_qasm"] for r in rk], "o-", color=COLORS["qasm"], lw=2, ms=6, label="QASM")
        for label, color, rr in _series(results, sim):
            by_id = {x["id"]: x for x in rr}
            c.plot(its, [by_id[r["id"]]["p_success_device"] for r in rk], "s--", color=color, lw=2, ms=6, label=label)
        c.set_xticks(its); c.set_ylim(0, 1.05); c.set_xlabel("Grover iterations"); c.set_ylabel("P(solution)")
        c.set_title(f"{k}-SAT, {k} vars, {len(r1['clauses'])} clauses", loc="left"); c.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / f"steps4-5_ksat_{tag}.png", dpi=200); plt.close(fig)


def mitigation_figure(results, tag, sim=None):
    rows = [r for r in results["rows"] if r["step"] == "mitigation"]
    ks = sorted({r["k"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(2, len(ks), figsize=(5.2 * len(ks), 6.4), squeeze=False)
    for col, k in enumerate(ks):
        rk = {r["method"]: r for r in rows if r["k"] == k}
        for row, (key, ylabel) in enumerate([("p_success_device", "P(solution)"),
                                             ("hellinger_fidelity", "Hellinger fidelity to ideal")]):
            ax = axes[row][col]
            groups = []
            for label, color, rr in _series(results, sim):
                by_id = {x["id"]: x for x in rr}
                groups.append((label, color, [by_id[rk[m]["id"]][key] for m in methods]))
            _bars(ax, range(len(methods)), groups)
            ref = rk[methods[0]]["p_success_qasm"] if row == 0 else 1.0
            ax.axhline(ref, color=COLORS["qasm"], lw=1.5, ls="--",
                       label="QASM simulator" if row == 0 else "Ideal")
            ax.set_xticks(range(len(methods))); ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
            ax.set_ylim(0, 1.05); ax.set_ylabel(ylabel); ax.legend(fontsize=7, loc="lower right")
            if row == 0:
                ax.set_title(f"{k}-SAT ({k} vars, {len(rk[methods[0]]['clauses'])} clauses), "
                             f"{rk[methods[0]]['iterations']} iteration", loc="left")
    fig.suptitle("Error mitigation: DD, TREX, Pauli twirling, ZNE", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(ROOT / "Figures" / f"step6_mitigation_{tag}.png", dpi=200); plt.close(fig)


def make_all(results, tag, sim=None):
    (ROOT / "Figures").mkdir(exist_ok=True)
    grover_figure(results, tag, sim)
    ksat_figure(results, tag, sim)
    mitigation_figure(results, tag, sim)
    print(f"wrote Figures/*_{tag}.png")
