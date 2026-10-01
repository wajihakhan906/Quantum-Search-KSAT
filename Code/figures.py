"""Publication figures and tables for pipeline.py results.

Every figure is written as a vector PDF (for LaTeX) and a 300-dpi PNG, at IEEE/APS column widths
(single 3.5 in, double 7.16 in) with serif (STIX, Times-compatible) fonts. Error bars are binomial
standard errors sqrt(p(1-p)/shots). Tables are written as LaTeX (booktabs) and CSV.
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SINGLE, DOUBLE = 3.5, 7.16
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"theory": MUTED, "qasm": "#2a78d6", "sim": "#eb6834", "hw": "#1baf7a"}
HATCH = {"qasm": "", "sim": "////", "hw": ""}  # second encoding besides colour (greyscale print)
METHOD_LABELS = {"none": "None", "dd": "DD", "trex": "TREX", "twirl": "Twirling", "zne": "ZNE", "all": "All"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.axisbelow": True, "lines.linewidth": 1.4, "lines.markersize": 4,
    "hatch.linewidth": 0.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def _is_hw(name):
    return not name.startswith("fake_")


def _pretty(name):
    return name.removeprefix("fake_")


def _series(results, sim):
    """[(key, label, rows)]: the noisy-simulator run (if given) and the device run."""
    out = []
    if sim is not None:
        out.append(("sim", f"Noisy sim ({_pretty(sim['device'])})", sim["rows"]))
    dev = results["device"]
    out.append(("hw", f"QPU ({dev})", results["rows"]) if _is_hw(dev)
               else ("sim", f"Noisy sim ({_pretty(dev)})", results["rows"]))
    return out


def _se(p, shots):
    return math.sqrt(max(p * (1 - p), 0) / shots) if shots else 0


def _bars(ax, xs, groups, shots, width=0.8):
    w = width / len(groups)
    for j, (key, label, ys) in enumerate(groups):
        pos = [x + (j - (len(groups) - 1) / 2) * w for x in xs]
        err = {"yerr": [_se(y, shots) for y in ys], "error_kw": {"elinewidth": 0.6, "capsize": 1.2, "ecolor": INK}} \
            if shots else {}
        ax.bar(pos, ys, w * 0.9, color=COLORS[key], label=label, hatch=HATCH[key], edgecolor="white",
               linewidth=0.3, **err)


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "Figures" / f"{name}.{ext}", dpi=300)
    plt.close(fig)


def _panel(ax, letter):
    ax.text(-0.13, 1.04, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def grover_figure(results, tag, sim, shots):
    rows = [r for r in results["rows"] if r["step"] == "grover"]
    ns = [r["n"] for r in rows]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(DOUBLE, 2.3), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    groups = [("qasm", "QASM simulator", [r["p_success_qasm"] for r in rows])]
    for key, label, rr in _series(results, sim):
        by_n = {r["n"]: r for r in rr if r["step"] == "grover"}
        groups.append((key, label, [by_n[n]["p_success_device"] if n in by_n else 0 for n in ns]))
    _bars(a, ns, groups, shots)
    a.plot(ns, [r["p_theory"] for r in rows], "-", color=COLORS["theory"], lw=1, label="Theory")
    a.plot(ns, [1 / 2**n for n in ns], ":", color=COLORS["theory"], lw=1, label=r"Random, $2^{-n}$")
    a.set(xlabel=r"Qubits $n$", ylabel="P(marked state)", ylim=(0, 1.08), xticks=ns)
    a.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02), borderaxespad=0, columnspacing=0.8)
    _panel(a, "a")

    b.semilogy(ns, [r["two_qubit_gates"] for r in rows], "o-", color=INK, label="CZ gates")
    b.semilogy(ns, [r["depth"] for r in rows], "s--", color=MUTED, mfc="white", label="Depth")
    b.set(xlabel=r"Qubits $n$", ylabel="Transpiled circuit size", xticks=ns[::2])
    b.legend(loc="upper left"); b.grid(True, which="major", axis="both")
    _panel(b, "b")

    c.semilogy(ns, [r["qasm_time_s"] for r in rows], "o-", color=COLORS["qasm"], label="QASM sim (wall)")
    c.semilogy(ns, [r["est_qpu_time_s"] for r in rows], "s--", color=COLORS["sim"], label="QPU (estimated)")
    if _is_hw(results["device"]) and all(r["qpu_time_s"] is not None for r in rows):
        c.semilogy(ns, [r["qpu_time_s"] for r in rows], "D-", color=COLORS["hw"], label="QPU (measured)")
    c.set(xlabel=r"Qubits $n$", ylabel="Search time (s)", xticks=ns[::2])
    c.legend(loc="center right"); c.grid(True, which="major", axis="both")
    _panel(c, "c")
    fig.tight_layout(w_pad=1.2)
    _save(fig, f"fig_steps1-3_grover_{tag}")


def ksat_figure(results, tag, sim, shots):
    rows = [r for r in results["rows"] if r["step"] == "ksat"]
    ks = sorted({r["k"] for r in rows})
    fig, axes = plt.subplots(len(ks), 2, figsize=(DOUBLE, 2.1 * len(ks)), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    letters = iter("abcdefgh")
    for (h, c), k in zip(axes, ks):
        rk = sorted([r for r in rows if r["k"] == k], key=lambda r: r["iterations"])
        r1 = next((r for r in rk if r["iterations"] == 1), rk[-1])
        keys = [format(i, f"0{k}b") for i in range(2**k)]
        groups = [("qasm", "Ideal", [r1["exact_dist"].get(b, 0) for b in keys])]
        for key, label, rr in _series(results, sim):
            m = next(x for x in rr if x["id"] == r1["id"])
            groups.append((key, label, [m["device_dist"].get(b, 0) for b in keys]))
        _bars(h, range(len(keys)), groups, 0)
        h.set_xticks(range(len(keys)))
        h.set_xticklabels(keys, rotation=90, fontsize=4.5 if k > 5 else 5.5, family="monospace")
        h.set_xlim(-0.8, len(keys) - 0.2)
        for t, b in zip(h.get_xticklabels(), keys):
            if b not in r1["marked"]:
                t.set_color("#e34948"); t.set_weight("bold")
        h.set_ylabel("Probability"); h.set_ylim(0, max(max(g[2]) for g in groups) * 1.22)
        h.legend(loc="upper left", ncol=len(groups))
        h.set_title(f"{k}-SAT, {k} variables, {len(r1['clauses'])} clauses: {len(r1['marked'])}/{2**k} "
                    f"assignments satisfy, {r1['iterations']} iteration (non-solutions in red)", loc="left")
        _panel(h, next(letters))
        its = [r["iterations"] for r in rk]
        c.plot(its, [r["p_theory"] for r in rk], "-", color=COLORS["theory"], label="Theory")
        c.plot(its, [r["p_success_qasm"] for r in rk], "o", color=COLORS["qasm"], label="QASM")
        for key, label, rr in _series(results, sim):
            by_id = {x["id"]: x for x in rr}
            ys = [by_id[r["id"]]["p_success_device"] for r in rk]
            c.errorbar(its, ys, [_se(y, shots) for y in ys], fmt="s--", color=COLORS[key], capsize=1.5,
                       label=label.split(" (")[0])
        c.set(xticks=its, ylim=(0, 1.05), xlabel="Grover iterations", ylabel="P(solution)")
        c.legend(loc="lower left")
        _panel(c, next(letters))
    fig.tight_layout(h_pad=1.0)
    _save(fig, f"fig_steps4-5_ksat_{tag}")


def mitigation_figure(results, tag, sim, shots):
    rows = [r for r in results["rows"] if r["step"] == "mitigation"]
    ks = sorted({r["k"] for r in rows})
    methods = list(dict.fromkeys(r["method"] for r in rows))
    fig, axes = plt.subplots(2, len(ks), figsize=(DOUBLE, 3.9), squeeze=False, sharex=True)
    letters = iter("abcdefgh")
    for row, (key_, ylabel) in enumerate([("p_success_device", "P(solution)"),
                                          ("hellinger_fidelity", "Hellinger fidelity")]):
        for col, k in enumerate(ks):
            ax = axes[row][col]
            rk = {r["method"]: r for r in rows if r["k"] == k}
            groups = []
            for key, label, rr in _series(results, sim):
                by_id = {x["id"]: x for x in rr}
                groups.append((key, label, [by_id[rk[m]["id"]][key_] for m in methods]))
            _bars(ax, range(len(methods)), groups, shots if row == 0 else 0)
            ref = rk[methods[0]]["p_success_exact"] if row == 0 else 1.0
            ax.axhline(ref, color=COLORS["qasm"], lw=1, ls="--", label="Ideal (noise-free)")
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels([METHOD_LABELS[m] for m in methods])
            lo = min(min(g[2]) for g in groups)
            ax.set_ylim(max(0, lo - 0.25) if row == 1 else 0, 1.05)
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(f"{k}-SAT ({k} variables, {len(rk[methods[0]]['clauses'])} clauses), "
                             f"{rk[methods[0]]['iterations']} Grover iteration", loc="left")
            _panel(ax, next(letters))
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.8, rect=(0, 0, 1, 0.95))
    _save(fig, f"fig_step6_mitigation_{tag}")


def tables(results, tag, sim):
    """LaTeX (booktabs) and CSV tables of every experiment."""
    sim_by_id = {r["id"]: r for r in sim["rows"]} if sim else {}
    hw = _is_hw(results["device"])
    cols = ["id", "n", "iterations", "method", "two_qubit_gates", "depth", "p_theory", "p_success_qasm",
            "p_success_device", "hellinger_fidelity", "qasm_time_s", "est_qpu_time_s", "qpu_time_s"]
    with open(ROOT / "Results" / f"table_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + (["p_success_noisy_sim", "hellinger_noisy_sim"] if sim else []))
        for r in results["rows"]:
            extra = [sim_by_id[r["id"]]["p_success_device"], sim_by_id[r["id"]]["hellinger_fidelity"]] \
                if r["id"] in sim_by_id else []
            w.writerow([r.get(c) for c in cols] + extra)

    dev = "QPU" if hw else "Noisy sim"
    lines = [f"% generated by Code/figures.py from Results/pipeline_{tag}.json"]

    def block(caption, label, head, body):
        caption = caption.replace("_", r"\_")
        lines.extend([r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{caption}}}",
                      rf"\label{{{label}}}", r"\begin{tabular}{" + "r" * len(head) + "}", r"\toprule",
                      " & ".join(head) + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}",
                      r"\end{table}", ""])

    def f3(x):
        return "--" if x is None else f"{x:.3f}"

    g = [r for r in results["rows"] if r["step"] == "grover"]
    head = ["$n$", "Iter.", "CZ", "Depth", "Theory", "QASM"] + (["Noisy sim"] if sim else []) + [dev, "Time (s)"]
    body = []
    for r in g:
        t = r["qpu_time_s"] if r["qpu_time_s"] is not None else r["est_qpu_time_s"]
        body.append(" & ".join([str(r["n"]), str(r["iterations"]), str(r["two_qubit_gates"]), str(r["depth"]),
                                f3(r["p_theory"]), f3(r["p_success_qasm"])]
                               + ([f3(sim_by_id[r["id"]]["p_success_device"])] if sim else [])
                               + [f3(r["p_success_device"]), f"{t:.2f}"]) + r" \\")
    block(f"Grover search for one marked state on {results['device']}: probability of the marked state.",
          f"tab:grover_{tag}", head, body)

    kr = [r for r in results["rows"] if r["step"] == "ksat"]
    head = ["$K$", "Iter.", "CZ", "Theory", "QASM"] + (["Noisy sim"] if sim else []) + [dev]
    body = [" & ".join([str(r["k"]), str(r["iterations"]), str(r["two_qubit_gates"]), f3(r["p_theory"]),
                        f3(r["p_success_qasm"])] + ([f3(sim_by_id[r["id"]]["p_success_device"])] if sim else [])
                       + [f3(r["p_success_device"])]) + r" \\" for r in kr]
    block(f"K-SAT with 3 clauses on {results['device']}: probability of a satisfying assignment.",
          f"tab:ksat_{tag}", head, body)

    mr = [r for r in results["rows"] if r["step"] == "mitigation"]
    methods = list(dict.fromkeys(r["method"] for r in mr))
    head = ["$K$", "Metric"] + [METHOD_LABELS[m] for m in methods]
    body = []
    for k in sorted({r["k"] for r in mr}):
        by = {r["method"]: r for r in mr if r["k"] == k}
        for src, name in ([(sim_by_id, "sim")] if sim else []) + [(None, dev)]:
            get = (lambda m, key: src[by[m]["id"]][key]) if src else (lambda m, key: by[m][key])
            body.append(" & ".join([str(k), f"P(sol), {name}"] + [f3(get(m, "p_success_device")) for m in methods])
                        + r" \\")
            body.append(" & ".join([str(k), f"Fidelity, {name}"]
                                   + [f3(get(m, "hellinger_fidelity")) for m in methods]) + r" \\")
    block(f"Error mitigation on {results['device']} (ideal P(sol) at 1 iteration: "
          + ", ".join(f"$K={r['k']}$: {r['p_success_exact']:.3f}" for r in mr if r["method"] == "none") + ").",
          f"tab:mitigation_{tag}", head, body)
    (ROOT / "Results" / f"tables_{tag}.tex").write_text("\n".join(lines))


def make_all(results, tag, sim=None, shots=None):
    shots = shots or results.get("shots", 4096)
    (ROOT / "Figures").mkdir(exist_ok=True)
    grover_figure(results, tag, sim, shots)
    ksat_figure(results, tag, sim, shots)
    mitigation_figure(results, tag, sim, shots)
    tables(results, tag, sim)
    print(f"wrote Figures/fig_*_{tag}.pdf/.png and Results/tables_{tag}.tex, Results/table_{tag}.csv")


if __name__ == "__main__":  # re-plot from saved JSON:  python figures.py ibm_kingston
    import json
    import sys

    tag = sys.argv[1] if len(sys.argv) > 1 else "fake_kingston"
    res = json.loads((ROOT / "Results" / f"pipeline_{tag}.json").read_text())
    twin = ROOT / "Results" / f"pipeline_fake_{tag.removeprefix('ibm_')}.json"
    make_all(res, tag, json.loads(twin.read_text()) if _is_hw(tag) and twin.exists() else None)
