"""Grover K-SAT: ideal vs. noisy (IBM Brisbane model or real device) + error-propagation analysis.

    python run_experiments.py                          # FakeBrisbane noise model
    python run_experiments.py --backend ibm_brisbane   # real device (saved IBM Quantum account)
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from qiskit import transpile
from qiskit_aer import AerSimulator

from error_analysis import gate_error_budget, predicted_success
from grover import grover_circuit, optimal_iterations, success_probability
from ksat import random_ksat, solutions, write_dimacs


def get_backend(name):
    if name.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        device = getattr(fake_provider, "Fake" + name[5:].capitalize())()
        return AerSimulator.from_backend(device), device
    from qiskit_ibm_runtime import QiskitRuntimeService

    device = QiskitRuntimeService().backend(name)
    return device, device


def run(backend, circuits, shots):
    if isinstance(backend, AerSimulator):
        res = backend.run(circuits, shots=shots).result()
        return [res.get_counts(i) for i in range(len(circuits))]
    from qiskit_ibm_runtime import SamplerV2

    return [r.data.meas.get_counts() for r in SamplerV2(mode=backend).run(circuits, shots=shots).result()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="fake_brisbane")
    ap.add_argument("--vars", type=int, default=3)
    ap.add_argument("--clauses", type=int, default=3)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--solutions", type=int, default=2, help="resample until the instance has exactly this many")
    ap.add_argument("--max-iter", type=int, default=3)
    ap.add_argument("--shots", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    clauses = random_ksat(args.vars, args.clauses, args.k, seed=args.seed, n_solutions=args.solutions)
    sols = solutions(clauses, args.vars)
    n, m = args.vars, len(sols)
    write_dimacs(Path("../Dataset") / f"instance_v{n}_c{len(clauses)}_s{args.seed}.cnf", clauses, n)
    print("clauses:", clauses, "| solutions:", sols, "| optimal iterations:", optimal_iterations(n, m))

    backend, device = get_backend(args.backend)
    ideal_sim = AerSimulator()
    rows, hist = [], {}
    for it in range(0, args.max_iter + 1):
        qc = grover_circuit(clauses, n, it)
        ideal = ideal_sim.run(transpile(qc, ideal_sim), shots=args.shots).result().get_counts()
        tqc = transpile(qc, device, optimization_level=3, seed_transpiler=11)
        noisy = run(backend, [tqc], args.shots)[0]
        p_ff, ops, per_type = gate_error_budget(tqc, device)
        p_ideal = success_probability(n, m, it)
        rows.append({
            "iterations": it,
            "two_qubit_gates": sum(v for k, v in ops.items() if k in ("ecr", "cx", "cz")),
            "depth": tqc.depth(),
            "p_success_theory": p_ideal,
            "p_success_ideal_sim": sum(ideal.get(s, 0) for s in sols) / args.shots,
            "p_success_noisy": sum(noisy.get(s, 0) for s in sols) / args.shots,
            "p_fault_free": p_ff,
            "p_success_predicted": predicted_success(p_ff, p_ideal, m, n),
            "error_budget": per_type,
        })
        hist[it] = (ideal, noisy)
        r = rows[-1]
        print(f"iter {it}: 2q gates {r['two_qubit_gates']:4d}  ideal {r['p_success_ideal_sim']:.3f}  "
              f"noisy {r['p_success_noisy']:.3f}  predicted {r['p_success_predicted']:.3f}  fault-free {p_ff:.3f}")

    Path("../Results").mkdir(exist_ok=True)
    tag = f"{args.backend}_v{n}_c{len(clauses)}_s{args.seed}"
    Path(f"../Results/{tag}.json").write_text(json.dumps(
        {"backend": args.backend, "clauses": clauses, "solutions": sols, "shots": args.shots, "runs": rows}, indent=2))

    its = [r["iterations"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(its, [r["p_success_theory"] for r in rows], "k-", label="Theory (noise-free)")
    ax.plot(its, [r["p_success_noisy"] for r in rows], "o-", label=f"Noisy ({args.backend})")
    ax.plot(its, [r["p_success_predicted"] for r in rows], "s--", label="Error-propagation model")
    ax.axhline(m / 2**n, color="gray", ls=":", label="Random guess")
    ax.set_xlabel("Grover iterations"); ax.set_ylabel("P(solution)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
    ax.set_title(f"{args.k}-SAT, {n} vars, {len(clauses)} clauses, {m} solution(s)")
    fig.tight_layout(); fig.savefig(f"../Figures/success_{tag}.png", dpi=200)

    best = optimal_iterations(n, m)
    ideal, noisy = hist[best]
    keys = sorted(set(ideal) | set(noisy))
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = range(len(keys))
    ax.bar([x - 0.2 for x in xs], [ideal.get(k, 0) / args.shots for k in keys], 0.4, label="Ideal")
    ax.bar([x + 0.2 for x in xs], [noisy.get(k, 0) / args.shots for k in keys], 0.4, label=args.backend)
    ax.set_xticks(list(xs)); ax.set_xticklabels(keys, rotation=45)
    for i, k in enumerate(keys):
        if k in sols:
            ax.get_xticklabels()[i].set_color("green"); ax.get_xticklabels()[i].set_weight("bold")
    ax.set_ylabel("Probability"); ax.set_title(f"Measurement distribution after {best} iteration(s) (solutions in green)")
    ax.legend(); fig.tight_layout(); fig.savefig(f"../Figures/histogram_{tag}.png", dpi=200)

    grover_circuit(clauses, n, 1).draw("mpl", filename=f"../Figures/circuit_{tag}.png", fold=40)


if __name__ == "__main__":
    main()
