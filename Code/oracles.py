"""Grover circuits with selectable multi-controlled-gate constructions.

Variants for an m-controlled X (MCT); MCZ = H . MCT . H on the last qubit:
    noancilla   : Qiskit's ancilla-free synthesis (no extra qubits)
    vchain      : V-chain with m-2 clean ancillas and full Toffolis
    vchain_rtof : the same V-chain with relative-phase Toffolis (RCCX, Margolus) everywhere except the
                  middle gate; the relative phases cancel on uncomputation (Maslov 2016)

Oracles for a CNF formula with exactly M solutions:
    A ("marks")  : one X-conjugated MCZ per solution state (needs the solutions, as in the original method)
    B ("clauses"): one ancilla per clause computes "clause satisfied", an MCZ over the clause ancillas
                   marks the assignments satisfying every clause, then the clause ancillas are uncomputed
"""
import itertools
import math
import random

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

VARIANTS = ("noancilla", "vchain", "vchain_rtof")


# ----------------------------------------------------------------------------- multi-controlled gates
def mcx(qc, controls, target, anc, variant):
    m = len(controls)
    if m == 0:
        qc.x(target)
    elif m == 1:
        qc.cx(controls[0], target)
    elif m == 2:
        qc.ccx(controls[0], controls[1], target)
    elif variant == "noancilla":
        qc.mcx(list(controls), target)
    else:
        a = anc[: m - 2]
        if len(a) < m - 2:
            raise ValueError(f"{variant} needs {m - 2} ancillas, got {len(anc)}")
        tof = qc.rccx if variant == "vchain_rtof" else qc.ccx
        chain = [(controls[0], controls[1], a[0])] + [(controls[i + 1], a[i - 1], a[i]) for i in range(1, m - 2)]
        for g in chain:
            tof(*g)
        qc.ccx(controls[m - 1], a[m - 3], target)
        for g in reversed(chain):
            tof(*g)


def mcz(qc, qubits, anc, variant):
    if len(qubits) == 1:
        qc.z(qubits[0])
        return
    qc.h(qubits[-1])
    mcx(qc, qubits[:-1], qubits[-1], anc, variant)
    qc.h(qubits[-1])


def ancillas_needed(n_vars, clauses, oracle, variant):
    if variant == "noancilla":
        return 0
    # an m-qubit MCZ is an (m-1)-controlled X and a w-literal clause a w-controlled X; c controls need c-2
    need = [n_vars - 3] + ([len(c) - 2 for c in clauses] + [len(clauses) - 3] if oracle == "B" else [])
    return max(0, max(need))


# ----------------------------------------------------------------------------- oracles and Grover
def _mark(qc, x, bitstring, anc, variant):
    zeros = [x[i] for i, b in enumerate(reversed(bitstring)) if b == "0"]
    if zeros:
        qc.x(zeros)
    mcz(qc, list(x), anc, variant)
    if zeros:
        qc.x(zeros)


def oracle_a(qc, x, solutions, anc, variant):
    for s in solutions:
        _mark(qc, x, s, anc, variant)


def oracle_b(qc, x, c, clauses, anc, variant):
    def compute(order):
        for j in order:
            clause = clauses[j]
            pos = [x[l - 1] for l in clause if l > 0]
            if pos:
                qc.x(pos)
            mcx(qc, [x[abs(l) - 1] for l in clause], c[j], anc, variant)  # fires when every literal is false
            if pos:
                qc.x(pos)
            qc.x(c[j])  # c_j = clause satisfied

    compute(range(len(clauses)))
    mcz(qc, list(c), anc, variant)
    compute(reversed(range(len(clauses))))


def diffuser(qc, x, anc, variant):
    qc.h(x)
    qc.x(x)
    mcz(qc, list(x), anc, variant)
    qc.x(x)
    qc.h(x)


def grover(n_vars, k, variant="noancilla", solutions=None, clauses=None, oracle="A", measure=True, barriers=False):
    """Grover circuit: H layer, k x (oracle + diffuser). Measures the n variable qubits only."""
    x = QuantumRegister(n_vars, "x")
    regs = [x]
    c = None
    if oracle == "B":
        c = QuantumRegister(len(clauses), "c")
        regs.append(c)
    n_anc = ancillas_needed(n_vars, clauses or [], oracle, variant)
    anc = QuantumRegister(n_anc, "a") if n_anc else []
    if n_anc:
        regs.append(anc)
    qc = QuantumCircuit(*regs, name=f"grover_{oracle}_{variant}")
    qc.h(x)
    for _ in range(k):
        if barriers:
            qc.barrier(label="oracle")
        if oracle == "A":
            oracle_a(qc, x, solutions, list(anc), variant)
        else:
            oracle_b(qc, x, c, clauses, list(anc), variant)
        if barriers:
            qc.barrier(label="diffuser")
        diffuser(qc, x, list(anc), variant)
    if barriers:
        qc.barrier()
    if measure:
        meas = ClassicalRegister(n_vars, "meas")
        qc.add_register(meas)
        qc.measure(x, meas)
    return qc


# ----------------------------------------------------------------------------- theory
def theta(n, m):
    return math.asin(math.sqrt(m / 2**n))


def p_theory(n, m, k):
    return math.sin((2 * k + 1) * theta(n, m)) ** 2


def k_opt(n, m):
    return max(1, math.floor(math.pi / (4 * theta(n, m))))


# ----------------------------------------------------------------------------- CNF instances
def satisfies(clauses, bitstring):
    """bitstring is x_n ... x_1 (Qiskit order)."""
    bits = bitstring[::-1]
    return all(any((bits[abs(l) - 1] == "1") == (l > 0) for l in c) for c in clauses)


def solutions(clauses, n_vars):
    return [b for b in (format(i, f"0{n_vars}b") for i in range(2**n_vars)) if satisfies(clauses, b)]


def find_instance(n_vars, n_clauses, n_solutions=3, widths=(2, 3), seed=0, max_tries=200000):
    """Random CNF over n_vars with n_clauses distinct clauses (widths drawn from `widths`) and exactly
    n_solutions satisfying assignments, every variable used."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        clauses = []
        while len(clauses) < n_clauses:
            w = rng.choice(widths)
            vs = sorted(rng.sample(range(1, n_vars + 1), w))
            cl = [v if rng.random() < 0.5 else -v for v in vs]
            if cl not in clauses:
                clauses.append(cl)
        if {abs(l) for c in clauses for l in c} != set(range(1, n_vars + 1)):
            continue
        if len(solutions(clauses, n_vars)) == n_solutions:
            return clauses
    raise ValueError("no instance found")


def cnf_string(clauses):
    def lit(l):
        return ("¬" if l < 0 else "") + f"x{abs(l)}"

    return " ∧ ".join("(" + " ∨ ".join(lit(l) for l in c) + ")" for c in clauses)


def cnf_latex(clauses):
    def lit(l):
        return (r"\bar{x}" if l < 0 else "x") + f"_{{{abs(l)}}}"

    return r" \land ".join("(" + r" \lor ".join(lit(l) for l in c) + ")" for c in clauses)
