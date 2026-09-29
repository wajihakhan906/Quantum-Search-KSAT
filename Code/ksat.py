"""K-SAT instances in CNF: literals are signed 1-based integers (DIMACS style)."""
import itertools
import random


def random_ksat(n_vars, n_clauses, k=3, seed=None, satisfiable=True, n_solutions=None, max_tries=20000):
    """Random K-SAT instance; with `n_solutions`, resample until it has exactly that many."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        clauses = []
        for _ in range(n_clauses):
            vars_ = rng.sample(range(1, n_vars + 1), k)
            clauses.append([v if rng.random() < 0.5 else -v for v in vars_])
        sols = solutions(clauses, n_vars)
        if n_solutions is not None and len(sols) != n_solutions:
            continue
        if not satisfiable or sols:
            return clauses
    raise ValueError(f"no {k}-SAT instance with {n_vars} vars, {n_clauses} clauses and "
                     f"{n_solutions} solutions found in {max_tries} tries; change --clauses or --solutions")


def evaluate(clauses, assignment):
    """assignment: tuple of bools, index 0 -> variable 1."""
    return all(any(assignment[abs(l) - 1] == (l > 0) for l in c) for c in clauses)


def solutions(clauses, n_vars):
    """All satisfying assignments as bitstrings x_n ... x_1 (Qiskit ordering)."""
    sols = []
    for bits in itertools.product([False, True], repeat=n_vars):
        if evaluate(clauses, bits):
            sols.append("".join("1" if b else "0" for b in reversed(bits)))
    return sols


def read_dimacs(path):
    clauses, n_vars = [], 0
    for line in open(path):
        line = line.strip()
        if not line or line[0] in "c%":
            continue
        if line.startswith("p"):
            n_vars = int(line.split()[2])
            continue
        lits = [int(x) for x in line.split() if x != "0"]
        if lits:
            clauses.append(lits)
    return clauses, n_vars


def write_dimacs(path, clauses, n_vars):
    with open(path, "w") as f:
        f.write(f"p cnf {n_vars} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")
