"""Grover search for K-SAT with a clause-ancilla phase oracle."""
import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import MCXGate


def phase_oracle(clauses, n_vars):
    """Flips the phase of assignments satisfying every clause.

    One ancilla per clause stores (clause is satisfied); a multi-controlled Z on all
    clause ancillas applies the phase, then the ancillas are uncomputed.
    """
    x = QuantumRegister(n_vars, "x")
    c = QuantumRegister(len(clauses), "clause")
    qc = QuantumCircuit(x, c, name="Oracle")

    def compute():
        for j, clause in enumerate(clauses):
            # clause = OR(l_i) = NOT AND(NOT l_i): flip positive literals so the
            # MCX fires when every literal is false, then invert the ancilla.
            for lit in clause:
                if lit > 0:
                    qc.x(x[lit - 1])
            qc.append(MCXGate(len(clause)), [x[abs(l) - 1] for l in clause] + [c[j]])
            for lit in clause:
                if lit > 0:
                    qc.x(x[lit - 1])
            qc.x(c[j])

    compute()
    if len(clauses) == 1:
        qc.z(c[0])
    else:
        qc.h(c[-1])
        qc.append(MCXGate(len(clauses) - 1), list(c))
        qc.h(c[-1])
    qc.compose(_uncompute(clauses, n_vars), inplace=True)
    return qc


def _uncompute(clauses, n_vars):
    x = QuantumRegister(n_vars, "x")
    c = QuantumRegister(len(clauses), "clause")
    qc = QuantumCircuit(x, c)
    for j, clause in reversed(list(enumerate(clauses))):
        qc.x(c[j])
        for lit in clause:
            if lit > 0:
                qc.x(x[lit - 1])
        qc.append(MCXGate(len(clause)), [x[abs(l) - 1] for l in clause] + [c[j]])
        for lit in clause:
            if lit > 0:
                qc.x(x[lit - 1])
    return qc


def diffuser(n_vars):
    qc = QuantumCircuit(n_vars, name="Diffuser")
    qc.h(range(n_vars))
    qc.x(range(n_vars))
    qc.h(n_vars - 1)
    qc.append(MCXGate(n_vars - 1), list(range(n_vars)))
    qc.h(n_vars - 1)
    qc.x(range(n_vars))
    qc.h(range(n_vars))
    return qc


def optimal_iterations(n_vars, n_solutions):
    if n_solutions == 0:
        return 0
    theta = math.asin(math.sqrt(n_solutions / 2**n_vars))
    return max(1, math.floor(math.pi / (4 * theta)))


def success_probability(n_vars, n_solutions, iterations):
    theta = math.asin(math.sqrt(n_solutions / 2**n_vars))
    return math.sin((2 * iterations + 1) * theta) ** 2


def grover_circuit(clauses, n_vars, iterations):
    x = QuantumRegister(n_vars, "x")
    c = QuantumRegister(len(clauses), "clause")
    out = ClassicalRegister(n_vars, "meas")
    qc = QuantumCircuit(x, c, out)
    qc.h(x)
    oracle, diff = phase_oracle(clauses, n_vars), diffuser(n_vars)
    for _ in range(iterations):
        qc.compose(oracle, qubits=list(x) + list(c), inplace=True)
        qc.compose(diff, qubits=list(x), inplace=True)
    qc.measure(x, out)
    return qc


def marked_state_oracle(n_qubits, marked):
    """Phase oracle for one basis state |marked> (bitstring q_{n-1} ... q_0): X-conjugated MCT with H on the target."""
    qc = QuantumCircuit(n_qubits, name="Oracle")
    zeros = [i for i, b in enumerate(reversed(marked)) if b == "0"]
    if zeros:
        qc.x(zeros)
    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.append(MCXGate(n_qubits - 1), list(range(n_qubits)))
        qc.h(n_qubits - 1)
    if zeros:
        qc.x(zeros)
    return qc


def marked_state_circuit(n_qubits, marked, iterations):
    """Grover search for a single marked state among 2^n: H, (MCT oracle + MCZ diffuser)^iterations, measure."""
    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))
    oracle, diff = marked_state_oracle(n_qubits, marked), diffuser(n_qubits)
    for _ in range(iterations):
        qc.compose(oracle, inplace=True)
        qc.compose(diff, inplace=True)
    qc.measure(range(n_qubits), range(n_qubits))
    return qc
