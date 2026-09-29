"""Hardware-error propagation model for a transpiled circuit.

Assuming independent, depolarizing-like errors, the probability that the
circuit runs without any fault is

    P_fault-free = prod_g (1 - e_g) * prod_q (1 - r_q) * prod_q exp(-t_idle,q / T2_q)

The measured success probability is then approximately
    P_success ~ P_fault-free * P_ideal + (1 - P_fault-free) * M / 2^n,
where a faulty run returns a uniformly random bitstring and M is the number of solutions.
"""
import math
from collections import Counter


def gate_error_budget(transpiled, backend):
    target = backend.target
    ops = Counter()
    log_ok = 0.0
    per_type = Counter()
    for inst in transpiled.data:
        name = inst.operation.name
        if name in ("barrier", "delay"):
            continue
        qargs = tuple(transpiled.find_bit(q).index for q in inst.qubits)
        ops[name] += 1
        try:
            err = target[name][qargs].error or 0.0
        except KeyError:
            err = 0.0
        if err >= 1.0:
            err = 0.999
        log_ok += math.log1p(-err)
        per_type[name] += -math.log1p(-err)
    return math.exp(log_ok), ops, dict(per_type)


def predicted_success(p_fault_free, p_ideal, n_solutions, n_vars):
    return p_fault_free * p_ideal + (1 - p_fault_free) * n_solutions / 2**n_vars
