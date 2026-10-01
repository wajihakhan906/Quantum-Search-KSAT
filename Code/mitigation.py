"""Client-side error mitigation for SamplerV2 runs, identical on Aer noise models and IBM QPUs.

Every technique is applied to the transpiled (ISA) circuit before submission and undone in
post-processing, so it works on the IBM Open plan (job mode, no Runtime resilience options needed):

- DD    : X-X dynamical decoupling inserted into idle windows (ALAP schedule from the backend target).
- TREX  : twirled readout error extinction for sampling - random X before each measurement (undone
          classically) makes readout noise symmetric, and per-qubit flip rates measured on twirled
          |0...0> calibration circuits are inverted (tensored inverse).
- Twirl : Pauli twirling of every 2-qubit gate (CZ on Heron), averaging over random equivalents.
- ZNE   : global unitary folding U (U^dag U)^k at noise scales 1, 3, 5 and linear extrapolation
          of every bitstring probability to zero noise.
"""
import math

import numpy as np
from qiskit.circuit import pauli_twirl_2q_gates
from qiskit.circuit.library import XGate
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ALAPScheduleAnalysis, PadDelay, PadDynamicalDecoupling

METHODS = {
    "none": {},
    "dd": {"dd": True},
    "trex": {"trex": True},
    "twirl": {"twirl": True},
    "zne": {"zne": True},
    "all": {"dd": True, "trex": True, "twirl": True, "zne": True},
}
ZNE_SCALES = (1, 3, 5)


def schedule(circuit, target, dd=False):
    """ALAP-schedule and pad idle time with delays (so Aer applies idle relaxation) or with X-X DD."""
    pad = PadDynamicalDecoupling(target=target, dd_sequence=[XGate(), XGate()]) if dd else PadDelay(target=target)
    return PassManager([ALAPScheduleAnalysis(target=target), pad]).run(circuit)


def _split(circuit):
    """Unitary body and trailing measurements of an ISA circuit (delays/barriers dropped)."""
    body, meas = [], []
    for inst in circuit.data:
        name = inst.operation.name
        if name == "measure":
            meas.append(inst)
        elif name not in ("barrier", "delay"):
            if meas:
                raise ValueError("mid-circuit measurement is not supported for folding")
            body.append(inst)
    return body, meas


def _append_inverse(qc, inst):
    op, qargs = inst.operation, inst.qubits
    if op.name == "rz":
        qc.rz(-op.params[0], qargs[0])
    elif op.name == "sx":  # SX^dag = RZ(pi) SX RZ(pi) up to global phase; stays in the Heron basis
        qc.rz(math.pi, qargs[0])
        qc.sx(qargs[0])
        qc.rz(math.pi, qargs[0])
    elif op.name in ("x", "cz", "id", "ecr", "cx"):  # self-inverse (ECR, CX: Eagle/older devices)
        qc.append(op, qargs)
    else:
        qc.append(op.inverse(), qargs)


def fold(circuit, scale):
    """Global folding U (U^dag U)^((scale-1)/2) followed by the original measurements."""
    if scale % 2 != 1:
        raise ValueError("fold scale must be an odd integer")
    body, meas = _split(circuit)
    out = circuit.copy_empty_like()
    for rep in range(scale):
        if rep % 2 == 0:
            for inst in body:
                out.append(inst.operation, inst.qubits)
        else:
            for inst in reversed(body):
                _append_inverse(out, inst)
        out.barrier()
    for inst in meas:
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out


def _measure_twirl(circuit, rng, calibration=False):
    """Random X before each measurement; returns (circuit, flip mask as a bitstring c_{n-1}..c_0)."""
    body, meas = _split(circuit)
    out = circuit.copy_empty_like()
    if not calibration:
        for inst in body:
            out.append(inst.operation, inst.qubits)
    flips = [0] * circuit.num_clbits
    for inst in meas:
        if rng.random() < 0.5:
            out.x(inst.qubits[0])
            flips[circuit.find_bit(inst.clbits[0]).index] = 1
        out.append(inst.operation, inst.qubits, inst.clbits)
    return out, "".join(map(str, reversed(flips)))


def build(tqc, target, method, shots, randomizations=8, seed=0):
    """Circuits (ISA, scheduled) + metadata for one transpiled circuit under one mitigation method.

    Returns a list of (circuit, shots, meta); meta = {"role": "main"|"cal", "scale": s, "mask": bitstring}.
    """
    cfg = METHODS[method]
    rng = np.random.default_rng(seed)
    n_rand = randomizations if (cfg.get("twirl") or cfg.get("trex")) else 1
    per = max(1, shots // n_rand)
    out = []
    for scale in (ZNE_SCALES if cfg.get("zne") else (1,)):
        base = fold(tqc, scale) if scale > 1 else tqc
        if cfg.get("twirl"):
            variants = pauli_twirl_2q_gates(base, num_twirls=n_rand, seed=int(rng.integers(2**31)), target=target)
        else:
            variants = [base] * n_rand
        for v in variants:
            mask = "0" * tqc.num_clbits
            if cfg.get("trex"):
                v, mask = _measure_twirl(v, rng)
            out.append((schedule(v, target, cfg.get("dd", False)), per, {"role": "main", "scale": scale, "mask": mask}))
    if cfg.get("trex"):
        for _ in range(n_rand):
            cal, mask = _measure_twirl(tqc, rng, calibration=True)
            out.append((schedule(cal, target), per, {"role": "cal", "scale": 1, "mask": mask}))
    return out


def _unflip(counts, mask):
    if "1" not in mask:
        return dict(counts)
    m = int(mask, 2)
    w = len(mask)
    out = {}
    for b, c in counts.items():
        k = format(int(b, 2) ^ m, f"0{w}b")
        out[k] = out.get(k, 0) + c
    return out


def _to_vec(counts, n_bits):
    v = np.zeros(2**n_bits)
    for b, c in counts.items():
        v[int(b, 2)] += c
    return v / v.sum()


def _readout_correct(p, flip_rates):
    """Apply the tensored inverse of symmetric per-bit confusion matrices; clip and renormalise."""
    n = len(flip_rates)
    t = p.reshape((2,) * n)
    for i, e in enumerate(flip_rates):  # clbit i is axis n-1-i of the C-ordered tensor
        e = min(e, 0.45)
        inv = np.array([[1 - e, -e], [-e, 1 - e]]) / (1 - 2 * e)
        t = np.moveaxis(np.tensordot(inv, t, axes=([1], [n - 1 - i])), 0, n - 1 - i)
    return _normalise(t.reshape(-1))


def _normalise(p):
    p = np.clip(p, 0, None)
    return p / p.sum()


def combine(counts_list, metas, n_bits, method):
    """Mitigated probability vector (index = int(bitstring, 2)) from the results of `build`."""
    cfg = METHODS[method]
    by_scale, cal = {}, {}
    for counts, meta in zip(counts_list, metas):
        c = _unflip(counts, meta["mask"])
        target = cal if meta["role"] == "cal" else by_scale.setdefault(meta["scale"], {})
        for b, k in c.items():
            target[b] = target.get(b, 0) + k
    probs = {s: _to_vec(c, n_bits) for s, c in by_scale.items()}
    if cfg.get("trex"):
        p_cal = _to_vec(cal, n_bits).reshape((2,) * n_bits)
        flip_rates = [float(np.take(p_cal, 1, axis=n_bits - 1 - i).sum()) for i in range(n_bits)]
        probs = {s: _readout_correct(p, flip_rates) for s, p in probs.items()}
    if len(probs) == 1:
        return probs[1]
    scales = np.array(sorted(probs))
    stacked = np.array([probs[s] for s in scales])
    slope, intercept = np.polyfit(scales, stacked, 1)
    return _normalise(intercept)
