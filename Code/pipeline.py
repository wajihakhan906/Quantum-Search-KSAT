"""Six-step Grover / K-SAT pipeline on the QASM simulator, IBM Heron noise models and IBM Heron QPUs.

Step 1-3  Grover search for one marked state among 2^n, n = 2..10 (H + MCT oracle, MCZ diffuser):
          P(marked) and search time on the QASM simulator and on the QPU.
Step 4-5  K-SAT, K = 5 and 6, 3 clauses, 5 and 6 variable qubits (+1 ancilla per clause):
          P(solution) and search time on the QASM simulator and on the QPU.
Step 6    K-SAT with and without error mitigation (none, DD, TREX, twirling, ZNE, all combined)
          on the noisy simulator and on the QPU.

    python pipeline.py simulate                          # QASM + FakeKingston noise model (no account needed)
    python pipeline.py estimate --backend ibm_kingston   # QPU-time estimate vs. the Open plan's 10 min
    python pipeline.py submit   --backend ibm_kingston   # 3 jobs (one per hardware step), job IDs saved
    python pipeline.py collect  --backend ibm_kingston   # fetch results, write Results/ and Figures/
    python pipeline.py wait     --backend ibm_kingston   # poll the queue until the jobs finish, then collect

The IBM account is read from --token/--instance, the QISKIT_IBM_TOKEN/QISKIT_IBM_INSTANCE environment
variables, or a saved account (QiskitRuntimeService.save_account). --backend auto picks the least busy of
ibm_kingston, ibm_fez and ibm_marrakesh.
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector, hellinger_fidelity
from qiskit_aer import AerSimulator

import mitigation
from grover import grover_circuit, marked_state_circuit, optimal_iterations, success_probability
from ksat import random_ksat, solutions, write_dimacs

ROOT = Path(__file__).resolve().parent.parent
OPEN_PLAN_QPUS = ("ibm_kingston", "ibm_fez", "ibm_marrakesh")
STEPS = {"grover": "Steps 1-3", "ksat": "Steps 4-5", "mitigation": "Step 6"}
SIM_DEVICE = "CPU"  # set by --device; "GPU" needs qiskit-aer-gpu-cu11 (Linux / WSL2)


def aer_options():
    return {"device": "GPU", "batched_shots_gpu": True} if SIM_DEVICE == "GPU" else {}


def pick_device(requested):
    available = AerSimulator().available_devices()
    if requested == "auto":
        return "GPU" if "GPU" in available else "CPU"
    if requested == "GPU" and "GPU" not in available:
        raise SystemExit("Aer reports no GPU; install qiskit-aer-gpu-cu11 on Linux/WSL2 (see SETUP.md) or use --device CPU")
    return requested


# ----------------------------------------------------------------------------- backends
def service(args):
    from qiskit_ibm_runtime import QiskitRuntimeService

    token = args.token or os.environ.get("QISKIT_IBM_TOKEN")
    instance = args.instance or os.environ.get("QISKIT_IBM_INSTANCE")
    if token:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=token, instance=instance)
    return QiskitRuntimeService(instance=instance) if instance else QiskitRuntimeService()


def get_device(args):
    if args.backend.startswith("fake_"):
        from qiskit_ibm_runtime import fake_provider

        return getattr(fake_provider, "Fake" + args.backend[5:].capitalize())()
    svc = service(args)
    if args.backend == "auto":
        return svc.least_busy(operational=True, simulator=False,
                              filters=lambda b: b.name in OPEN_PLAN_QPUS)
    return svc.backend(args.backend)


def rep_delay(device):
    try:
        return float(device.configuration().default_rep_delay)
    except Exception:
        return 250e-6


# ----------------------------------------------------------------------------- experiments
def experiments(args):
    """Logical circuits + what counts as success, for the three hardware steps."""
    rng = np.random.default_rng(args.seed)
    exps = []
    for n in range(args.n_min, args.n_max + 1):
        marked = "".join(rng.choice(["0", "1"], size=n))
        it = optimal_iterations(n, 1)
        exps.append({"id": f"grover_n{n}", "step": "grover", "n": n, "marked": [marked], "iterations": it,
                     "method": "none", "p_theory": success_probability(n, 1, it),
                     "circuit": marked_state_circuit(n, marked, it)})
    for k in args.ksat_k:
        clauses = random_ksat(k, args.ksat_clauses, k, seed=args.seed + k)
        sols = solutions(clauses, k)
        write_dimacs(ROOT / "Dataset" / f"instance_k{k}_v{k}_c{len(clauses)}_s{args.seed + k}.cnf", clauses, k)
        common = {"n": k, "k": k, "clauses": clauses, "marked": sols}
        for it in args.ksat_iterations:
            exps.append({"id": f"ksat_k{k}_it{it}", "step": "ksat", "iterations": it, "method": "none",
                         "p_theory": success_probability(k, len(sols), it),
                         "circuit": grover_circuit(clauses, k, it), **common})
        for method in mitigation.METHODS:
            exps.append({"id": f"mit_k{k}_{method}", "step": "mitigation", "iterations": args.mitigation_iterations,
                         "method": method, "p_theory": success_probability(k, len(sols), args.mitigation_iterations),
                         "circuit": grover_circuit(clauses, k, args.mitigation_iterations), **common})
    return exps


def ideal_reference(exps, shots):
    """Exact distribution and QASM-simulator run (probabilities and wall time) for each experiment."""
    sim = AerSimulator(seed_simulator=7, **aer_options())
    ideal = {}
    cache = {}
    for e in exps:
        qc = e["circuit"]
        key = (e["n"], tuple(e["marked"]), str(e.get("clauses")), e["iterations"])
        if key in cache:
            ideal[e["id"]] = cache[key]
            continue
        exact = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict(qargs=range(e["n"]))
        tq = transpile(qc, sim)
        t0 = time.perf_counter()
        res = sim.run(tq, shots=shots).result()
        wall = time.perf_counter() - t0
        counts = res.get_counts()
        ideal[e["id"]] = cache[key] = {"exact": {b: p for b, p in exact.items() if p > 1e-12},
                                       "qasm_counts": counts, "qasm_time_s": res.time_taken or wall}
    return ideal


def build_pubs(exps, device, args):
    """ISA circuits for every experiment, grouped into one job per hardware step."""
    target, rdelay = device.target, rep_delay(device)
    jobs = {s: {"pubs": [], "meta": []} for s in STEPS}
    for e in exps:
        tqc = transpile(e["circuit"], device, optimization_level=3, seed_transpiler=args.seed)
        e["two_qubit_gates"] = sum(v for k, v in tqc.count_ops().items() if k in ("cz", "ecr", "cx"))
        e["depth"] = tqc.depth()
        for circ, shots, meta in mitigation.build(tqc, target, e["method"], args.shots,
                                                  args.randomizations, seed=args.seed):
            est = shots * (circ.estimate_duration(target, unit="s") + rdelay)
            jobs[e["step"]]["pubs"].append((circ, None, shots))
            jobs[e["step"]]["meta"].append({"exp": e["id"], **meta, "shots": shots, "est_qpu_s": est})
    return jobs


def estimate(jobs, overhead=3.0):
    per_step = {s: sum(m["est_qpu_s"] for m in j["meta"]) + overhead for s, j in jobs.items()}
    return per_step, sum(per_step.values())


def print_estimate(jobs, budget):
    per_step, total = estimate(jobs)
    for s, t in per_step.items():
        print(f"  {STEPS[s]:<10} {len(jobs[s]['pubs']):4d} circuits  ~{t:6.1f} s QPU")
    print(f"  total ~{total:.0f} s of the {budget:.0f} s budget (Open plan: 600 s / 28 days)")
    return total


# ----------------------------------------------------------------------------- execution
def counts_of(pub_result):
    return pub_result.join_data().get_counts()


def pub_times(result, n_pubs):
    """Per-PUB QPU time from the execution spans (shots-weighted share of each span)."""
    times = [None] * n_pubs
    try:
        for span in result.metadata["execution"]["execution_spans"]:
            for idx in span.pub_idxs:
                share = span.mask(idx).sum() / span.size
                times[idx] = (times[idx] or 0.0) + span.duration * share
    except Exception:
        pass
    return times


def run_local(jobs, device):
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=AerSimulator.from_backend(device, seed_simulator=7, **aer_options()))
    out = {}
    for step, job in jobs.items():
        print(f"simulating {STEPS[step]} ({len(job['pubs'])} circuits) on {device.name} noise model [{SIM_DEVICE}]")
        res = sampler.run(job["pubs"]).result()
        out[step] = {"counts": [counts_of(r) for r in res], "times": [None] * len(job["pubs"])}
    return out


def analyse(exps, ideal, jobs, raw, device_name):
    rows = []
    for e in exps:
        idx = [i for i, m in enumerate(jobs[e["step"]]["meta"]) if m["exp"] == e["id"]]
        metas = [jobs[e["step"]]["meta"][i] for i in idx]
        counts = [raw[e["step"]]["counts"][i] for i in idx]
        p = mitigation.combine(counts, metas, e["n"], e["method"])
        dist = {format(i, f"0{e['n']}b"): float(v) for i, v in enumerate(p) if v > 0}
        times = [raw[e["step"]]["times"][i] for i in idx]
        ref = ideal[e["id"]]
        q_shots = sum(ref["qasm_counts"].values())
        row = {k: e[k] for k in ("id", "step", "n", "marked", "iterations", "method", "p_theory",
                                 "two_qubit_gates", "depth")}
        row.update({k: e[k] for k in ("k", "clauses") if k in e})
        row.update({
            "p_success_exact": sum(ref["exact"].get(s, 0) for s in e["marked"]),
            "p_success_qasm": sum(ref["qasm_counts"].get(s, 0) for s in e["marked"]) / q_shots,
            "qasm_time_s": ref["qasm_time_s"],
            "p_success_device": sum(dist.get(s, 0) for s in e["marked"]),
            "hellinger_fidelity": hellinger_fidelity(ref["exact"], dist),
            "found": max(dist, key=dist.get) in e["marked"],
            "est_qpu_time_s": sum(m["est_qpu_s"] for m in metas),
            "qpu_time_s": sum(times) if all(t is not None for t in times) else None,
            "device_dist": dist if e["n"] <= 6 else dict(sorted(dist.items(), key=lambda kv: -kv[1])[:16]),
            "exact_dist": ref["exact"] if e["n"] <= 6 else None,
        })
        rows.append(row)
    return {"device": device_name, "rows": rows}


def report(results, tag, args, sim=None):
    rows = results["rows"]
    print(f"\n{'experiment':<18}{'2q':>6}{'theory':>8}{'qasm':>8}{'device':>8}{'fid':>7}  time(s)")
    for r in rows:
        t = r["qpu_time_s"] if r["qpu_time_s"] is not None else r["est_qpu_time_s"]
        print(f"{r['id']:<18}{r['two_qubit_gates']:6d}{r['p_theory']:8.3f}{r['p_success_qasm']:8.3f}"
              f"{r['p_success_device']:8.3f}{r['hellinger_fidelity']:7.3f}  {t:.3f}")
    (ROOT / "Results").mkdir(exist_ok=True)
    path = ROOT / "Results" / f"pipeline_{tag}.json"
    path.write_text(json.dumps({**results, "shots": args.shots, "seed": args.seed}, indent=1))
    print("wrote", path.relative_to(ROOT))
    import figures

    figures.make_all(results, tag, sim, args.shots)


# ----------------------------------------------------------------------------- commands
def main(argv=None):
    global SIM_DEVICE
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["simulate", "estimate", "submit", "collect", "wait"])
    ap.add_argument("--backend", help="fake_kingston/fake_fez/fake_marrakesh, ibm_kingston/ibm_fez/ibm_marrakesh "
                                      "or auto (default: fake_kingston for simulate, auto otherwise)")
    ap.add_argument("--token")
    ap.add_argument("--instance", help="instance name or CRN, e.g. the 'Grovers K_SAT' instance")
    ap.add_argument("--shots", type=int, default=4096)
    ap.add_argument("--n-min", type=int, default=2)
    ap.add_argument("--n-max", type=int, default=10)
    ap.add_argument("--ksat-k", type=int, nargs="+", default=[5, 6])
    ap.add_argument("--ksat-clauses", type=int, default=3)
    ap.add_argument("--ksat-iterations", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--mitigation-iterations", type=int, default=1)
    ap.add_argument("--randomizations", type=int, default=8, help="twirling / TREX randomizations")
    ap.add_argument("--budget", type=float, default=480, help="refuse to submit above this QPU estimate (s)")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--device", choices=["auto", "CPU", "GPU"], default="auto", help="Aer simulator device")
    ap.add_argument("--force", action="store_true", help="submit even if jobs for this backend were already sent")
    args = ap.parse_args(argv)
    SIM_DEVICE = pick_device(args.device)
    args.backend = args.backend or ("fake_kingston" if args.command == "simulate" else "auto")
    if args.command == "simulate" and not args.backend.startswith("fake_"):
        ap.error("simulate runs on a noise model; use a fake_ backend (e.g. fake_kingston)")

    if args.command in ("collect", "wait"):
        return collect(args, wait=args.command == "wait")
    exps = experiments(args)

    device = get_device(args)
    print(f"backend: {device.name}")
    jobs = build_pubs(exps, device, args)
    total = print_estimate(jobs, args.budget)
    if args.command == "estimate":
        return total
    if args.command == "simulate":
        raw = run_local(jobs, device)
        return report(analyse(exps, ideal_reference(exps, args.shots), jobs, raw, device.name), device.name, args)

    previous = ROOT / "Results" / f"jobs_{device.name}.json"
    if previous.exists() and not args.force:
        raise SystemExit(f"{previous.relative_to(ROOT)} exists: these jobs were already submitted. Run "
                         f"`wait`/`collect` instead (or --force to spend QPU time again)")
    if total > args.budget:
        raise SystemExit(f"estimated {total:.0f} s exceeds --budget {args.budget:.0f} s; "
                         "lower --shots/--n-max or raise --budget")
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=device)  # job mode: the Open plan does not allow sessions
    manifest = {"backend": device.name, "args": vars(args) | {"token": None},
                "jobs": {}, "meta": {s: j["meta"] for s, j in jobs.items()},
                "exps": [{k: v for k, v in e.items() if k != "circuit"} for e in exps]}
    for step, job in jobs.items():
        j = sampler.run(job["pubs"])
        manifest["jobs"][step] = j.job_id()
        print(f"submitted {STEPS[step]}: job {j.job_id()}")
    path = ROOT / "Results" / f"jobs_{device.name}.json"
    path.write_text(json.dumps(manifest, indent=1))
    print("job IDs saved to", path.relative_to(ROOT), f"- run `python pipeline.py collect --backend {device.name}`")


def collect(args, wait=False):
    path = ROOT / "Results" / f"jobs_{args.backend}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - run submit first (use the concrete backend name, not auto)")
    manifest = json.loads(path.read_text())
    for k, v in manifest["args"].items():  # rebuild exactly the experiments that were submitted
        if k not in ("command", "backend", "token", "instance"):
            setattr(args, k, v)
    exps = experiments(args)
    saved = {e["id"]: e for e in manifest["exps"]}
    for e in exps:  # transpilation-dependent figures come from submit time
        e.update({k: saved[e["id"]][k] for k in ("two_qubit_gates", "depth")})
    svc = service(args)
    raw, usage = {}, {}
    for step, job_id in manifest["jobs"].items():
        job = svc.job(job_id)
        status = job.status()
        while wait and status in ("INITIALIZING", "QUEUED", "VALIDATING", "RUNNING"):
            print(f"{time.strftime('%H:%M:%S')}  {STEPS[step]} job {job_id}: {status} - checking again in 60 s")
            time.sleep(60)
            status = job.status()
        if status != "DONE":
            raise SystemExit(f"job {job_id} ({STEPS[step]}) is {status}; "
                             + ("see the IBM dashboard for the error" if status in ("ERROR", "CANCELLED")
                                else "run `wait` or try `collect` again later"))
        res = job.result()
        raw[step] = {"counts": [counts_of(r) for r in res], "times": pub_times(res, len(res))}
        usage[step] = job.usage() if hasattr(job, "usage") else None
        print(f"{STEPS[step]}: job {job_id} done, QPU usage {usage[step]} s")
    jobs = {s: {"meta": m} for s, m in manifest["meta"].items()}
    results = analyse(exps, ideal_reference(exps, args.shots), jobs, raw, manifest["backend"])
    results["job_ids"], results["qpu_usage_s"] = manifest["jobs"], usage
    twin = ROOT / "Results" / f"pipeline_fake_{manifest['backend'].removeprefix('ibm_')}.json"
    sim = json.loads(twin.read_text()) if twin.exists() else None  # noisy-simulator run, plotted alongside
    report(results, manifest["backend"], args, sim)


if __name__ == "__main__":
    main()
