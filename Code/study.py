"""Full Grover / K-SAT study (Stages A-C) on IBM Heron devices: ibm_kingston and ibm_marrakesh.

Stage A (no QPU)    python study.py design      Step 1  circuit/instance table, 3 MCT constructions
                    python study.py baseline    Step 2  all marked states n=2..10, k=1 and k*, 5 repetitions
Stage B             python study.py predict     Step 3  Steps 5-8 circuits on FakeKingston/FakeMarrakesh
                    python study.py estimate    QPU-time estimate per device (free plan: 600 s)
                    python study.py submit      Step 4 calibration record + Steps 5-8 on the real QPUs
                    python study.py collect     fetch hardware results (add --wait to poll the queue)
Stage C (no QPU)    python study.py analyze     Steps 9-12, figures, LaTeX/CSV tables, report
                    python study.py all         design + baseline + predict + analyze (everything offline)

Hardware credentials: QISKIT_IBM_TOKEN / QISKIT_IBM_INSTANCE environment variables (or a saved account).
All outputs go to Results/study/ and Figures/study/.
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
from scipy.stats import binom

import mitigation
import oracles as orc

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Results" / "study"
FIG = ROOT / "Figures" / "study"
DEVICES = {"kingston": ("fake_kingston", "ibm_kingston"), "marrakesh": ("fake_marrakesh", "ibm_marrakesh")}
INSTANCES = {"5v": (5, 6, 1), "6v": (6, 7, 2)}  # name: (variables, clauses, search seed) -> exactly 3 solutions
METHODS_S5 = ["none", "dd", "twirl", "trex", "zne", "all"]
SHOTS = {"s5": 4096, "s6": 4096, "s7": 2048, "s8": 2048, "opt": 4096}
SEED = 11


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, indent=1, default=float))


def load(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def instances():
    out = {}
    for name, (n, c, seed) in INSTANCES.items():
        clauses = orc.find_instance(n, c, 3, seed=seed)
        out[name] = {"n": n, "clauses": clauses, "solutions": orc.solutions(clauses, n),
                     "cnf": orc.cnf_string(clauses), "cnf_latex": orc.cnf_latex(clauses)}
    return out


def fake(name):
    from qiskit_ibm_runtime import fake_provider

    return getattr(fake_provider, "Fake" + name.removeprefix("fake_").capitalize())()


def two_q(circ):
    return sum(v for k, v in circ.count_ops().items() if k in ("cz", "ecr", "cx"))


# ============================================================================ Stage A
def step1_design():
    """Circuit and instance table: every construction, logical qubits, CZ count and depth on both devices."""
    inst = instances()
    devs = {d: fake(f) for d, (f, _) in DEVICES.items()}
    rows = []
    specs = [(f"Grover n={n}", dict(n_vars=n, k=1, solutions=["1" * n], oracle="A"), n, 1) for n in range(2, 11)]
    for name, I in inst.items():
        for o in "AB":
            specs.append((f"{name} Oracle {o}", dict(n_vars=I["n"], k=1, solutions=I["solutions"],
                                                     clauses=I["clauses"], oracle=o), I["n"], 3))
    for label, kw, n, m in specs:
        for v in orc.VARIANTS:
            qc = orc.grover(variant=v, **kw)
            sv = Statevector(qc.remove_final_measurements(inplace=False)).probabilities_dict(qargs=range(n))
            marked = kw["solutions"]
            p = sum(sv.get(s, 0) for s in marked)
            row = {"circuit": label, "variant": v, "n_vars": n, "qubits": qc.num_qubits,
                   "ancillas": qc.num_qubits - n - (len(kw.get("clauses") or []) if kw["oracle"] == "B" else 0),
                   "p_ideal": p, "p_theory": orc.p_theory(n, m, 1), "verified": abs(p - orc.p_theory(n, m, 1)) < 1e-6}
            for d, dev in devs.items():
                t0 = time.perf_counter()
                tq = transpile(qc, dev, optimization_level=3, seed_transpiler=SEED)
                row[f"cz_{d}"], row[f"depth_{d}"] = two_q(tq), tq.depth()
                row[f"transpile_s_{d}"] = time.perf_counter() - t0
            rows.append(row)
            print(f"  {label:<16}{v:<13}qubits {row['qubits']:2d}  CZ kingston {row['cz_kingston']:6d}  "
                  f"marrakesh {row['cz_marrakesh']:6d}  verified {row['verified']}")
    save("step1_design.json", {"instances": inst, "rows": rows})
    return rows


def step2_baseline(reps=5, shots=1000, n_max=10):
    """Every marked state for n=2..n_max at k=1 and k*, `reps` repetitions on the ideal QASM simulator."""
    sim = AerSimulator()
    rows = []
    for n in range(2, n_max + 1):
        ks = sorted({1, orc.k_opt(n, 1)})
        for k in ks:
            states = [format(i, f"0{n}b") for i in range(2**n)]
            circs = transpile([orc.grover(n, k, "noancilla", [s]) for s in states], sim, optimization_level=0)
            p = np.zeros((len(states), reps))
            t_sim = 0.0
            for r in range(reps):
                res = sim.run(circs, shots=shots, seed_simulator=1000 * r + n).result()
                t_sim += res.time_taken
                for i, s in enumerate(states):
                    p[i, r] = res.get_counts(i).get(s, 0) / shots
            th = orc.p_theory(n, 1, k)
            hits = np.rint(p * shots)  # exact two-sided binomial test at the 3-sigma level (alpha = 0.0027)
            pval = np.minimum(1, 2 * np.minimum(binom.cdf(hits, shots, th), binom.sf(hits - 1, shots, th)))
            rows.append({"n": n, "k": k, "k_opt": orc.k_opt(n, 1), "states": len(states), "reps": reps,
                         "shots": shots, "p_theory": th, "p_mean": float(p.mean()), "p_std": float(p.std(ddof=1)),
                         "p_min": float(p.min()), "p_max": float(p.max()),
                         "max_abs_dev": float(np.abs(p.mean(axis=1) - th).max()),
                         "within_3sigma": float((pval > 0.0027).mean()),
                         "sim_time_per_circuit_s": t_sim / (reps * len(states)),
                         "p_per_state": p.mean(axis=1).tolist() if n <= 6 else None})
            r_ = rows[-1]
            print(f"  n={n:2d} k={k:2d}: theory {th:.4f}  sim {r_['p_mean']:.4f} ± {r_['p_std']:.4f}  "
                  f"in 3σ {r_['within_3sigma']:.3f}  {r_['sim_time_per_circuit_s'] * 1e3:.2f} ms/circuit")
    save("step2_baseline.json", {"rows": rows})
    return rows


# ============================================================================ Stage B: experiment list
def experiments():
    """Steps 5-8 logical circuits (identical for every device). Each: id, step, n, marked, k, method, rep."""
    inst = instances()
    rng = np.random.default_rng(SEED)
    exps = []
    for name, I in inst.items():
        base = {"instance": name, "n": I["n"], "marked": I["solutions"], "clauses": I["clauses"]}
        for method in METHODS_S5:
            for rep in (0, 1):
                exps.append({**base, "id": f"s5_{name}_{method}_r{rep}", "step": "s5", "k": 1, "method": method,
                             "rep": rep, "m3cal": method == "none" and rep == 0})
        for k in sorted({1, orc.k_opt(I["n"], 3)}):
            exps.append({**base, "id": f"s6_{name}_k{k}", "step": "s6", "k": k, "method": "none", "rep": 0})
    for n, marked in ((3, "101"), (4, "1011")):
        for k in range(4):
            exps.append({"id": f"s7_n{n}_k{k}", "step": "s7", "n": n, "marked": [marked], "k": k,
                         "method": "none", "rep": 0})
    for n in range(2, 8):
        states = rng.choice(2**n, size=min(3, 2**n), replace=False)
        for j, s in enumerate(states):
            m = format(int(s), f"0{n}b")
            for k in sorted({1} | ({orc.k_opt(n, 1)} if n <= 4 else set())):
                exps.append({"id": f"s8_n{n}_m{j}_k{k}", "step": "s8", "n": n, "marked": [m], "k": k,
                             "method": "none", "rep": 0})
    for name, I in inst.items():  # optional: best Step-1 construction (relative-phase Toffolis), no mitigation
        for rep in (0, 1):
            exps.append({"instance": name, "n": I["n"], "marked": I["solutions"], "clauses": I["clauses"],
                         "id": f"opt_{name}_rtof_r{rep}", "step": "opt", "k": 1, "method": "none", "rep": rep,
                         "variant": "vchain_rtof"})
    for e in exps:
        e.setdefault("variant", "noancilla")
        e["circuit"] = orc.grover(e["n"], e["k"], e["variant"], e["marked"])
        e["p_theory"] = orc.p_theory(e["n"], len(e["marked"]), e["k"])
    return exps


def m3_cal_circuits(tqc):
    """Two calibration circuits on the measured physical qubits: prepare all |0>, all |1>."""
    out = []
    for state in (0, 1):
        c = tqc.copy_empty_like()
        for inst in tqc.data:
            if inst.operation.name == "measure":
                if state:
                    c.x(inst.qubits[0])
        for inst in tqc.data:
            if inst.operation.name == "measure":
                c.append(inst.operation, inst.qubits, inst.clbits)
        out.append((c, state))
    return out


def build_jobs(device, exps):
    """ISA circuits for one device, one job per step. Returns ({step: {"pubs", "meta"}}, exp info)."""
    target = device.target
    try:
        rdelay = float(device.configuration().default_rep_delay)
    except Exception:
        rdelay = 250e-6
    jobs = {s: {"pubs": [], "meta": []} for s in SHOTS}
    info = {}
    cache = {}
    for e in exps:
        key = (e["n"], tuple(e["marked"]), e["k"], e["variant"])
        if key not in cache:
            cache[key] = transpile(e["circuit"], device, optimization_level=3, seed_transpiler=SEED)
        tqc = cache[key]
        shots = SHOTS[e["step"]]
        info[e["id"]] = {"cz": two_q(tqc), "depth": tqc.depth(),
                         "layout": sorted(tqc.find_bit(i.qubits[0]).index for i in tqc.data
                                          if i.operation.name == "measure")}
        items = mitigation.build(tqc, target, e["method"], shots, 8, seed=SEED + e["rep"])
        if e.get("m3cal"):
            items += [(mitigation.schedule(c, target), shots, {"role": "m3cal", "state": s, "scale": 1,
                                                                "mask": "0" * e["n"]}) for c, s in m3_cal_circuits(tqc)]
        for circ, sh, meta in items:
            est = sh * (circ.estimate_duration(target, unit="s") + rdelay)
            jobs[e["step"]]["pubs"].append((circ, None, sh))
            jobs[e["step"]]["meta"].append({"exp": e["id"], **meta, "shots": sh, "est_qpu_s": est})
    return jobs, info


def calibration_record(device, info):
    """Step 4: calibration on the run day (device-wide medians and the qubits the circuits use)."""
    t = device.target
    used = sorted({q for v in info.values() for q in v["layout"]})

    def med(xs):
        xs = [x for x in xs if x is not None]
        return float(np.median(xs)) if xs else None

    cz = {k: v.error for k, v in t["cz"].items() if v is not None and v.error is not None}
    ro = {k[0]: v.error for k, v in t["measure"].items() if v is not None and v.error is not None}
    qp = t.qubit_properties or []
    t1 = {i: p.t1 for i, p in enumerate(qp) if p is not None and p.t1}
    t2 = {i: p.t2 for i, p in enumerate(qp) if p is not None and p.t2}
    try:
        date = str(device.properties().last_update_date)
    except Exception:
        date = None
    rec = {"device": device.name, "calibration_date": date, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "num_qubits": device.num_qubits, "used_qubits": used}
    for scope, qs in (("device", None), ("used", set(used))):
        rec[scope] = {
            "cz_error_median": med([e for k, e in cz.items() if qs is None or (k[0] in qs and k[1] in qs)]),
            "readout_error_median": med([e for q, e in ro.items() if qs is None or q in qs]),
            "t1_us_median": med([v * 1e6 for q, v in t1.items() if qs is None or q in qs]),
            "t2_us_median": med([v * 1e6 for q, v in t2.items() if qs is None or q in qs]),
        }
    return rec


def run_local(jobs, device):
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=AerSimulator.from_backend(device, seed_simulator=SEED))
    raw = {}
    for step, job in jobs.items():
        t0 = time.time()
        res = sampler.run(job["pubs"]).result()
        raw[step] = {"counts": [r.join_data().get_counts() for r in res], "times": [None] * len(res)}
        print(f"    {step}: {len(job['pubs'])} circuits simulated in {time.time() - t0:.0f} s")
    return raw


def pub_times(result, n):
    times = [None] * n
    try:
        for span in result.metadata["execution"]["execution_spans"]:
            for idx in span.pub_idxs:
                times[idx] = (times[idx] or 0.0) + span.duration * span.mask(idx).sum() / span.size
    except Exception:
        pass
    return times


# ============================================================================ analysis of one source
def exact_dist(e):
    sv = Statevector(e["circuit"].remove_final_measurements(inplace=False))
    return {b: p for b, p in sv.probabilities_dict(qargs=range(e["n"])).items() if p > 1e-12}


def m3_correct(counts, cal0, cal1, n):
    import mthree

    mats = []
    for i in range(n):  # clbit i is the (n-1-i)-th character
        s0, s1 = sum(cal0.values()), sum(cal1.values())
        p10 = sum(c for b, c in cal0.items() if b[n - 1 - i] == "1") / s0  # P(1 | prepared 0)
        p01 = sum(c for b, c in cal1.items() if b[n - 1 - i] == "0") / s1  # P(0 | prepared 1)
        mats.append(np.array([[1 - p10, p01], [p10, 1 - p01]]))
    m = mthree.M3Mitigation()
    m.cals_from_matrices(mats)
    q = m.apply_correction(counts, list(range(n)))
    pd = q.nearest_probability_distribution()
    v = np.zeros(2**n)
    for b, p in pd.items():
        v[int(b, 2)] = float(p)
    return v


def bootstrap(counts_list, metas, n, method, sol_idx, B, rng):
    vals = []
    for _ in range(B):
        res = []
        for c in counts_list:
            keys = list(c)
            tot = sum(c.values())
            draw = rng.multinomial(tot, np.array([c[k] for k in keys], dtype=float) / tot)
            res.append(dict(zip(keys, draw)))
        vals.append(mitigation.combine(res, metas, n, method)[sol_idx].sum())
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def analyse_source(source, exps, jobs_meta, raw, info, B=200):
    rng = np.random.default_rng(SEED)
    rows = []
    cal, cal_cost = {}, {}
    for e in exps:
        step_meta = jobs_meta[e["step"]]
        idx = [i for i, m in enumerate(step_meta) if m["exp"] == e["id"]]
        main = [i for i in idx if step_meta[i]["role"] != "m3cal"]
        metas = [step_meta[i] for i in main]
        counts = [raw[e["step"]]["counts"][i] for i in main]
        for i in idx:
            if step_meta[i]["role"] == "m3cal":
                cal.setdefault(e["instance"], {})[step_meta[i]["state"]] = raw[e["step"]]["counts"][i]
                cal_cost[e["instance"]] = cal_cost.get(e["instance"], 0) + step_meta[i]["est_qpu_s"]
        n = e["n"]
        sol_idx = np.array([int(s, 2) for s in e["marked"]])
        ideal = exact_dist(e)
        variants = [(e["method"], mitigation.combine(counts, metas, n, e["method"]))]
        if e["step"] == "s5" and e["method"] == "none":  # M3 on the unmitigated counts of every repetition
            variants.append(("m3", None))
        times = [raw[e["step"]]["times"][i] for i in main]
        for method, p in variants:
            if method == "m3":
                c0 = cal.get(e["instance"], {})
                if 0 not in c0:
                    continue
                p = m3_correct(counts[0], c0[0], c0[1], n)
                ci = None  # M3 output is a corrected distribution; its spread is set by the raw counts
                est = sum(m["est_qpu_s"] for m in metas) + cal_cost[e["instance"]]
            else:
                plain = method == "none"
                ci = bootstrap(counts, metas, n, method, sol_idx, 1000 if plain else B, rng)
                est = sum(m["est_qpu_s"] for m in metas)
            dist = {format(i, f"0{n}b"): float(v) for i, v in enumerate(p) if v > 0}
            p_sol = float(p[sol_idx].sum())
            clause_ok = (float(sum(v for b, v in dist.items() if orc.satisfies(e["clauses"], b)))
                         if e.get("clauses") else None)
            rows.append({"source": source, "id": e["id"] if method == e["method"] else e["id"].replace("_none_", "_m3_"),
                         "step": e["step"], "n": n, "k": e["k"], "method": method, "rep": e["rep"],
                         "variant": e["variant"],
                         "instance": e.get("instance"), "marked": e["marked"], "p_theory": e["p_theory"],
                         "p": p_sol, "ci95": ci, "p_clause_check": clause_ok,
                         "hellinger": hellinger_fidelity(ideal, dist),
                         "cz": info[e["id"]]["cz"], "depth": info[e["id"]]["depth"], "est_qpu_s": est,
                         "qpu_s": sum(times) if times and all(t is not None for t in times) else None,
                         "dist": dist if n <= 6 else None, "ideal": ideal if n <= 6 else None})
    return rows


# ============================================================================ commands
def cmd_predict(args):
    exps = experiments()
    for d, (fk, _) in DEVICES.items():
        if args.device and d not in args.device:
            continue
        dev = fake(fk)
        print(f"  [{fk}] building circuits")
        jobs, info = build_jobs(dev, exps)
        for s, j in jobs.items():
            print(f"    {s}: {len(j['pubs'])} circuits, est. QPU {sum(m['est_qpu_s'] for m in j['meta']):.1f} s")
        raw = run_local(jobs, dev)
        rows = analyse_source(fk, exps, {s: j["meta"] for s, j in jobs.items()}, raw, info)
        save(f"source_{fk}.json", {"source": fk, "kind": "noise model", "calibration": calibration_record(dev, info),
                                   "rows": rows})


def service():
    from qiskit_ibm_runtime import QiskitRuntimeService

    tok, inst = os.environ.get("QISKIT_IBM_TOKEN"), os.environ.get("QISKIT_IBM_INSTANCE")
    if tok:
        return QiskitRuntimeService(channel="ibm_quantum_platform", token=tok, instance=inst)
    return QiskitRuntimeService(instance=inst) if inst else QiskitRuntimeService()


def cmd_estimate(args, submit=False):
    exps = experiments()
    svc = service() if not args.offline else None
    total = 0.0
    plans = {}
    for d, (fk, hw) in DEVICES.items():
        if args.device and d not in args.device:
            continue
        dev = svc.backend(hw) if svc else fake(fk)
        jobs, info = build_jobs(dev, exps)
        per = {s: sum(m["est_qpu_s"] for m in j["meta"]) + 2.0 for s, j in jobs.items()}
        print(f"  {hw}: " + ", ".join(f"{s} {t:.0f} s" for s, t in per.items()) + f"  = {sum(per.values()):.0f} s")
        total += sum(per.values())
        plans[hw] = (dev, jobs, info)
    print(f"  total estimate {total:.0f} s (free plan: 600 s / 28 days, budget {args.budget:.0f} s)")
    if not submit:
        return
    if total > args.budget:
        raise SystemExit("estimate above --budget; use --device kingston (one device) or raise --budget")
    from qiskit_ibm_runtime import SamplerV2

    for hw, (dev, jobs, info) in plans.items():
        path = OUT / f"jobs_{hw}.json"
        if path.exists() and not args.force:
            print(f"  {hw}: already submitted ({path.name}) - skipped")
            continue
        manifest = {"device": hw, "calibration": calibration_record(dev, info), "info": info, "jobs": {},
                    "meta": {s: j["meta"] for s, j in jobs.items()}}
        sampler = SamplerV2(mode=dev)  # job mode (the Open plan has no sessions)
        for s, j in jobs.items():
            manifest["jobs"][s] = sampler.run(j["pubs"]).job_id()
            print(f"  {hw} {s}: job {manifest['jobs'][s]}")
        save(path.name, manifest)


def cmd_collect(args):
    svc = service()
    exps = experiments()
    for d, (fk, hw) in DEVICES.items():
        manifest = load(f"jobs_{hw}.json")
        if not manifest or (args.device and d not in args.device):
            continue
        raw = {}
        usage = {}
        for s, jid in manifest["jobs"].items():
            job = svc.job(jid)
            st = job.status()
            while args.wait and st in ("INITIALIZING", "QUEUED", "VALIDATING", "RUNNING"):
                print(f"  {time.strftime('%H:%M:%S')} {hw} {s} {jid}: {st}")
                time.sleep(60)
                st = job.status()
            if st != "DONE":
                raise SystemExit(f"{hw} {s} job {jid} is {st}")
            res = job.result()
            raw[s] = {"counts": [r.join_data().get_counts() for r in res], "times": pub_times(res, len(res))}
            try:
                usage[s] = job.usage()
            except Exception:
                usage[s] = None
        rows = analyse_source(hw, exps, manifest["meta"], raw, manifest["info"])
        save(f"source_{hw}.json", {"source": hw, "kind": "hardware", "calibration": manifest["calibration"],
                                   "job_ids": manifest["jobs"], "qpu_usage_s": usage, "rows": rows})
        print(f"  {hw}: collected, QPU usage {usage}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["design", "baseline", "predict", "estimate", "submit", "collect",
                                        "analyze", "all"])
    ap.add_argument("--device", nargs="+", choices=list(DEVICES), help="limit to these devices")
    ap.add_argument("--budget", type=float, default=560)
    ap.add_argument("--offline", action="store_true", help="estimate with the fake devices (no account)")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.command in ("design", "all"):
        print("Step 1: circuit design"); step1_design()
    if args.command in ("baseline", "all"):
        print("Step 2: simulator baseline"); step2_baseline()
    if args.command in ("predict", "all"):
        print("Step 3: noise-model predictions"); cmd_predict(args)
    if args.command == "estimate":
        cmd_estimate(args)
    if args.command == "submit":
        cmd_estimate(args, submit=True)
    if args.command == "collect":
        cmd_collect(args)
    if args.command in ("analyze", "all"):
        import study_analysis

        study_analysis.run()
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
