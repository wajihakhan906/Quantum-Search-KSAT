"""One command for the whole Grover / K-SAT study: simulator, paper figures, and IBM quantum hardware.

    python run_all.py --check        # show Python, library and GPU status only
    python run_all.py                # Steps 1-6 on the QASM simulator + ibm_kingston noise model, paper figures
    python run_all.py --hardware     # the above, then Steps 3, 5, 6 on the real ibm_kingston QPU

The IBM API key is read from the QISKIT_IBM_TOKEN environment variable, or typed in hidden when asked.
It is never written to disk. The instance CRN comes from QISKIT_IBM_INSTANCE or is asked for.

Outputs (all in the repository):
    Results/pipeline_<backend>.json   every number (probabilities, distributions, times, job IDs)
    Results/tables_<backend>.tex      LaTeX tables for the paper;  Results/table_<backend>.csv
    Figures/fig_*_<backend>.pdf/.png  publication figures (vector PDF + 300 dpi PNG)
"""
import argparse
import getpass
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ["qiskit", "qiskit-aer", "qiskit-ibm-runtime", "numpy", "scipy", "matplotlib", "pylatexenc"]
FREE_PLAN_SECONDS = 600


def _installed(pkg):
    try:
        return bool(md.version(pkg))
    except md.PackageNotFoundError:
        return False


def check_environment():
    print("=" * 70)
    print(f"Python      {platform.python_version()}  ({sys.executable})")
    print(f"System      {platform.system()} {platform.release()}  {platform.machine()}")
    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required (3.11 recommended) - see SETUP.md")
    missing = []
    for pkg in REQUIRED + ["qiskit-aer-gpu-cu11"]:
        try:
            print(f"{pkg:<20}{md.version(pkg)}")
        except md.PackageNotFoundError:
            gpu_aer = pkg == "qiskit-aer" and _installed("qiskit-aer-gpu-cu11")  # GPU build replaces qiskit-aer
            if pkg != "qiskit-aer-gpu-cu11" and not gpu_aer:
                missing.append(pkg)
                print(f"{pkg:<20}MISSING")
    if missing:
        raise SystemExit(f"missing packages: {' '.join(missing)}\n  pip install -r requirements.txt   (see SETUP.md)")
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        print(f"NVIDIA GPU  {gpu or 'not detected'}")
    from qiskit_aer import AerSimulator

    devices = AerSimulator().available_devices()
    print(f"Aer devices {', '.join(devices)}"
          + ("" if "GPU" in devices else "   (CPU only; for the RTX GPU see SETUP.md, Linux/WSL2)"))
    print("=" * 70)
    return "GPU" in devices


def ask_credentials():
    if not os.environ.get("QISKIT_IBM_TOKEN"):
        os.environ["QISKIT_IBM_TOKEN"] = getpass.getpass("IBM Quantum API key (hidden): ").strip()
    if not os.environ.get("QISKIT_IBM_INSTANCE"):
        crn = input("Instance CRN (crn:v1:bluemix:public:quantum-computing:...; Enter to let IBM pick): ").strip()
        if crn:
            os.environ["QISKIT_IBM_INSTANCE"] = crn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="only report the environment")
    ap.add_argument("--hardware", action="store_true", help="also run on the real IBM QPU")
    ap.add_argument("--backend", default="ibm_kingston", choices=["ibm_kingston", "ibm_fez", "ibm_marrakesh"])
    ap.add_argument("--device", default="auto", choices=["auto", "CPU", "GPU"], help="simulator device")
    ap.add_argument("--resimulate", action="store_true", help="redo the simulation even if results exist")
    ap.add_argument("--yes", action="store_true", help="do not ask before spending QPU time")
    args = ap.parse_args()

    os.chdir(Path(__file__).resolve().parent)
    check_environment()
    if args.check:
        return
    import pipeline

    twin = "fake_" + args.backend.removeprefix("ibm_")
    sim_json = ROOT / "Results" / f"pipeline_{twin}.json"
    if sim_json.exists() and not args.resimulate:
        print(f"[simulation] {sim_json.relative_to(ROOT)} exists - re-plotting only (--resimulate to redo)")
        import figures

        figures.make_all(json.loads(sim_json.read_text()), twin)
    else:
        t0 = time.time()
        print(f"[simulation] Steps 1-6 on the QASM simulator and the {twin} noise model "
              "(about 10-30 min on a desktop CPU)")
        pipeline.main(["simulate", "--backend", twin, "--device", args.device])
        print(f"[simulation] done in {(time.time() - t0) / 60:.1f} min")
    if not args.hardware:
        print("\nSimulation results and figures are ready. Add --hardware to run on the real quantum computer.")
        return

    ask_credentials()
    jobs_json = ROOT / "Results" / f"jobs_{args.backend}.json"
    if jobs_json.exists():
        print(f"[hardware] jobs already submitted ({jobs_json.relative_to(ROOT)}) - not spending QPU time again")
    else:
        print(f"[hardware] building circuits for {args.backend}")
        est = pipeline.main(["estimate", "--backend", args.backend, "--device", args.device])
        print(f"[hardware] this uses about {est:.0f} s of your {FREE_PLAN_SECONDS} s free monthly QPU time")
        if not args.yes and input("Submit to the real quantum computer now? [y/N] ").strip().lower() != "y":
            raise SystemExit("not submitted")
        pipeline.main(["submit", "--backend", args.backend, "--device", args.device])
    print("[hardware] waiting for IBM's queue (can take minutes to hours; safe to stop with Ctrl+C and "
          "run this command again later - nothing is resubmitted)")
    pipeline.main(["wait", "--backend", args.backend, "--device", args.device])
    print(f"\nAll results are in Results/ and Figures/ (fig_*_{args.backend}.pdf compare QPU with simulation).")


if __name__ == "__main__":
    main()
