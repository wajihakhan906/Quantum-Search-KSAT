# Setup and Run Guide

This guide takes you from a fresh PC to every result and paper figure of the 6-step pipeline:
first on the simulator, then on the real IBM quantum computer (free Open plan).

| What | Version |
|---|---|
| Python | **3.11** or **3.12** (3.10 also works) |
| qiskit | 2.5.2 |
| qiskit-aer (CPU) / qiskit-aer-gpu-cu11 (GPU) | 0.17.2 |
| qiskit-ibm-runtime | 0.50.0 |
| numpy / scipy / matplotlib / pylatexenc | 2.4.6 / 1.17.1 / 3.11.2 / 2.11 |

All versions are pinned in `Code/requirements.txt` (CPU) and `Code/requirements-gpu.txt` (GPU).

---

## Choose one way to install

### A. Windows, CPU only (simplest)
1. Install **Python 3.12** from <https://www.python.org/downloads/>. In the installer, tick **"Add python.exe to PATH"**.
2. Install **Git** from <https://git-scm.com/download/win>, then open **Command Prompt** and run:
   ```bat
   git clone -b claude/ibm-quantum-free-plan-li1whg https://github.com/wajihakhan906/Quantum-Search-KSAT
   cd Quantum-Search-KSAT
   install_windows.bat
   ```
The full simulation takes about 10–30 minutes on a desktop CPU.

### B. Windows with the RTX 4070 GPU (through WSL2)
IBM's GPU simulator only runs on Linux. WSL2 runs Ubuntu inside Windows and gives it access to your GPU.
1. Update the **NVIDIA driver** in Windows (GeForce Experience, or <https://www.nvidia.com/Download/index.aspx>).
   Do **not** install a separate CUDA toolkit or driver inside Ubuntu.
2. Open **PowerShell as Administrator** and run `wsl --install -d Ubuntu-24.04`. Restart the PC, then open **Ubuntu** from
   the Start menu and choose a user name and password.
3. In the Ubuntu window:
   ```bash
   sudo apt update && sudo apt install -y git python3-venv python3-pip
   nvidia-smi                      # must list "NVIDIA GeForce RTX 4070"
   git clone -b claude/ibm-quantum-free-plan-li1whg https://github.com/wajihakhan906/Quantum-Search-KSAT
   cd Quantum-Search-KSAT
   ./install_linux.sh              # installs the GPU simulator because nvidia-smi works
   ```
   The check at the end should print `Aer devices  CPU, GPU`.

Note: these circuits use at most 10 qubits, which is small for a GPU. The GPU mainly speeds up the many noisy-shot simulations.
A CPU gives the same results.

### C. Linux / macOS
`./install_linux.sh` (macOS always uses the CPU).

---

## Run

Activate the environment first: `.venv\Scripts\activate` on Windows, or `source .venv/bin/activate` on Linux/WSL2.

```bash
python Code/run_all.py --check        # show Python, libraries and whether the GPU is used
python Code/run_all.py                # Steps 1-6 on the simulator + ibm_kingston noise model, figures, tables
python Code/run_all.py --hardware     # then Steps 3, 5, 6 on the real ibm_kingston quantum computer
```

### Real quantum computer (free Open plan: 10 minutes per month)
* `--hardware` asks for your **IBM API key** (hidden while you type) and your **instance CRN**. The CRN is the
  `crn:v1:bluemix:public:quantum-computing:...` text on your instance page; use its copy button. Neither is saved.
  To avoid typing them every time, set the environment variables `QISKIT_IBM_TOKEN` and `QISKIT_IBM_INSTANCE`.
* The script shows the QPU time it will use (about **90 s of your 600 s**) and asks before submitting.
  It sends 3 jobs to `ibm_kingston` (use `--backend ibm_fez` or `--backend ibm_marrakesh` for another computer).
* It then waits for IBM's queue, which can take from minutes to hours. You can close the window. Running the same command again
  **does not resubmit or spend time again**: it picks up the saved job IDs in `Results/jobs_ibm_kingston.json`.
* Never share your API key. If it was shown to anyone, delete it and create a new one
  (IBM Cloud → Manage → Access (IAM) → API keys).

---

## Outputs

| File | Content |
|---|---|
| `Figures/fig_steps1-3_grover_<backend>.pdf/.png` | (a) P(marked state) vs. n: theory, QASM, noisy simulator, QPU; (b) circuit size; (c) search time |
| `Figures/fig_steps4-5_ksat_<backend>.pdf/.png` | 5-SAT and 6-SAT output distributions and P(solution) vs. Grover iterations |
| `Figures/fig_step6_mitigation_<backend>.pdf/.png` | P(solution) and fidelity without/with DD, TREX, twirling, ZNE, and all combined |
| `Results/tables_<backend>.tex` | LaTeX tables (booktabs) ready for `\input{}` in the paper |
| `Results/table_<backend>.csv` | the same numbers for Excel |
| `Results/pipeline_<backend>.json` | every number, including full distributions, times and IBM job IDs |

`<backend>` is `fake_kingston` for the simulation and `ibm_kingston` for the real computer. The `ibm_kingston` figures
show the QPU and the simulator side by side. The PDFs are vector graphics sized for one-column (3.5 in) and two-column
(7.16 in) journal layouts. To re-plot without re-running anything, use `python Code/figures.py ibm_kingston`.

## Troubleshooting
| Message | Fix |
|---|---|
| `missing packages` | Activate `.venv` first, then `pip install -r Code/requirements.txt` |
| `Aer devices CPU` under WSL2 | Run `nvidia-smi` in Ubuntu; if it fails, update the Windows NVIDIA driver and run `wsl --update` |
| `Aer reports no GPU` | Use `--device CPU`, or `pip uninstall -y qiskit-aer && pip install -r Code/requirements-gpu.txt` |
| `401` / `Unable to retrieve instances` | Wrong API key or CRN; create a new key and copy the CRN again |
| `estimated ... exceeds --budget` | Your monthly minutes are low; run `python Code/pipeline.py submit --backend ibm_kingston --n-max 7` |
| Job `ERROR` on n = 10 | The 40 000-gate circuit may be rejected; use `--n-max 9` as above |
