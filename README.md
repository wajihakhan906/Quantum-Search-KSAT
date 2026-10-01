# Quantum Search for K-SAT

Grover's algorithm applied to Boolean satisfiability (K-SAT), with an explicit analysis of how hardware errors
propagate through the circuit on IBM Brisbane (127-qubit Eagle).

![Success probability vs. Grover iterations](Figures/success_fake_brisbane_v3_c3_s3.png)

## Method
1. **Oracle**: one ancilla per clause computes *clause satisfied* with an X-conjugated multi-controlled X;
   a multi-controlled Z over all clause ancillas marks satisfying assignments; the ancillas are then uncomputed.
2. **Diffuser**: standard inversion about the mean on the variable register.
3. **Iterations**: ⌊π / 4θ⌋ with sin θ = √(M/N) for M solutions among N = 2ⁿ assignments.
4. **Error propagation**: using the backend calibration, the probability that the transpiled circuit runs without a
   fault is P<sub>ff</sub> = Π<sub>g</sub>(1 − e<sub>g</sub>); the predicted success is
   P<sub>ff</sub> · P<sub>ideal</sub> + (1 − P<sub>ff</sub>) · M/N. This is compared with the noisy measurement.

![Grover circuit](Figures/circuit_fake_brisbane_v3_c3_s3.png)

## Repository Structure
```
Quantum-Search-KSAT/
├── Code/
│   ├── ksat.py              # CNF instances, brute-force solutions, DIMACS I/O
│   ├── grover.py            # clause-ancilla phase oracle, diffuser, Grover circuit
│   ├── error_analysis.py    # gate-error budget and success prediction
│   ├── run_experiments.py   # ideal vs. noisy runs, writes Results/ and Figures/
│   ├── pipeline.py          # 6-step pipeline: simulate / estimate / submit / collect on IBM Heron QPUs
│   ├── mitigation.py        # DD, TREX, Pauli twirling, ZNE (client-side, works on Aer and QPUs)
│   ├── figures.py           # publication figures (PDF + PNG) and LaTeX/CSV tables
│   ├── run_all.py           # one command: environment check, simulation, real QPU, figures
│   ├── requirements-gpu.txt # NVIDIA GPU simulator (Linux / WSL2)
│   └── requirements.txt
├── Dataset/                 # DIMACS CNF instances
├── Figures/                 # success curves, histograms, circuit diagram
├── Results/                 # JSON outputs and summary tables
├── LICENSE
└── README.md
```

## Quick Start
```bash
cd Code
pip install -r requirements.txt
python run_experiments.py                                  # FakeBrisbane noise model
python run_experiments.py --vars 4 --clauses 7 --k 3        # 3-SAT, 4 variables, 2 solutions (11 qubits)
python run_experiments.py --backend ibm_brisbane           # real hardware (saved IBM Quantum account)
```

## Full 12-step study (Stages A–C)
`Code/study.py` with `Code/oracles.py` and `Code/study_analysis.py` runs the complete study:
- Stage A: circuit design with three multi-controlled-gate constructions, and a simulator baseline over every marked state.
- Stage B: K-SAT error mitigation (DD, twirling, TREX, ZNE, M3), K-SAT search, the iteration sweep and Grover scaling on
  `ibm_kingston` and `ibm_marrakesh`, with noise-model predictions alongside.
- Stage C: model accuracy, the noise-scaling fit, time-to-solution, bootstrap statistics and the cost–benefit comparison.

**All results are collected in [Results/study/REPORT.md](Results/study/REPORT.md).** Figures are in `Figures/study/`.
Run it with `python Code/run_all.py --study [--hardware]` (see [SETUP.md](SETUP.md)).

## Six-Step Pipeline on IBM Heron (Open plan)
**New here? Follow [SETUP.md](SETUP.md)**: it covers installation (Windows, or WSL2 with an NVIDIA GPU) and the one-command run `python Code/run_all.py [--hardware]`.

`Code/pipeline.py` runs the full study on the Open-plan QPUs **ibm_kingston**, **ibm_fez** and **ibm_marrakesh**
(156-qubit Heron r2, CZ basis) or on their local noise models (`fake_kingston`, `fake_fez`, `fake_marrakesh`).

| Step | What | Figure |
|---|---|---|
| 1-3 | Grover, one marked state of 2ⁿ, n = 2…10 (H + MCT oracle, MCZ diffuser, optimal iterations): P(marked) and search time, QASM simulator vs. QPU | `Figures/fig_steps1-3_grover_<backend>.pdf` |
| 4-5 | K-SAT with K = 5, 6, 3 clauses, on 5 and 6 variable qubits (+1 ancilla per clause), 0-2 iterations | `Figures/fig_steps4-5_ksat_<backend>.pdf` |
| 6 | K-SAT with and without DD, TREX, Pauli twirling, ZNE, and all combined, simulator vs. QPU | `Figures/fig_step6_mitigation_<backend>.pdf` |

```bash
cd Code
python pipeline.py simulate                          # QASM + FakeKingston noise model, no account needed
python pipeline.py estimate --backend ibm_kingston   # QPU-time estimate (~90 s of the 600 s monthly allowance)
python pipeline.py submit   --backend ibm_kingston --instance "<CRN of your instance>" --token "<API key>"
python pipeline.py collect  --backend ibm_kingston --instance "<CRN>" --token "<API key>"
```
`submit` sends 3 jobs (one per hardware step) in job mode, because the Open plan does not allow sessions. It also refuses to submit
when the estimate exceeds `--budget` (480 s by default), and it saves the job IDs to `Results/jobs_<backend>.json`. Run `collect` once
the jobs are done (queues can take hours). It plots the QPU results next to the matching `fake_` simulation.
Error mitigation is applied on the client side (`Code/mitigation.py`), so the same code runs on the simulator and on the QPU.

## Author
**Wajiha Rahim Khan**  
[Google Scholar](https://scholar.google.com/citations?user=ctvOkbYAAAAJ)

## License
MIT. See [LICENSE](LICENSE).
