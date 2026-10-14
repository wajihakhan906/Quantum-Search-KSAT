# HAGCS: Hardware-Aware Grover Configuration Selection

Code and results for hardware-aware configuration selection of Grover-based 2-SAT search on IBM Heron r2 processors.

## Contents

| File | Description |
|---|---|
| `HAGCS_88_Primary_Configurations.ipynb` | Full six-stage pipeline: 8 logical configurations, ideal references, RAW and error-managed runs on `ibm_kingston` and `ibm_marrakesh` (88 primary configurations), HAGCS selection, and independent validation (184 records). Outputs are kept in the notebook. |
| `figures/` | Figures produced by the notebook. |

## Requirements

Qiskit 2.5.2, Qiskit Aer 0.17.2, Qiskit IBM Runtime 0.50.0, NumPy, pandas, SciPy, Matplotlib, openpyxl.

## Running

1. Set your IBM Quantum credentials as environment variables before starting Jupyter:
   `IBM_QUANTUM_TOKEN` (API key) and `IBM_QUANTUM_INSTANCE` (instance CRN).
   If the key is not set, the notebook asks for it with hidden input.
2. Run all cells. Completed jobs are cached and are not resubmitted.

Running the hardware stages uses IBM Quantum QPU time. No credentials are stored in this repository.
