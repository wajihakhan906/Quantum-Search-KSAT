# K-SAT Instances

Instances are stored in standard **DIMACS CNF** format (`p cnf <vars> <clauses>`, one clause per line ending in `0`).

- `instance_v3_c3_s3.cnf`: the 2-SAT instance used for the results in `Results/` (3 variables, 3 clauses, 2 solutions).

Generate new instances with `ksat.random_ksat(n_vars, n_clauses, k, seed, n_solutions)`, or load your own with
`ksat.read_dimacs(path)`. Standard benchmark sets (e.g. SATLIB uf20-91) can be dropped here too, but note that
each clause costs one ancilla qubit, so only small instances fit on current hardware.
- `instance_k5_v5_c3_s10.cnf`, `instance_k6_v6_c3_s11.cnf`: the 5-SAT and 6-SAT instances (3 clauses) used by `Code/pipeline.py`.
