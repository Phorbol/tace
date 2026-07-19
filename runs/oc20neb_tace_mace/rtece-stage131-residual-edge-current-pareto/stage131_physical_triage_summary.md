# rTECE Stage131 Physical Triage

Job 685161 completed with exit code 0:0 for the L1 cavity-vector residual row.

| variant | physical gate | dimer gate | rattle gate | C/N RMSD | C/N max fmax | DFT F max | atoms/s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| l1_active_species24_cavity_vec_residual_h64 | False | True | False | 0.117 | 12.463 | 2296.722 | 567140 |

Compared with stage129 species24, DFT F max improves by 400.810 meV/A and C/N RMSD changes by -0.000286 A, but rattle max fmax worsens by 1.626 eV/A and throughput drops by 657278 atoms/s.

Interpretation: the edge residual carries some benchmark tail signal, but this implementation does not repair the physical rattle failure. The next priority should not be naive higher-L edge residuals; it should be cheaper edge projection diagnostics, broader teacher trajectory/Jacobian distillation, or front representation learning targeted at high-force tails.
