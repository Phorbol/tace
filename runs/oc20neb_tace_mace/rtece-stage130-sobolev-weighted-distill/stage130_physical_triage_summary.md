# Stage130 physical triage summary

Both stage130 physical triage sbatch jobs completed: species20 job 685092 and species24 job 685091.

| row | gate | score | DFT F RMSE | DFT F max | C/N mean RMSD | C/N max RMSD | C/N max fmax | delta max fmax vs stage129 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 0 | 20.420 | 119.029 | 2396.761 | 0.119 | 0.175 | 16.393 | 2.371 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 0 | 16.165 | 120.172 | 2300.489 | 0.107 | 0.144 | 11.958 | 1.122 |

Interpretation:

- Sobolev weighting does not pass the physical gate for either row.
- species24 is still the better physical candidate: C/N mean RMSD is 0.107 A, but C/N max fmax is 11.958 eV/A, worse than stage129 by 1.122 eV/A.
- species20 worsens both C/N RMSD and max fmax relative to stage129.
- This weakens the case for more weighting-only tuning. The next algorithmic step should return to TECE representation/projection changes, especially edge-relational scalar sketches or broader teacher trajectory/Jacobian distillation.
