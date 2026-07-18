# Stage 51: Cell-List Oracle Work Analysis

Question: after Stage50 showed padded Triton is the peak-throughput provider and counted Triton is mainly a memory-headroom point, how much padded candidate scan can a direct-active cell-list provider remove on the same real OC20NEB trajectory geometry?

Setup:

- Same radial8h24 `rtece_element_density` cutoff: 5.0 A.
- Same OC20NEB valid limits as Stage50: 1024, 2048, 4096, 8192, 10000 configs.
- Same trajectory geometry as Stage50's final invalid-cache rebuild: synthetic replay step 19, displacement std 0.001.
- CPU oracle only: cell size = cutoff, scan 27 neighboring cells, count directed candidate pairs, count direct-active edges, and verify active edge count against the loop direct-radius graph.

| limit configs | atoms | padded slots | all-pair nonself | cell candidates | active edges | candidate/padded | candidate/active | max cell occupancy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 59,193 | 7,750,656 | 3,640,296 | 2,850,522 | 825,584 | 0.367778 | 3.452734 | 18 |
| 2048 | 119,271 | 15,501,312 | 7,386,312 | 5,754,520 | 1,634,364 | 0.371228 | 3.520954 | 18 |
| 4096 | 250,355 | 40,960,000 | 16,402,870 | 12,406,382 | 3,459,786 | 0.302890 | 3.585881 | 18 |
| 8192 | 525,769 | 81,920,000 | 35,936,006 | 26,219,846 | 7,303,632 | 0.320066 | 3.589974 | 20 |
| 10000 | 646,473 | 100,000,000 | 44,242,578 | 32,313,616 | 9,006,668 | 0.323136 | 3.587744 | 20 |

Interpretation:

- A cell-list provider has a real work-reduction target: it removes roughly 63-70% of padded candidate checks on the measured OC20NEB batches. At 10000 configs, candidate checks fall from 100.0M padded slots to 32.3M cell candidates.
- It is not a 10x candidate-work reduction by itself. The cell-list candidate set is still about 3.5x the active directed edge count, because the 27-cell stencil includes geometrically near cells that still fail the cutoff.
- This makes the Stage50 next step sharper: the clean high-throughput path is not another counted/two-pass provider. It is a cell-list or cell-list-fused descriptor/force path that removes most padded scan work and avoids extra pass/edge-buffer lifetime where possible.
- The TECE/TACE interpretation is consistent: semantic/model renormalization already reduced channels and descriptors; the current bottleneck is hardware representation of local topology. Cell-list is the next representation renormalization, but the largest gain likely requires fusing candidate generation with descriptor accumulation instead of materializing an intermediate edge list.

Verification:

- `python -m pytest test/test_rtece_scalar.py -q`: 60 passed, 1 warning.
- `run_oracle.py` produced all five JSON outputs plus `cell_list_oracle_summary.json`; active edge counts match Stage50 trajectory-step provider metadata.
