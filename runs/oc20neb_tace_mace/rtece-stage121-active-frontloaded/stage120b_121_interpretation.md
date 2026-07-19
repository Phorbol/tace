# Stage120b/121 Interpretation

Date: 2026-07-20

Code baseline: `ec2bf08` (`Fix rTECE bottleneck lightning training path`)

## Validity

The original Stage120 runs are invalid for descriptor-bottleneck conclusions because the Lightning training path accepted the CLI flag but did not pass `descriptor_bottleneck_dim` into `fit_rtece_lightning`. Jobs `684806` and `684810` were canceled; completed old Stage120 jobs should be treated as duplicate no-bottleneck baselines.

Stage120b and Stage121 were run after fixing the Lightning path and adding a regression test that loads the portable checkpoint config. The completed sbatch jobs were:

- Stage120b: `684875`, `684876`, `684877`, `684878`
- Stage121: `684880`, `684881`, `684882`, `684883`

All eight jobs completed with exit code `0:0`.

## Stage120b Result

Stage120b tests descriptor bottlenecks and a front low-rank path mixer on broader L0 and L2/cavity rows.

The best DFT force RMSE in this block is `l0_species8_bneck16_h64`:

- params: `7417`
- atoms/s: `2.135e+06`
- F RMSE: `115.390` meV/A
- F MAE: `37.182` meV/A
- F max: `4835.172` meV/A
- E RMSE: `302.912` meV/atom
- E MAE: `243.452` meV/atom
- E max: `713.779` meV/atom

The L2/cavity rows are slower (`5.27e+05` atoms/s) and have worse F RMSE (`127.650` to `137.316` meV/A). This does not support keeping the current cavity/L2 implementation on the throughput-first Pareto front.

Against the Stage119 L0 reference discussed before this run (`F RMSE 111.498`, `E RMSE 304.855`, `2.403e+06` atoms/s), the bottleneck row slightly improves E RMSE but worsens F RMSE and throughput. Therefore this is not a useful Pareto movement.

## Stage121 Result

Stage121 tests the Stage114 active E/F path selection plus frontloaded trainable representation capacity: low-rank species basis, learnable cross-radial projection, descriptor bottleneck, and a front low-rank mixer.

The only DFT Pareto row in this block is `l1_active_species16_bneck16_h64`:

- params: `9449`
- atoms/s: `1.910e+06`
- F RMSE: `113.073` meV/A
- F MAE: `33.998` meV/A
- F max: `4915.273` meV/A
- E RMSE: `302.927` meV/atom
- E MAE: `241.193` meV/atom
- E max: `721.859` meV/atom

This row is better than the wider bneck32 and L2 active variants inside Stage121, but still does not beat the Stage119 L0 reference in F RMSE or throughput. The result supports the user concern that increasing trainable parameters only through a light low-rank conditioning layer is still too weak if the fixed descriptor basis is the limiting representation.

## TECE/rTECE Interpretation

This is aligned with `TECE_design_space.md` in the negative sense: simply adding a trainable bottleneck/head on top of mostly fixed descriptors is still a parameter/head adjustment, not a true TECE-style model compiler over teacher operator space. It does not produce a new Pareto point.

It also matches `rTECE_review.md`: the current successful rows are closer to scalar endpoint/T4 than true rTECE/T3 because the edge-relational scalar sketch and teacher-projection compiler are not yet doing useful work. Current cavity/L2 paths carry cost without improving the physical RMSE frontier.

The next algorithmic priority should not be to widen the final MLP. The next useful branch is a front-representation change that is still systematic:

1. Build a semantic supernet path basis with explicit path IDs for radial, element-density, atomic `L_max`, and edge-relational sketches.
2. Compute teacher-projected Gram/active-set scores for candidate paths using E/F residuals, not only value labels.
3. Add learnable radial/POD mixing before scalarization, with rank as the controlled renormalization knob.
4. Run a capacity ladder in the `10k` to `50k` parameter range by adding capacity in the representation basis and path mixers, not by widening only the final head.
5. Keep DFT E/F RMSE, max error, throughput, dimer scans, and rattle/relax tests as separate acceptance axes.

