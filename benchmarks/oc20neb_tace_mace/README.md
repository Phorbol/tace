# OC20NEB TACE vs MACE benchmark

This directory compares TACE against the existing MACE OC20NEB fullcase-200 subset run from job `674543`.

Reference MACE run:
- Train: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/train.extxyz`
- Valid: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- MACE model: `/home/gengjianrui/worktrees/mace-update-boundary/runs/oc20neb_fullcase200_ef_20k/674543/cueq_hybrid_muon/models/oc20neb_fullcase200_ef20k_cueq_hybrid_muon.model`
- Existing result: final `mae_e_mev_atom=2.63`, final `mae_f_mev_a=35.44`, best force MAE `30.21`, peak dmon FB memory `10162 MB`, `34.92 s/epoch`.

## Steps

1. Create the environment:
   `bash benchmarks/oc20neb_tace_mace/setup_env.sh`

2. Train TACE on the same subset:
   `sbatch benchmarks/oc20neb_tace_mace/train_tace_oc20neb.sbatch`

3. Run inference/accuracy/memory benchmark. Before TACE is trained this runs only the MACE baseline:
   `sbatch benchmarks/oc20neb_tace_mace/benchmark_oc20neb.sbatch`

4. After training, pass the TACE checkpoint:
   `TACE_MODEL=/path/to/TACE.ckpt sbatch benchmarks/oc20neb_tace_mace/benchmark_oc20neb.sbatch`

The benchmark JSON reports `mae_e_mev_atom`, `mae_f_mev_a`, RMSEs, `seconds_per_pass`, `configs_per_second`, `atoms_per_second`, CUDA peak allocated/reserved memory, and model parameter count.

## V100 compatibility notes

The scripts set `TORCH_CUDA_ARCH_LIST=7.0` for V100/sm70. The environment script reuses the existing CUDA 12.6 MACE environment through a `venv --system-site-packages` layer to avoid downloading a second PyTorch CUDA stack, then upgrades `e3nn` only in the TACE venv because this TACE revision needs `e3nn.Irreps.regroup()`. MACE benchmarks keep using the original MACE Python to avoid the `mace-torch` `e3nn==0.4.4` constraint conflict. Optional TACE accelerators are disabled by default in training (`TACE_USE_CUE=0`, `TACE_USE_OEQ=0`, `TACE_USE_EQT=0`) until their V100 wheel compatibility is verified.


## TECE renormalized distillation loop

This branch adds a minimal closed-loop test for the TECE design-space idea: generate teacher labels, train reduced TACE students on those labels, benchmark against both DFT and teacher validation targets, then review throughput/memory against the TECE source-document priorities.

Default teacher checkpoint:
`runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt`

1. Generate teacher-labeled train/valid files:
   `MODE=labels DISTILL_ROOT=/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/tece-distill-manual sbatch benchmarks/oc20neb_tace_mace/tece_distill_matrix.sbatch`

2. Generate reduced student configs:
   `MODE=configs DISTILL_ROOT=/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/tece-distill-manual sbatch benchmarks/oc20neb_tace_mace/tece_distill_matrix.sbatch`

3. Train one student variant:
   `MODE=train VARIANT=scalar_fast DISTILL_ROOT=/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/tece-distill-manual sbatch benchmarks/oc20neb_tace_mace/tece_distill_matrix.sbatch`

4. Benchmark a trained student checkpoint against DFT and teacher labels:
   `MODE=bench VARIANT=scalar_fast TACE_MODEL=/path/to/student.ckpt DISTILL_ROOT=/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/tece-distill-manual sbatch benchmarks/oc20neb_tace_mace/tece_distill_matrix.sbatch`

5. Summarize completed benchmark pairs:
   `python benchmarks/oc20neb_tace_mace/summarize_tece_distill.py --student scalar_fast:/path/to/scalar_fast_dft_benchmark.json:/path/to/scalar_fast_teacher_benchmark.json --output-md /path/to/tece_distill_summary.md --output-json /path/to/tece_distill_summary.json`

Stage review after each run: if throughput and memory do not improve after reducing persistent angular state, prioritize benchmark/fusion investigation; if teacher error is low but DFT error is high, prioritize teacher coverage and deployment distribution; if both errors are high, revise the projection axes before longer training.
