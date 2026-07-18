# Stage 53: Formal rTECE Scalar Model Entrypoint

Question: after Stage52 proved the high-throughput rTECE endpoint as a benchmark-local prototype, should the model remain hidden under `benchmarks/`, or should the repository expose it as a first-class TACE/TECE model endpoint?

Implementation:

- Promoted the real rTECE scalar model implementation to `tace/models/rtece_scalar.py`.
- Kept `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py` as a compatibility shim that re-exports the core module, so existing training, benchmark, and historical scripts continue to work.
- Promoted the existing rTECE Triton backend to `tace/models/rtece_triton_kernels.py`; the old benchmark kernel path is now a compatibility shim.
- Exported `RTECEScalarModel`, `RTECEScalarConfig`, `RTECEGraph`, descriptor helpers, and rTECE descriptor variants from `tace.models`.
- Made heavy `tace.models` exports (`e3nnTACE`, `TensorModel`, `CompileTensorModel`) optional at import time. This matters because the scalar endpoint should not fail to import just because the full equivariant teacher stack or its e3nn constants are incompatible with the current PyTorch safe-load behavior.

Why this matters for the TECE/TACE route:

- The high-throughput branch is no longer only a benchmark script. It now has a formal model namespace under `tace.models`, which makes it a real candidate endpoint of the TECE model compiler route.
- This does not claim the rTECE endpoint is production-complete. The current Triton backend is now in the model package, but the Stage52 cell-list fused descriptor/force kernel is still only an oracle and must become a real backend before this endpoint is deployment-complete.
- The migration clarifies the architecture boundary: `tace.models.rtece_scalar` owns the semantic model and conservative scalar descriptors; `tace.models.rtece_triton_kernels` owns reusable low-level rTECE kernels; benchmark scripts own dataset loading, Pareto measurement, and experimental provider plumbing.

Verification:

- TDD red: the new formal-entrypoint test first failed when `tace.models` tried to import heavy e3nn-dependent exports before rTECE.
- Target tests: `python -m pytest test/test_rtece_scalar.py -q -k formal_tace_models_entrypoint or rtece_triton_kernels_have_formal` -> 2 passed.
- Full rTECE test file: `python -m pytest test/test_rtece_scalar.py -q` -> 64 passed.
