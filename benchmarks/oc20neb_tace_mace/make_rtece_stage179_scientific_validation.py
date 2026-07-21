#!/usr/bin/env python3
"""Create the Stage179 scientific validation matrix for the rTECE program."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_STAGE178_ANALYSIS = Path(
    "runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/stage178_benchmark_analysis.json"
)
DEFAULT_STAGE177_AUDIT = Path("runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage177_pareto_audit.json")
DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage179-scientific-validation")


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _row_by_name(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, Any] | None:
    for row in rows:
        if row.get(key) == name:
            return row
    return None


def _fmt_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def _stage178_delta(stage178: dict[str, Any]) -> dict[str, Any]:
    rows = list(stage178.get("rows") or [])
    l0 = _row_by_name(rows, "variant", "stage178_l0_local_species")
    t3 = _row_by_name(rows, "variant", "stage178_t3_minimal_cavity_direct")
    if l0 is None or t3 is None:
        return {}
    return {
        "l0_variant": l0.get("variant"),
        "t3_variant": t3.get("variant"),
        "l0_f_rmse_mev_a": l0.get("dft_f_rmse_mev_a"),
        "t3_f_rmse_mev_a": t3.get("dft_f_rmse_mev_a"),
        "l0_e_rmse_mev_atom": l0.get("dft_e_rmse_mev_atom"),
        "t3_e_rmse_mev_atom": t3.get("dft_e_rmse_mev_atom"),
        "l0_atoms_per_second_limit1024": l0.get("atoms_per_second_limit1024"),
        "t3_atoms_per_second_limit1024": t3.get("atoms_per_second_limit1024"),
        "l0_rattle_rmsd_a": l0.get("focus_rattle_final_rmsd_a"),
        "t3_rattle_rmsd_a": t3.get("focus_rattle_final_rmsd_a"),
    }


def _stage177_context(stage177: dict[str, Any]) -> dict[str, Any]:
    rows = list(stage177.get("rows") or [])
    stage176 = _row_by_name(rows, "candidate", "stage176_local_l0_rank3")
    nep = _row_by_name(rows, "candidate", "nep4_mixed_smoke")
    dpa = _row_by_name(rows, "candidate", "deepmd_dpa_like_mixed_smoke")
    return {
        "stage176": stage176 or {},
        "nep4_mixed_smoke": nep or {},
        "deepmd_dpa_like_mixed_smoke": dpa or {},
        "caveats": list(stage177.get("comparison_caveats") or stage177.get("caveats") or []),
    }


def build_stage179_validation_matrix(
    *,
    stage178_analysis: str | Path = DEFAULT_STAGE178_ANALYSIS,
    stage177_audit: str | Path = DEFAULT_STAGE177_AUDIT,
) -> dict[str, Any]:
    stage178 = _read_json(stage178_analysis)
    stage177 = _read_json(stage177_audit)
    delta178 = _stage178_delta(stage178)
    context177 = _stage177_context(stage177)
    best = (stage178.get("best_by_error") or {}).get("variant")

    required = [
        {
            "id": "early_scalarization_hardware_hypothesis",
            "question": "Can a TECE-derived student approach NEP/MTP/DPA1-0-style throughput by dropping persistent high-l edge state and scalarizing after one moment pass?",
            "why_required": "This is the basic hardware premise of scalar-sketched rTECE; without it the program collapses into ordinary small equivariant GNN tuning.",
            "status": "supported",
            "evidence": [
                "TECE_design_space.md argues NEP/DPA1-0 are fast because geometry is compressed early into scalar sufficient statistics.",
                "rTECE_review.md judged the current prototype as validating early scalarization as a throughput direction, while not full MD ready.",
                (
                    "Stage178 L0 local/species endpoint reaches "
                    f"{_fmt_float(delta178.get('l0_atoms_per_second_limit1024'), 0)} atoms/s at limit1024 in the current protocol."
                ),
            ],
            "evidence_limits": [
                "This is not full MD throughput: PBC, Verlet rebuild amortization, virial, integrator, output, and matched community engines remain unaligned.",
                "The result supports the execution topology, not final production performance.",
            ],
            "next_verification": "Promote throughput taxonomy to model-only, graph-amortized, full E/F/V, and full MD; then run matched NEP/DPA1-0-layer baselines.",
        },
        {
            "id": "low_rank_chemistry_front_reduces_fixed_descriptor_bias",
            "question": "Does moving learnable capacity into the species/local L0 front help more than keeping a fixed tiny descriptor and only training a head?",
            "why_required": "The review identified chemical collisions and fixed descriptors as likely causes of poor energy accuracy and weak extrapolation.",
            "status": "supported",
            "evidence": [
                "Stage178 L0 local/species is best by current DFT F RMSE, E RMSE, force max, and throughput among Stage178 variants.",
                "Stage178 L0 improves substantially over Stage176 local_l0_rank3 on energy and force errors under the available audit context.",
            ],
            "evidence_limits": [
                "This does not prove a unique optimal chemistry basis; it only supports trainable low-rank chemistry as a necessary direction.",
                "The evidence is on OC20NEB-style data, not broad MD datasets.",
            ],
            "next_verification": "Run held-out composition/reaction splits and teacher fake-label augmentation to separate architecture bias from data scarcity.",
        },
        {
            "id": "edge_relational_rtece_advantage",
            "question": "Do TECE-specific edge-relational scalar sketches add a useful intermediate layer beyond NEP/MTP-like atomic scalar endpoints?",
            "why_required": "This is the main proposed uniqueness of rTECE over standard NEP/MTP/DPA1-0 endpoints.",
            "status": "contradicted_or_unrealized_current_recipe",
            "evidence": [
                (
                    "Stage178 T3 minimal cavity/direct worsens current F RMSE "
                    f"({_fmt_float(delta178.get('t3_f_rmse_mev_a'))} vs {_fmt_float(delta178.get('l0_f_rmse_mev_a'))}) "
                    "and E RMSE "
                    f"({_fmt_float(delta178.get('t3_e_rmse_mev_atom'))} vs {_fmt_float(delta178.get('l0_e_rmse_mev_atom'))}) "
                    "relative to the L0 local/species endpoint."
                ),
                (
                    "Stage178 T3 is also slower at limit1024 "
                    f"({_fmt_float(delta178.get('t3_atoms_per_second_limit1024'), 0)} vs {_fmt_float(delta178.get('l0_atoms_per_second_limit1024'), 0)} atoms/s)."
                ),
            ],
            "evidence_limits": [
                "This does not falsify edge-relational rTECE in general; it falsifies direct addition under the current training recipe.",
                "The current T3 path lacks teacher-projected initialization and systematic path selection.",
            ],
            "next_verification": "Test edge paths only after semantic projection, cavity/direct separation, and renormalized initialization are available.",
        },
        {
            "id": "operator_space_renormalization_beats_from_scratch",
            "question": "Does Schur/Gauss-Newton/operator-space downfolding outperform from-scratch training for the same retained TECE path set?",
            "why_required": "This is the central claim that the project is a TECE renormalization compiler rather than an ad hoc student architecture search.",
            "status": "missing",
            "evidence": [
                "TECE_design_space.md defines renormalization as retained coefficients plus a Gram/Schur correction from deleted paths.",
                "rTECE_review.md states the current branch has scalar students and benchmarking infrastructure but not true renormalization/distillation.",
            ],
            "evidence_limits": [
                "No current stage proves pruning/downfolded initialization beats from-scratch for a fixed student architecture.",
                "Teacher semantic path projections, Gram blocks, and GN updates are not yet a production training path.",
            ],
            "next_verification": "Stage180 should compare from-scratch, teacher truncation, GN/Schur initialization, and full distillation for the same student path manifest.",
        },
        {
            "id": "sobolev_physics_metric_is_needed",
            "question": "Are E/F RMSE alone insufficient, requiring relative NEB/barrier, force max tail, dimer smoothness, rattle-relax RMSD, and eventually virial/HVP?",
            "why_required": "The objective is accuracy, extrapolation, and physical reasonableness under deployment distributions, not leaderboard MAE alone.",
            "status": "supported_but_incomplete",
            "evidence": [
                "Stage178 records E/F RMSE, E/F max, relative image RMSE, barrier RMSE, dimer, rattle-relax RMSD, throughput, and memory.",
                "The rattle-relax policy is now continuous RMSD/force-tail evidence, not a binary gate.",
            ],
            "evidence_limits": [
                "Virial/stress and Hessian-vector evidence are still missing.",
                "Rattle-relax thresholds remain deployment-task dependent rather than universal.",
            ],
            "next_verification": "Add virial finite-strain tests and HVP or curvature probes before claiming long-time MD reliability.",
        },
        {
            "id": "matched_nep_dpa_protocol",
            "question": "Can rTECE be placed on a fair Pareto curve against NEP/DPA1-0-layer with matched data, hardware scope, metrics, physical tests, and throughput taxonomy?",
            "why_required": "The project target is not just internal improvement; it must show where rTECE sits relative to standard high-throughput potentials.",
            "status": "protocol_mismatched",
            "evidence": [
                "Stage177 explicitly marks non-matching throughput protocols and incomplete physical triage as caveats.",
                "Stage178 repeats that NEP/DPA rows are smoke baselines and cannot support a superiority claim.",
            ],
            "evidence_limits": [
                "Current NEP/DPA rows are useful context, not rankable production baselines.",
                "The available atoms/s numbers mix community engine wall time, prebuilt graph model-only, and ASE/autograd protocols.",
            ],
            "next_verification": "Run matched NEP/DPA1-0-layer training, E/F/max/relative energy metrics, dimer/rattle tests, memory, and throughput scaling.",
        },
        {
            "id": "user_ready_production_model",
            "question": "Is the current rTECE implementation ready for normal users as a reliable potential with training, inference, ASE, PBC, virial, and full MD semantics?",
            "why_required": "A deployable model family must not remain a demo or benchmark-only prototype.",
            "status": "not_ready",
            "evidence": [
                "rTECE_review.md scores PBC/virial/long-time MD completeness and true distillation as early-stage.",
                "Current Stage178 evidence is useful research evidence but not full MD validation.",
            ],
            "evidence_limits": [
                "There is a production training entrypoint and ASE-oriented path, but production reliability requires geometry, stress, and long-run tests.",
            ],
            "next_verification": "Close PBC edge-vector semantics, virial/stress tests, calculator contract, package entrypoints, and a documented user workflow before release.",
        },
    ]

    optional = [
        {
            "id": "triton_or_nvalchemi_kernel_fusion",
            "status": "nice_to_have_after_science_closure",
            "reason": "Kernel fusion can improve throughput, but it does not prove TECE operator-space degradation or renormalized distillation.",
        },
        {
            "id": "multi_gpu_full_md_scaling",
            "status": "nice_to_have_after_single_gpu_protocol_alignment",
            "reason": "Multi-GPU scaling matters for deployment, but only after single-GPU semantics and matched baselines are scientifically clean.",
        },
        {
            "id": "zbl_dispersion_electrostatic_baselines",
            "status": "useful_physics_prior_not_core_proof",
            "reason": "Physical baselines can reduce residual burden, but the central claim is systematic TECE path downfolding.",
        },
    ]

    return {
        "schema_version": "rtece_stage179_scientific_validation.v1",
        "stage": "stage179_scientific_validation_matrix",
        "purpose": "Separate the scientific claims required for the TECE/rTECE renormalization program from useful engineering optimizations.",
        "source_documents": [
            "TECE_design_space.md",
            "rTECE_review.md",
            str(stage178_analysis),
            str(stage177_audit),
        ],
        "rattle_relax_policy": stage178.get("rattle_relax_policy"),
        "stage178_best_current_endpoint": best,
        "required_scientific_hypotheses": required,
        "nice_to_have_optimizations": optional,
        "current_user_readiness": {
            "status": "research_prototype_not_user_ready",
            "usable_for": [
                "research experiments on scalar-sketched TECE student design",
                "controlled training/benchmark/physical-diagnostic runs by developers",
            ],
            "not_yet_usable_for": [
                "ordinary user deployment as a reliable production potential",
                "claims of superiority over production NEP/DPA1-0-layer",
                "long-time periodic MD with validated virial/stress semantics",
            ],
        },
        "stage177_context": context177,
        "stage178_delta": delta178,
        "recommended_next_stage": {
            "id": "stage180_minimal_renormalization_proof",
            "goal": "Prove or falsify that TECE-style renormalized initialization/distillation improves a fixed student path set over from-scratch training.",
            "must_compare": [
                "same student architecture from scratch",
                "teacher/truncated or ordinary supervised initialization",
                "GN or Schur-style renormalized initialization",
                "renormalized initialization plus teacher E/F residual distillation",
            ],
            "must_report": [
                "E RMSE/MAE/max",
                "F RMSE/MAE/max",
                "relative image and barrier RMSE",
                "dimer smoothness",
                "rattle-relax RMSD and force tail",
                "atoms/s and peak allocated/reserved memory under the same protocol",
            ],
        },
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines = [
        "# Stage179 Scientific Validation Matrix",
        "",
        matrix["purpose"],
        "",
        f"Current best Stage178 endpoint: `{matrix['stage178_best_current_endpoint']}`.",
        f"Rattle-relax policy: `{matrix['rattle_relax_policy']}`.",
        "",
        "## Required Scientific Hypotheses",
        "",
        "| id | status | why required | next verification |",
        "|---|---|---|---|",
    ]
    for row in matrix["required_scientific_hypotheses"]:
        lines.append(
            f"| `{row['id']}` | `{row['status']}` | {row['why_required']} | {row['next_verification']} |"
        )
    lines.extend(["", "## Evidence Details", ""])
    for row in matrix["required_scientific_hypotheses"]:
        lines.append(f"### {row['id']}")
        lines.append("")
        lines.append(f"Question: {row['question']}")
        lines.append("")
        lines.append("Evidence:")
        for item in row["evidence"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Evidence limits:")
        for item in row["evidence_limits"]:
            lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Nice-To-Have Optimizations",
            "",
            "| id | status | reason |",
            "|---|---|---|",
        ]
    )
    for row in matrix["nice_to_have_optimizations"]:
        lines.append(f"| `{row['id']}` | `{row['status']}` | {row['reason']} |")
    readiness = matrix["current_user_readiness"]
    lines.extend(
        [
            "",
            "## User Readiness",
            "",
            f"Status: `{readiness['status']}`.",
            "",
            "Usable for:",
        ]
    )
    for item in readiness["usable_for"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Not yet usable for:")
    for item in readiness["not_yet_usable_for"]:
        lines.append(f"- {item}")
    next_stage = matrix["recommended_next_stage"]
    lines.extend(
        [
            "",
            "## Recommended Next Stage",
            "",
            f"`{next_stage['id']}`: {next_stage['goal']}",
            "",
            "Must compare:",
        ]
    )
    for item in next_stage["must_compare"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Must report:")
    for item in next_stage["must_report"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_stage179_validation_matrix(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    stage178_analysis: str | Path = DEFAULT_STAGE178_ANALYSIS,
    stage177_audit: str | Path = DEFAULT_STAGE177_AUDIT,
) -> dict[str, Any]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    matrix = build_stage179_validation_matrix(stage178_analysis=stage178_analysis, stage177_audit=stage177_audit)
    json_path = root / "stage179_validation_matrix.json"
    md_path = root / "stage179_validation_matrix.md"
    json_path.write_text(json.dumps(matrix, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(matrix), encoding="utf-8")
    return {"matrix": matrix, "json_path": str(json_path), "md_path": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage178-analysis", type=Path, default=DEFAULT_STAGE178_ANALYSIS)
    parser.add_argument("--stage177-audit", type=Path, default=DEFAULT_STAGE177_AUDIT)
    args = parser.parse_args()
    result = write_stage179_validation_matrix(
        output_root=args.output_root,
        stage178_analysis=args.stage178_analysis,
        stage177_audit=args.stage177_audit,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "matrix"}, indent=2))


if __name__ == "__main__":
    main()
