#!/usr/bin/env python3
"""Verify the complete revised-paper evidence package.

The default mode checks every declared raw file by exact byte count and checks
all compact/processed evidence by content.  ``--full-raw-hash`` additionally
recomputes SHA-256 for the 11.6 GB EAGLE-I archive and all other raw files.
The machine-readable JSON report is the authoritative verification record.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        passed: bool,
        detail: str,
        artifact: str | Path | None = None,
    ) -> None:
        record: dict[str, Any] = {
            "id": check_id,
            "status": "PASS" if passed else "FAIL",
            "detail": detail,
        }
        if artifact is not None:
            path = Path(artifact)
            if path.is_absolute():
                try:
                    artifact_text = path.resolve().relative_to(ROOT).as_posix()
                except ValueError:
                    artifact_text = "../" + path.resolve().relative_to(ROOT.parent).as_posix()
            else:
                artifact_text = path.as_posix()
            record["artifact"] = artifact_text
        self.checks.append(record)

    def require_file(self, check_id: str, relative_path: str) -> Path:
        path = ROOT / relative_path
        self.add(
            check_id,
            path.is_file() and path.stat().st_size > 0,
            f"exists={path.is_file()}; bytes={path.stat().st_size if path.is_file() else 0}",
            path,
        )
        return path

    @property
    def passed(self) -> bool:
        return all(item["status"] == "PASS" for item in self.checks)


def raw_manifest_records() -> list[dict[str, Any]]:
    boulder = json.loads(
        (ROOT / "data/external/metadata/boulder_ev_sessions_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    eaglei = json.loads(
        (ROOT / "data/external/metadata/eaglei_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    smartds = json.loads(
        (
            ROOT
            / "data/external/metadata/smartds_v1.0_sfo_p1u_dt104_manifest.json"
        ).read_text(encoding="utf-8")
    )
    records = [
        {
            "dataset": "Boulder EV",
            "path": boulder["file"]["path"],
            "bytes": boulder["file"]["bytes"],
            "sha256": boulder["file"]["sha256"],
        }
    ]
    records.extend(
        {
            "dataset": "EAGLE-I",
            "path": item["path"],
            "bytes": item["bytes"],
            "sha256": item["sha256"],
        }
        for item in eaglei["files"]
    )
    records.extend(
        {
            "dataset": "SMART-DS",
            "path": item["path"],
            "bytes": item["size"],
            "sha256": item["sha256"],
        }
        for item in smartds["files"]
    )
    return records


def verify_raw_files(audit: Audit, full_hash: bool) -> dict[str, Any]:
    records = raw_manifest_records()
    dataset_counts: dict[str, int] = {}
    dataset_bytes: dict[str, int] = {}
    size_failures: list[str] = []
    hash_failures: list[str] = []
    for item in records:
        path = ROOT / item["path"]
        dataset = item["dataset"]
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        dataset_bytes[dataset] = dataset_bytes.get(dataset, 0) + int(item["bytes"])
        if not path.is_file() or path.stat().st_size != int(item["bytes"]):
            size_failures.append(item["path"])
            continue
        if full_hash and sha256_file(path) != item["sha256"]:
            hash_failures.append(item["path"])

    audit.add(
        "raw_file_size_manifest",
        not size_failures,
        f"files={len(records)}; failures={len(size_failures)}",
        "data/external/metadata",
    )
    if full_hash:
        audit.add(
            "raw_file_sha256_manifest",
            not hash_failures and not size_failures,
            f"files_hashed={len(records) - len(size_failures)}; failures={len(hash_failures)}",
            "data/external/metadata",
        )
    return {
        "mode": "full_sha256" if full_hash else "exact_bytes",
        "files": len(records),
        "bytes": sum(dataset_bytes.values()),
        "dataset_file_counts": dataset_counts,
        "dataset_bytes": dataset_bytes,
        "size_failures": size_failures,
        "hash_failures": hash_failures,
    }


def verify_boulder_conservation(audit: Audit) -> None:
    base = ROOT / "data/external/processed/boulder_ev"
    report_path = base / "boulder_ev_quality_report.json"
    hourly_path = base / "boulder_ev_hourly.npz"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    hourly = np.load(hourly_path, allow_pickle=False)
    conserved = report["conservation_checks"]
    arrivals = int(hourly["arrivals"].sum(dtype=np.int64))
    start_energy = float(hourly["transaction_start_energy_kwh"].sum(dtype=np.float64))
    reconstructed = float(hourly["reconstructed_load_kwh"].sum(dtype=np.float64))
    measured = float(conserved["measured_positive_energy_kwh"])
    audit.add(
        "boulder_arrival_conservation",
        arrivals == int(conserved["arrival_records"]) == int(conserved["hourly_arrivals"]),
        f"source={conserved['arrival_records']}; hourly_tensor={arrivals}",
        hourly_path,
    )
    audit.add(
        "boulder_energy_conservation",
        abs(start_energy - measured) < 0.5 and abs(reconstructed - measured) < 0.5,
        (
            f"measured={measured:.6f} kWh; transaction_start={start_energy:.6f} kWh; "
            f"reconstructed_proxy={reconstructed:.6f} kWh; tolerance=0.5 kWh"
        ),
        report_path,
    )


def verify_scenario_counts(audit: Audit) -> None:
    expected = {
        "robust_rollout_rows": (
            "results/operational/boulder_robust_rollout/rollout_scenarios.csv",
            24_576,
        ),
        "smartds_action_loop_rows": (
            "results/operational/smartds_ev/smartds_ev_scenarios.csv",
            1_920,
        ),
        "smartds_long_horizon_rows": (
            "results/operational/smartds_ev_long_horizon/smartds_ev_scenarios.csv",
            14_902,
        ),
        "packet_network_rows": (
            "results/operational/packet_network/packet_network_scenarios.csv",
            12_000,
        ),
        "station_choice_rows": (
            "results/operational/station_choice/station_choice_scenarios.csv",
            2_592,
        ),
        "crew_routing_rows": (
            "results/operational/crew_routing_128/crew_routing_scenarios.csv",
            6_912,
        ),
        "integrated_stress_rows": (
            "results/operational/boulder_integrated_execution_stress/rollout_scenarios.csv",
            6_144,
        ),
    }
    for check_id, (relative_path, expected_rows) in expected.items():
        path = ROOT / relative_path
        rows = len(pd.read_csv(path)) if path.is_file() else -1
        audit.add(
            check_id,
            rows == expected_rows,
            f"expected={expected_rows}; observed={rows}",
            path,
        )


def verify_invariants(audit: Audit) -> None:
    path = ROOT / "results/operational/proof_checks/proof_check_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    checks = report.get("checks", [])
    passed = report.get("status") == "PASS" and len(checks) == 11 and all(
        item.get("status") == "PASS" for item in checks
    )
    audit.add(
        "implementation_invariants",
        passed,
        f"report={report.get('status')}; passed={sum(c.get('status') == 'PASS' for c in checks)}/{len(checks)}",
        path,
    )


def verify_integrated_execution(audit: Audit) -> None:
    primary_path = ROOT / "results/operational/boulder_robust_rollout/rollout_scenarios.csv"
    stress_path = ROOT / "results/operational/boulder_integrated_execution_stress/rollout_scenarios.csv"
    primary = pd.read_csv(primary_path)
    stress = pd.read_csv(stress_path)

    policies_per_scenario = primary.groupby("scenario_id")["policy"].nunique()
    mapping_per_scenario = primary.groupby("scenario_id")["smartds_mapping_index"].nunique()
    jobs_per_scenario = primary.groupby("scenario_id")["crew_job_count"].nunique()
    event_counts = primary["crew_completion_events"].fillna("").map(
        lambda value: sum(
            float(token.rsplit("@", 1)[1]) <= 6.0 + 1e-12
            for token in value.split(";")
            if token and "@" in token
        )
    )
    routed_counts = primary["crew_routes"].fillna("").map(lambda value: value.count("@"))
    audit.add(
        "integrated_primary_pairing",
        (
            len(primary) == 24_576
            and primary["scenario_id"].nunique() == 4_096
            and policies_per_scenario.eq(6).all()
            and mapping_per_scenario.eq(1).all()
            and jobs_per_scenario.eq(1).all()
        ),
        (
            f"rows={len(primary)}; scenarios={primary['scenario_id'].nunique()}; "
            f"six_policy_failures={int((policies_per_scenario != 6).sum())}; "
            f"mapping_pair_failures={int((mapping_per_scenario != 1).sum())}; "
            f"job_pair_failures={int((jobs_per_scenario != 1).sum())}"
        ),
        primary_path,
    )
    audit.add(
        "integrated_primary_online_ac",
        (
            int(primary["smartds_projected_infeasible_hours"].sum()) == 0
            and primary["mean_smartds_projection_fraction"].between(0, 1).all()
        ),
        (
            f"online_policy_hours={len(primary) * 6}; "
            f"raw_infeasible_hours={int(primary['smartds_raw_infeasible_hours'].sum())}; "
            f"projected_infeasible_hours={int(primary['smartds_projected_infeasible_hours'].sum())}"
        ),
        primary_path,
    )
    audit.add(
        "integrated_primary_crew_events",
        (
            primary["crew_jobs_completed"].le(primary["crew_jobs_dispatched"]).all()
            and primary["crew_jobs_dispatched"].le(primary["crew_job_count"]).all()
            and event_counts.eq(primary["crew_jobs_completed"]).all()
            and routed_counts.eq(primary["crew_jobs_dispatched"]).all()
        ),
        (
            f"policy_scenarios={len(primary)}; "
            f"completed={int(primary['crew_jobs_completed'].sum())}; "
            f"dispatched={int(primary['crew_jobs_dispatched'].sum())}"
        ),
        primary_path,
    )
    audit.add(
        "integrated_stress_feedback",
        (
            int(stress["smartds_raw_infeasible_hours"].sum()) > 0
            and int(stress["smartds_projected_infeasible_hours"].sum()) == 0
            and float(stress["smartds_curtailed_energy_kwh"].sum()) > 0
            and float(stress["mean_smartds_projection_fraction"].mean()) < 1
        ),
        (
            f"online_policy_hours={len(stress) * 6}; "
            f"raw_infeasible_hours={int(stress['smartds_raw_infeasible_hours'].sum())}; "
            f"projected_infeasible_hours={int(stress['smartds_projected_infeasible_hours'].sum())}; "
            f"curtailed_kwh={float(stress['smartds_curtailed_energy_kwh'].sum()):.3f}"
        ),
        stress_path,
    )


def verify_required_artifacts(audit: Audit) -> None:
    required = {
        "central_paired_statistics": "results/operational/boulder_robust_rollout/paired_statistics_central.csv",
        "robust_paired_statistics": "results/operational/boulder_robust_rollout/paired_statistics_robust.csv",
        "robust_central_risk_audit": "results/operational/boulder_robust_rollout/robust_vs_central_risk.csv",
        "central_objective_sensitivity": "results/operational/boulder_robust_rollout/objective_weight_sensitivity_central.csv",
        "packet_feedback_statistics": "results/operational/packet_feedback_statistics_central.csv",
        "integrated_execution_summary": "results/operational/integrated_execution_summary.csv",
        "integrated_execution_manifest": "results/operational/integrated_execution_summary_manifest.json",
        "integrated_smartds_mappings": "results/operational/boulder_robust_rollout/station_to_smartds_load_mappings.csv",
        "integrated_stress_manifest": "results/operational/boulder_integrated_execution_stress/rollout_manifest.csv",
        "integrated_sensitivity_summary": "results/revision_sensitivity/sensitivity_summary.csv",
        "crew_routing_statistics": "results/operational/crew_routing_128/crew_routing_paired_statistics.csv",
        "calibration_uncertainty": "results/operational/transition_calibration/transition_parameter_uncertainty.csv",
        "forecast_baselines": "results/real_ev/forecast_table.csv",
        "inductive_spatial_holdout": "results/real_ev_spatial_holdout_inductive/spatial_holdout_metrics.csv",
        "figure_system_png": "paper/fig_executable_system.png",
        "figure_evidence_pdf": "paper/fig_operational_evidence.pdf",
        "figure_evidence_svg": "paper/fig_operational_evidence.svg",
        "figure_evidence_tiff": "paper/fig_operational_evidence.tiff",
        "figure_source_data": "paper/fig_operational_evidence_source_data.csv",
        "compiled_manuscript": "paper/main.pdf",
        "compiled_marked_manuscript": "paper/main_marked.pdf",
        "revision_reproducibility_guide": "REVISION_REPRODUCIBILITY.md",
        "third_party_data_licenses": "data/external/THIRD_PARTY_DATA_LICENSES.md",
        "response_pdf": "../revision/response_to_reviewers.pdf",
        "response_docx": "../revision/response_to_reviewers.docx",
        "chinese_assessment_pdf": "../revision/assess.pdf",
        "revision_cover_letter_pdf": "../revision/revision_cover_letter.pdf",
    }
    for check_id, relative_path in required.items():
        audit.require_file(check_id, relative_path)


def verify_correspondence(audit: Audit) -> None:
    response = ROOT.parent / "revision/response_to_reviewers.md"
    response_text = response.read_text(encoding="utf-8")
    ids = re.findall(r"^### ([ER][0-9.]+)\b", response_text, flags=re.MULTILINE)
    audit.add(
        "response_42_unique_items",
        len(ids) == 42 and len(set(ids)) == 42,
        f"items={len(ids)}; unique={len(set(ids))}",
        response,
    )
    reviewer_ids = [item for item in ids if item.startswith("R")]
    reviewer_quotes = re.findall(r"^> ", response_text, flags=re.MULTILINE)
    audit.add(
        "response_reviewer_comment_coverage",
        len(reviewer_ids) == 35 and len(reviewer_quotes) == 42,
        f"reviewer_items={len(reviewer_ids)}; total_comment_quotes={len(reviewer_quotes)}",
        response,
    )
    assess = ROOT.parent / "revision/assess_cn.md"
    assess_text = assess.read_text(encoding="utf-8")
    assessed = re.findall(
        r"^\*\*([ER][0-9.]+)｜[^\n]+\*\*$", assess_text, flags=re.MULTILINE
    )
    verdicts = len(
        re.findall(r"^\*\*当前内部判断：A\+?。\*\*", assess_text, flags=re.MULTILINE)
    )
    audit.add(
        "chinese_assessment_42_items",
        len(assessed) == 42 and verdicts == 42 and set(assessed) == set(ids),
        (
            f"assessed={len(assessed)}; verdicts={verdicts}; "
            f"matches_response={set(assessed) == set(ids)}"
        ),
        assess,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-raw-hash",
        action="store_true",
        help="Recompute SHA-256 for all raw data (about 11.7 GB).",
    )
    parser.add_argument(
        "--output",
        default="results/operational/revision_package_verification.json",
        help="JSON report path relative to the project root.",
    )
    args = parser.parse_args()

    audit = Audit()
    raw_summary = verify_raw_files(audit, args.full_raw_hash)
    verify_boulder_conservation(audit)
    verify_scenario_counts(audit)
    verify_invariants(audit)
    verify_integrated_execution(audit)
    verify_required_artifacts(audit)
    verify_correspondence(audit)

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "PASS" if audit.passed else "FAIL",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(ROOT),
        "full_raw_hash": bool(args.full_raw_hash),
        "raw_data": raw_summary,
        "summary": {
            "passed": sum(item["status"] == "PASS" for item in audit.checks),
            "failed": sum(item["status"] == "FAIL" for item in audit.checks),
            "total": len(audit.checks),
        },
        "checks": audit.checks,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(f"status={result['status']}; report={output_path.relative_to(ROOT)}")
    return 0 if audit.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
