"""Allowlisted project catalog and evidence readers for the browser workspace."""
from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = Path(__file__).with_name("ui") / "project_catalog.json"
ROOT_RESOLVED = ROOT.resolve()


def _safe_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    try:
        path.relative_to(ROOT_RESOLVED)
    except ValueError as exc:
        raise FileNotFoundError(relative_path) from exc
    return path


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def list_evidence() -> list[dict[str, Any]]:
    rows = []
    for item in load_catalog()["evidence"]:
        path = _safe_path(item["path"])
        rows.append({
            **{key: item[key] for key in ("id", "name", "kind", "dataType")},
            "available": path.is_file(),
            "sourceUrl": f"/api/project/evidence/{item['id']}/source",
        })
    return rows


def list_media() -> list[dict[str, Any]]:
    rows = []
    for item in load_catalog().get("media", []):
        path = _safe_path(item["path"])
        rows.append({
            **{key: item[key] for key in ("id", "name", "stage", "description")},
            "available": path.is_file(),
            "url": f"/api/project/media/{item['id']}",
        })
    return rows


def _entry(section: str, entry_id: str) -> dict[str, Any]:
    for item in load_catalog()[section]:
        if item["id"] == entry_id:
            return item
    raise KeyError(entry_id)


def source_path(source_id: str) -> tuple[dict[str, Any], Path]:
    item = _entry("sources", source_id)
    return item, _safe_path(item["path"])


def evidence_path(evidence_id: str) -> tuple[dict[str, Any], Path]:
    item = _entry("evidence", evidence_id)
    return item, _safe_path(item["path"])


def media_path(media_id: str) -> tuple[dict[str, Any], Path]:
    item = _entry("media", media_id)
    if Path(item["path"]).suffix.lower() not in (".mp4", ".webm"):
        raise FileNotFoundError(media_id)
    return item, _safe_path(item["path"])


def _public_json(value: Any) -> Any:
    """Filter machine-local paths and checkpoint locations from saved reports."""
    private_names = ("path", "checkpoint", "outdir", "run_root", "filename")
    if isinstance(value, dict):
        return {
            key: _public_json(item)
            for key, item in value.items()
            if not any(private in key.lower() for private in private_names)
        }
    if isinstance(value, list):
        return [_public_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and re.match(r"^(?:[A-Za-z]:[\\/]|[/\\]{2}|/Users/|/home/)", value):
        return "[machine-local path omitted]"
    return value


def _goal_hold_payload(data: dict[str, Any]) -> dict[str, Any]:
    cfg = data.get("config", {})
    summary = data.get("summary", {})
    rows = data.get("rows", [])
    # Reduce the raw report to the fields needed by charts and tables. In
    # particular, omit machine-specific checkpoint paths.
    row_fields = (
        "controller", "seed", "integrated_resonator_energy_mJs",
        "position_rmse_mm", "angle_rms_deg", "peak_angle_deg",
        "peak_rail_fraction", "exclusion_zone_violations", "goal_count",
        "goal_targets_mm", "goal_targets_respect_margin",
        "completed_two_second_holds", "two_second_hold_fraction",
        "longest_contiguous_in_tolerance_hold_s", "mpc_fallback_fraction",
        "mpc_solve_p95_ms", "terminated",
    )
    return {
        "benchmarkVersion": data.get("benchmark_version"),
        "sourceRevision": data.get("source_revision"),
        "holdToleranceMm": data.get("hold_tolerance_mm"),
        "task": {
            "referenceMode": cfg.get("reference_mode"),
            "railHalfTravelMm": 1000 * 0.065,
            "edgeExclusionMm": 1000 * float(cfg.get("goal_edge_margin_m", 0.0)),
            "targetBufferMm": 1000 * float(cfg.get("goal_target_buffer_m", 0.0)),
            "goalMoveSeconds": cfg.get("goal_move_seconds"),
            "configuredMinimumHoldSeconds": cfg.get("goal_hold_min_seconds"),
            "holdSuccessSeconds": 2.0,
            "holdOnlyEnergyReward": cfg.get("goal_energy_gate_hold_only"),
            "sensorTiming": cfg.get("sensor_timing", {}),
        },
        "summary": summary,
        "promotion": data.get("promotion", {}),
        "rows": [{key: row.get(key) for key in row_fields if key in row} for row in rows],
    }


def _physics_payload(data: dict[str, Any]) -> dict[str, Any]:
    spring = data.get("spring_model_comparison", {})
    convergence = data.get("timestep_convergence", {})
    models = []
    for model_name, model in convergence.get("models", {}).items():
        scenarios = []
        for scenario_name, scenario in model.get("scenarios", {}).items():
            steps = [{
                "physicsDtSeconds": step.get("physics_dt_s"),
                "maxNormalizedStateErrorRms": step.get("max_normalized_state_error_rms"),
                "maxRelativeEnergyDrift": step.get("max_relative_energy_drift"),
                "finalAbsoluteStateError": step.get("final_absolute_state_error", {}),
            } for step in scenario.get("steps", [])]
            scenarios.append({"name": scenario_name, "steps": steps})
        models.append({
            "name": model_name,
            "springMode": model.get("spring_mode"),
            "validatedMaxPhysicsDtSeconds": model.get("validated_max_physics_dt_s"),
            "scenarios": scenarios,
        })
    return {
        "springModelComparison": {
            key: spring.get(key)
            for key in ("geometry_parameters", "geometric_small_angle_stiffness_nm_per_rad",
                        "equivalent_small_angle_stiffness_nm_per_rad", "torque_comparison")
            if key in spring
        },
        "timestepConvergence": {
            "durationSeconds": convergence.get("duration_s"),
            "referenceDtSeconds": convergence.get("reference_dt_s"),
            "acceptanceLimits": convergence.get("acceptance_limits", {}),
            "models": models,
        },
    }


def get_evidence(evidence_id: str) -> tuple[dict[str, Any], Any]:
    item, path = evidence_path(evidence_id)
    if not path.is_file():
        raise FileNotFoundError(item["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if item["dataType"] == "goal-hold":
        data = _goal_hold_payload(data)
    elif item["dataType"] == "physics":
        data = _physics_payload(data)
    else:
        data = _public_json(data)
    return item, data


def public_evidence_source(evidence_id: str) -> tuple[dict[str, Any], str]:
    item, path = evidence_path(evidence_id)
    if not path.is_file():
        raise FileNotFoundError(item["path"])
    data = json.loads(path.read_text(encoding="utf-8"))
    if item["dataType"] == "goal-hold":
        data = _goal_hold_payload(data)
    data = _public_json(data)
    return item, json.dumps(data, indent=2, allow_nan=False)
