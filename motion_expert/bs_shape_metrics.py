"""Aggregate metrics for testing whether generated bones follow shape conditioning."""
from __future__ import annotations

import numpy as np


def _as_matching_matrices(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(x, dtype=np.float64) for x in arrays)
    if not converted or converted[0].ndim != 2:
        raise ValueError("bone-length inputs must be [samples, bones] matrices")
    shape = converted[0].shape
    if any(x.shape != shape for x in converted):
        raise ValueError(f"bone-length shapes must match, got {[x.shape for x in converted]}")
    if not all(np.isfinite(x).all() for x in converted):
        raise ValueError("bone-length inputs must be finite")
    return converted


def _safe_cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denom) if denom > 1e-12 else 0.0


def population_shape_tracking(
    generated_bones_m: np.ndarray,
    target_bones_m: np.ndarray,
) -> dict[str, float | int]:
    """Measure actor-specific shape tracking after removing each bone's population mean.

    Centering per bone prevents the metric from being dominated by the obvious difference between,
    for example, a femur and a finger. A shape-collapsed model has variance ratio/slope near zero;
    ideal conditioning has correlation, slope, and variance ratio near one.
    """
    generated, target = _as_matching_matrices(generated_bones_m, target_bones_m)
    generated_delta = generated - generated.mean(axis=0, keepdims=True)
    target_delta = target - target.mean(axis=0, keepdims=True)
    generated_flat = generated_delta.reshape(-1)
    target_flat = target_delta.reshape(-1)
    target_energy = float(np.dot(target_flat, target_flat))
    generated_energy = float(np.dot(generated_flat, generated_flat))

    per_bone = []
    for bone in range(target.shape[1]):
        target_column = target_delta[:, bone]
        generated_column = generated_delta[:, bone]
        if np.linalg.norm(target_column) > 1e-8 and np.linalg.norm(generated_column) > 1e-8:
            per_bone.append(_safe_cosine(target_column, generated_column))

    return {
        "actor_centered_correlation": _safe_cosine(target_flat, generated_flat),
        "actor_centered_response_slope": (
            float(np.dot(target_flat, generated_flat) / target_energy)
            if target_energy > 1e-12
            else 0.0
        ),
        "actor_centered_variance_ratio": (
            float(np.sqrt(generated_energy / target_energy))
            if target_energy > 1e-12
            else 0.0
        ),
        "actor_centered_mae_cm": float(np.abs(generated_delta - target_delta).mean() * 100.0),
        "per_bone_correlation_mean": float(np.mean(per_bone)) if per_bone else 0.0,
        "num_variable_bones": len(per_bone),
    }


def counterfactual_shape_response(
    generated_original_m: np.ndarray,
    generated_counterfactual_m: np.ndarray,
    target_original_m: np.ndarray,
    target_counterfactual_m: np.ndarray,
) -> dict[str, float | int]:
    """Score a paired same-text/same-noise skeleton intervention.

    Positive target advantage means the counterfactual generation is closer to the requested new
    skeleton than to the original skeleton. Delta slope/magnitude/cosine are ideally one.
    """
    generated_original, generated_cf, target_original, target_cf = _as_matching_matrices(
        generated_original_m,
        generated_counterfactual_m,
        target_original_m,
        target_counterfactual_m,
    )
    requested = target_cf - target_original
    observed = generated_cf - generated_original
    requested_flat = requested.reshape(-1)
    observed_flat = observed.reshape(-1)
    requested_energy = float(np.dot(requested_flat, requested_flat))
    observed_energy = float(np.dot(observed_flat, observed_flat))

    per_case_cosines = []
    for requested_row, observed_row in zip(requested, observed):
        if np.linalg.norm(requested_row) > 1e-8:
            per_case_cosines.append(_safe_cosine(requested_row, observed_row))

    requested_nonzero = np.abs(requested) > 1e-5
    direction_agreement = (
        float(np.mean(np.sign(requested[requested_nonzero]) == np.sign(observed[requested_nonzero])))
        if requested_nonzero.any()
        else 0.0
    )
    target_mae_cm = float(np.abs(generated_cf - target_cf).mean() * 100.0)
    source_mae_cm = float(np.abs(generated_cf - target_original).mean() * 100.0)

    return {
        "requested_bone_delta_cm_mean": float(np.abs(requested).mean() * 100.0),
        "generated_bone_delta_cm_mean": float(np.abs(observed).mean() * 100.0),
        "delta_cosine": _safe_cosine(requested_flat, observed_flat),
        "delta_cosine_per_case_mean": (
            float(np.mean(per_case_cosines)) if per_case_cosines else 0.0
        ),
        "delta_response_slope": (
            float(np.dot(requested_flat, observed_flat) / requested_energy)
            if requested_energy > 1e-12
            else 0.0
        ),
        "delta_magnitude_ratio": (
            float(np.sqrt(observed_energy / requested_energy))
            if requested_energy > 1e-12
            else 0.0
        ),
        "delta_direction_agreement": direction_agreement,
        "counterfactual_target_bone_mae_cm": target_mae_cm,
        "counterfactual_source_bone_mae_cm": source_mae_cm,
        "counterfactual_target_advantage_cm": source_mae_cm - target_mae_cm,
        "num_pairs": int(target_original.shape[0]),
    }


def farthest_shape_indices(target_bones_m: np.ndarray) -> np.ndarray:
    """Return each sample's most different natural skeleton by Euclidean bone distance."""
    target = np.asarray(target_bones_m, dtype=np.float64)
    if target.ndim != 2 or target.shape[0] < 2:
        raise ValueError("at least two [samples, bones] targets are required")
    if not np.isfinite(target).all():
        raise ValueError("target bone lengths must be finite")
    squared_norm = np.einsum("ij,ij->i", target, target)
    distance_squared = (
        squared_norm[:, None] + squared_norm[None, :] - 2.0 * target @ target.T
    )
    np.fill_diagonal(distance_squared, -np.inf)
    return np.argmax(distance_squared, axis=1).astype(np.int64)
