"""
metrics_harness.py

Unified evaluation harness for comparing anomaly detectors (PCA/SVD reconstruction
error, Isolation Forest anomaly score, and later the autoencoder) on the SMAP/MSL
labeled anomaly windows.

Design notes
------------
- Convention: for ALL models fed into this harness, HIGHER score = MORE anomalous.
  - Isolation Forest: if you're using sklearn's `.score_samples()` / `decision_function()`,
    those return LOWER = more anomalous by default. If your "0-1" score already flips
    this (i.e. 1 = anomaly), you're fine. If not, flip it before calling this harness
    (e.g. `score = 1 - raw_score` or `score = -raw_score`).
  - PCA/SVD: reconstruction error is naturally higher = more anomalous, no flip needed.

- Point-adjustment: SMAP/MSL benchmark convention is that if a model flags ANY point
  inside a true anomaly window, the whole window counts as a true positive. This is
  standard for comparing against published benchmarks, but it does inflate recall/F1
  relative to raw point-wise scoring. Both modes are provided — use point-adjusted
  for benchmark comparison, raw point-wise if you want a stricter internal metric.
  Worth stating explicitly in your writeup which one you're reporting.

- Detection latency logging is included now (list of per-event first-detection index)
  so the step-3 harness directly feeds the step-5 early-detection comparison later —
  no need to re-instrument when you get to the autoencoder.
"""

from dataclasses import dataclass, field
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
)


# ---------------------------------------------------------------------------
# Ground truth handling
# ---------------------------------------------------------------------------

def windows_to_binary(n_timesteps: int, windows: list[tuple[int, int]]) -> np.ndarray:
    """
    Convert labeled anomaly windows (list of (start_idx, end_idx), inclusive) into
    a binary ground-truth array of length n_timesteps.
    """
    y_true = np.zeros(n_timesteps, dtype=bool)
    for start, end in windows:
        y_true[start:end + 1] = True
    return y_true


def get_events(y_true: np.ndarray) -> list[tuple[int, int]]:
    """
    Recover contiguous anomaly windows (start_idx, end_idx) from a binary array.
    Useful if you already have a flat boolean array and want event boundaries back
    (e.g. for latency logging).
    """
    events = []
    in_event = False
    start = None
    for i, val in enumerate(y_true):
        if val and not in_event:
            in_event = True
            start = i
        elif not val and in_event:
            in_event = False
            events.append((start, i - 1))
    if in_event:
        events.append((start, len(y_true) - 1))
    return events


# ---------------------------------------------------------------------------
# Point-adjustment
# ---------------------------------------------------------------------------

def point_adjust(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Standard SMAP/MSL-style point adjustment: if ANY point within a true anomaly
    window is predicted positive, mark the ENTIRE window as predicted positive.
    Points outside true windows are left untouched.
    """
    y_adj = y_pred.copy()
    for start, end in get_events(y_true):
        if y_pred[start:end + 1].any():
            y_adj[start:end + 1] = True
    return y_adj


# ---------------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------------

@dataclass
class ThresholdSweepResult:
    thresholds: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    best_threshold: float
    best_f1: float


def sweep_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 200,
    use_point_adjust: bool = True,
) -> ThresholdSweepResult:
    """
    Sweep a range of thresholds over `scores`, computing precision/recall/F1 at each.
    Returns the full sweep plus the best (max-F1) operating point.
    """
    lo, hi = np.percentile(scores, [0.5, 99.5])
    thresholds = np.linspace(lo, hi, n_thresholds)

    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = scores >= t
        if use_point_adjust:
            y_pred = point_adjust(y_true, y_pred)

        tp = np.sum(y_pred & y_true)
        fp = np.sum(y_pred & ~y_true)
        fn = np.sum(~y_pred & y_true)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    precisions, recalls, f1s = map(np.array, (precisions, recalls, f1s))
    best_idx = int(np.argmax(f1s))

    return ThresholdSweepResult(
        thresholds=thresholds,
        precision=precisions,
        recall=recalls,
        f1=f1s,
        best_threshold=float(thresholds[best_idx]),
        best_f1=float(f1s[best_idx]),
    )


# ---------------------------------------------------------------------------
# AUC metrics (threshold-independent)
# ---------------------------------------------------------------------------

@dataclass
class AUCResult:
    auc_roc: float
    auc_pr: float


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> AUCResult:
    """
    Threshold-independent summary metrics. Computed on RAW point-wise labels
    (point-adjustment doesn't apply here — it's specific to threshold sweeps).
    AUC-PR is generally the more informative of the two for anomaly detection,
    since anomalies are rare and AUC-ROC can look artificially good.
    """
    return AUCResult(
        auc_roc=roc_auc_score(y_true, scores),
        auc_pr=average_precision_score(y_true, scores),
    )


# ---------------------------------------------------------------------------
# Detection latency (feeds the step-5 early-detection comparison)
# ---------------------------------------------------------------------------

@dataclass
class EventDetection:
    event_start: int
    event_end: int
    detected: bool
    latency: int | None  # timesteps from event_start to first detection, None if missed


def detection_latencies(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> list[EventDetection]:
    """
    For each true anomaly event, find the first timestep (relative to event start)
    where `scores >= threshold`. This is the core log you'll reuse across models
    for the early-detection comparison in step 5 — run this once per model at its
    own best threshold and compare `latency` distributions.
    """
    y_pred = scores >= threshold
    results = []
    for start, end in get_events(y_true):
        window_pred = y_pred[start:end + 1]
        hit_indices = np.where(window_pred)[0]
        if len(hit_indices) > 0:
            results.append(EventDetection(start, end, True, int(hit_indices[0])))
        else:
            results.append(EventDetection(start, end, False, None))
    return results


# ---------------------------------------------------------------------------
# Top-level convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class ModelEvaluation:
    model_name: str
    sweep: ThresholdSweepResult
    auc: AUCResult
    latencies: list[EventDetection]


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = 200,
    use_point_adjust: bool = True,
) -> ModelEvaluation:
    """
    Run the full evaluation pipeline for one model's scores: threshold sweep,
    AUC metrics, and per-event detection latency at the best F1 threshold.
    Call this once per model (PCA/SVD, Isolation Forest, later autoencoder)
    with a consistent y_true to get directly comparable results.
    """
    sweep = sweep_thresholds(y_true, scores, n_thresholds, use_point_adjust)
    auc = compute_auc(y_true, scores)
    latencies = detection_latencies(y_true, scores, sweep.best_threshold)

    return ModelEvaluation(
        model_name=model_name,
        sweep=sweep,
        auc=auc,
        latencies=latencies,
    )


def print_summary(evaluation: ModelEvaluation) -> None:
    ev = evaluation
    n_events = len(ev.latencies)
    n_detected = sum(1 for e in ev.latencies if e.detected)
    print(f"--- {ev.model_name} ---")
    print(f"Best threshold: {ev.sweep.best_threshold:.4f}")
    print(f"Best F1: {ev.sweep.best_f1:.4f}")
    print(f"AUC-ROC: {ev.auc.auc_roc:.4f}")
    print(f"AUC-PR: {ev.auc.auc_pr:.4f}")
    print(f"Events detected: {n_detected}/{n_events}")
    hit_latencies = [e.latency for e in ev.latencies if e.detected]
    if hit_latencies:
        print(f"Mean detection latency: {np.mean(hit_latencies):.2f} timesteps")


# ---------------------------------------------------------------------------
# Multi-channel support (real SMAP/MSL labeled_anomalies.csv)
# ---------------------------------------------------------------------------

import ast
import pandas as pd


def load_labeled_anomalies(csv_path: str) -> pd.DataFrame:
    """
    Load the real NASA labeled_anomalies.csv (chan_id, spacecraft,
    anomaly_sequences, class, num_values) and parse the stringified list
    columns into actual Python lists.
    """
    df = pd.read_csv(csv_path)
    df["anomaly_sequences"] = df["anomaly_sequences"].apply(ast.literal_eval)
    df["class"] = df["class"].apply(
        lambda s: [c.strip() for c in s.strip("[]").split(",")]
    )
    return df


def channel_y_true(df: pd.DataFrame, chan_id: str) -> np.ndarray:
    """Build the binary ground-truth array for a single channel."""
    row = df.loc[df["chan_id"] == chan_id].iloc[0]
    return windows_to_binary(int(row["num_values"]), row["anomaly_sequences"])


@dataclass
class MultiChannelEvaluation:
    model_name: str
    per_channel: dict  # chan_id -> ModelEvaluation
    global_precision: float
    global_recall: float
    global_f1: float
    by_class: dict  # 'point'/'contextual' -> {'n_events': int, 'n_detected': int}


def evaluate_multichannel(
    model_name: str,
    scores_by_channel: dict,  # chan_id -> np.ndarray of scores, one per channel
    df: pd.DataFrame,
    use_point_adjust: bool = True,
) -> MultiChannelEvaluation:
    """
    Run evaluate_model() per channel, then pool TP/FP/FN across ALL channels for
    a single global precision/recall/F1 (the convention used in the SMAP/MSL
    benchmark literature) — NOT a plain average of per-channel F1 scores, which
    would over-weight short/sparse channels.

    scores_by_channel must have one entry per chan_id you want evaluated, each
    an array the same length as that channel's num_values.
    """
    per_channel = {}
    total_tp = total_fp = total_fn = 0
    class_stats = {"point": {"n_events": 0, "n_detected": 0},
                    "contextual": {"n_events": 0, "n_detected": 0}}

    for chan_id, scores in scores_by_channel.items():
        row = df.loc[df["chan_id"] == chan_id].iloc[0]
        y_true = channel_y_true(df, chan_id)

        ev = evaluate_model(f"{model_name}:{chan_id}", y_true, scores,
                             use_point_adjust=use_point_adjust)
        per_channel[chan_id] = ev

        y_pred = point_adjust(y_true, scores >= ev.sweep.best_threshold) \
            if use_point_adjust else (scores >= ev.sweep.best_threshold)
        total_tp += np.sum(y_pred & y_true)
        total_fp += np.sum(y_pred & ~y_true)
        total_fn += np.sum(~y_pred & y_true)

        # per-event class breakdown (point vs contextual)
        windows = row["anomaly_sequences"]
        classes = row["class"]
        for (start, end), cls in zip(windows, classes):
            class_stats[cls]["n_events"] += 1
            if y_pred[start:end + 1].any():
                class_stats[cls]["n_detected"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return MultiChannelEvaluation(
        model_name=model_name,
        per_channel=per_channel,
        global_precision=float(precision),
        global_recall=float(recall),
        global_f1=float(f1),
        by_class=class_stats,
    )


def print_multichannel_summary(evaluation: MultiChannelEvaluation) -> None:
    ev = evaluation
    print(f"=== {ev.model_name} (global, {len(ev.per_channel)} channels) ===")
    print(f"Global precision: {ev.global_precision:.4f}")
    print(f"Global recall:    {ev.global_recall:.4f}")
    print(f"Global F1:        {ev.global_f1:.4f}")
    for cls, stats in ev.by_class.items():
        n, d = stats["n_events"], stats["n_detected"]
        rate = d / n if n > 0 else 0.0
        print(f"  {cls:>10}: {d}/{n} events detected ({rate:.1%})")


if __name__ == "__main__":
    # Minimal smoke test with synthetic data
    rng = np.random.default_rng(0)
    n = 1000
    y_true = np.zeros(n, dtype=bool)
    y_true[200:220] = True
    y_true[600:640] = True

    # Fake scores: mostly noise, elevated inside true windows with some delay
    scores = rng.normal(0, 1, n)
    scores[205:225] += rng.normal(5, 1, 20)   # detected a bit late
    scores[600:610] += rng.normal(5, 1, 10)   # detected right away, then drops off

    result = evaluate_model("SmokeTest-PCA", y_true, scores)
    print_summary(result)