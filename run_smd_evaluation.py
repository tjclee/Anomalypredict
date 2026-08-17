

import numpy as np

from FileHandler import sliding_windows
from Models import PCAReconstructionDetector, IsolationForestDetector
from metric_harness import evaluate_model, print_summary


def load_smd_series(path: str) -> np.ndarray:
    """Load a comma-separated, headerless SMD file -> (n_timesteps, n_features)."""
    return np.loadtxt(path, delimiter=",")


def load_smd_labels(path: str) -> np.ndarray:
    """Load the SMD test_label file (one 0/1 per line) -> boolean array."""
    return np.loadtxt(path, delimiter=",").astype(bool)


def map_window_scores_to_timesteps(
    window_scores: np.ndarray,
    end_indices: np.ndarray,
    window_size: int,
    n_timesteps: int,
    mode: str = "max",
) -> np.ndarray:
    """
    sliding_windows() returns end_indices (last timestep of each window), so
    each window covers [end - window_size + 1, end]. Map window scores back
    to per-timestep scores by that span.

    mode="max": a timestep takes the highest score of any window covering
    it -- i.e. if ANY window flags that region as anomalous, the timestep
    reflects that (matches "detect it as early/reliably as possible").
    mode="mean" averages instead, for a smoother/more conservative signal.
    """
    starts = end_indices - window_size + 1
    sums = np.zeros(n_timesteps)
    counts = np.zeros(n_timesteps)
    maxes = np.full(n_timesteps, -np.inf)

    for score, start, end in zip(window_scores, starts, end_indices):
        if mode == "max":
            maxes[start:end + 1] = np.maximum(maxes[start:end + 1], score)
        else:
            sums[start:end + 1] += score
            counts[start:end + 1] += 1

    if mode == "max":
        uncovered = np.isneginf(maxes)
        if uncovered.any():
            maxes[uncovered] = np.min(window_scores)
        return maxes
    else:
        counts[counts == 0] = 1
        return sums / counts


def run(
    train_path: str,
    test_path: str,
    test_label_path: str,
    window_size: int = 100,
    stride: int = 1,
    pca_n_components: int = 5,
):
    print("Loading data...")
    train_data = load_smd_series(train_path)
    test_data = load_smd_series(test_path)
    y_true = load_smd_labels(test_label_path)
    print(f"train: {train_data.shape}, test: {test_data.shape}, labels: {y_true.shape}")
    assert test_data.shape[0] == y_true.shape[0], (
        "test data and test_label must have the same number of rows"
    )

    print(f"\nWindowing (window_size={window_size}, stride={stride})...")
    train_windows, _ = sliding_windows(train_data, window_size, stride)
    test_windows, test_end_indices = sliding_windows(test_data, window_size, stride)
    print(f"train windows: {train_windows.shape}, test windows: {test_windows.shape}")

    models = {
        "PCA/SVD": PCAReconstructionDetector(n_components=pca_n_components),
        "Isolation Forest": IsolationForestDetector(random_state=0),
    }

    results = {}
    for name, model in models.items():
        print(f"\nFitting {name} on SMD train windows...")
        model.fit(train_windows)

        print(f"Scoring {name} on SMD test windows...")
        window_scores = model.score(test_windows)

        timestep_scores = map_window_scores_to_timesteps(
            window_scores, test_end_indices, window_size, len(y_true), mode="max"
        )

        eval_result = evaluate_model(name, y_true, timestep_scores)
        print_summary(eval_result)
        results[name] = eval_result

    print("\n=== AUC-PR head-to-head ===")
    for name, ev in results.items():
        print(f"{name}: {ev.auc.auc_pr:.4f}")

    return results


# ---------------------------------------------------------------------------
# Multi-machine aggregation
# ---------------------------------------------------------------------------

import os
from dataclasses import dataclass
from metric_harness import point_adjust, compute_auc


def discover_machines(train_dir: str, test_dir: str, label_dir: str) -> list:
    """
    Find machine ids (e.g. 'machine-1-1') present in all three directories,
    matched by identical filename. Returns a sorted list of ids so results
    are reproducible run to run.
    """
    def stems(d):
        return {os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".txt")}

    ids = stems(train_dir) & stems(test_dir) & stems(label_dir)
    missing_train = stems(test_dir) - stems(train_dir)
    missing_label = stems(test_dir) - stems(label_dir)
    if missing_train:
        print(f"Skipping (no train file): {sorted(missing_train)}")
    if missing_label:
        print(f"Skipping (no label file): {sorted(missing_label)}")
    return sorted(ids)


@dataclass
class MultiMachineResult:
    model_name: str
    per_machine: dict          # machine_id -> ModelEvaluation
    global_precision: float
    global_recall: float
    global_f1: float
    global_auc_roc: float
    global_auc_pr: float
    n_events_total: int
    n_events_detected: int
    mean_latency: float        # mean timesteps-to-first-detection, over DETECTED events only
    median_latency: float


def run_multi_machine(
    train_dir: str = None,
    test_dir: str = None,
    label_dir: str = None,
    train_path: str = None,
    test_path: str = None,
    test_label_path: str = None,
    machine_ids: list = None,
    window_size: int = 100,
    stride: int = 1,
    pca_n_components: int = 5,
    use_point_adjust: bool = True,
    verbose: bool = True,
):
    """
    Run the full pipeline independently on each SMD machine (fresh model fit
    per machine, matching the "train fresh per upload" product requirement),
    then pool results for a global comparison.

    Accepts either train_dir/test_dir/label_dir or train_path/test_path/
    test_label_path -- same thing, just in case you call it with either
    naming.

    Pooling convention (mirrors evaluate_multichannel for SMAP/MSL):
    - Global precision/recall/F1: each machine's own best-F1 threshold is
      applied to get predictions, then TP/FP/FN are summed across ALL
      machines before computing one global P/R/F1 (not an average of
      per-machine F1 scores, which would over-weight short machines).
    - Global AUC-ROC/AUC-PR: computed on y_true/scores concatenated across
      all machines. This assumes anomaly scores are on a comparable scale
      machine-to-machine -- reasonable for SMD since channels are already
      normalized to [0, 1], but worth keeping in mind if you see one machine
      dominating the pooled AUC.
    """
    train_dir = train_dir or train_path
    test_dir = test_dir or test_path
    label_dir = label_dir or test_label_path
    if not (train_dir and test_dir and label_dir):
        raise ValueError(
            "Need train_dir/test_dir/label_dir (or train_path/test_path/"
            "test_label_path) -- got train_dir="
            f"{train_dir!r}, test_dir={test_dir!r}, label_dir={label_dir!r}"
        )

    if machine_ids is None:
        machine_ids = discover_machines(train_dir, test_dir, label_dir)
    if verbose:
        print(f"Found {len(machine_ids)} machines: {machine_ids}\n")

    model_factories = {
        "PCA/SVD": lambda: PCAReconstructionDetector(n_components=pca_n_components),
        "Isolation Forest": lambda: IsolationForestDetector(random_state=0),
    }

    per_model_per_machine = {name: {} for name in model_factories}
    pooled_y_true = {name: [] for name in model_factories}
    pooled_scores = {name: [] for name in model_factories}
    pooled_tp = {name: 0 for name in model_factories}
    pooled_fp = {name: 0 for name in model_factories}
    pooled_fn = {name: 0 for name in model_factories}
    n_events_total = 0
    n_events_detected = {name: 0 for name in model_factories}
    all_latencies = {name: [] for name in model_factories}  # detected events only

    for mid in machine_ids:
        if verbose:
            print(f"--- {mid} ---")
        train_data = load_smd_series(os.path.join(train_dir, f"{mid}.txt"))
        test_data = load_smd_series(os.path.join(test_dir, f"{mid}.txt"))
        y_true = load_smd_labels(os.path.join(label_dir, f"{mid}.txt"))
        if test_data.shape[0] != y_true.shape[0]:
            if verbose:
                print(f"  SKIPPING {mid}: test rows ({test_data.shape[0]}) != "
                      f"label rows ({y_true.shape[0]})")
            continue

        train_windows, _ = sliding_windows(train_data, window_size, stride)
        test_windows, test_end_indices = sliding_windows(test_data, window_size, stride)

        for name, factory in model_factories.items():
            model = factory()
            model.fit(train_windows)
            window_scores = model.score(test_windows)
            timestep_scores = map_window_scores_to_timesteps(
                window_scores, test_end_indices, window_size, len(y_true), mode="max"
            )

            ev = evaluate_model(name, y_true, timestep_scores, use_point_adjust=use_point_adjust)
            per_model_per_machine[name][mid] = ev
            n_detected = sum(1 for e in ev.latencies if e.detected)
            machine_latencies = [e.latency for e in ev.latencies if e.detected]
            if verbose:
                print(f"  {name}: F1={ev.sweep.best_f1:.3f} AUC-PR={ev.auc.auc_pr:.3f} "
                      f"events {n_detected}/{len(ev.latencies)}"
                      + (f" mean_latency={np.mean(machine_latencies):.1f}" if machine_latencies else ""))

            y_pred = timestep_scores >= ev.sweep.best_threshold
            if use_point_adjust:
                y_pred = point_adjust(y_true, y_pred)
            pooled_tp[name] += int(np.sum(y_pred & y_true))
            pooled_fp[name] += int(np.sum(y_pred & ~y_true))
            pooled_fn[name] += int(np.sum(~y_pred & y_true))
            pooled_y_true[name].append(y_true)
            pooled_scores[name].append(timestep_scores)
            n_events_detected[name] += n_detected
            all_latencies[name].extend(machine_latencies)

        n_events_total += len(next(iter(per_model_per_machine.values()))[mid].latencies)
        if verbose:
            print()

    results = {}
    for name in model_factories:
        tp, fp, fn = pooled_tp[name], pooled_fp[name], pooled_fn[name]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        all_y_true = np.concatenate(pooled_y_true[name])
        all_scores = np.concatenate(pooled_scores[name])
        auc = compute_auc(all_y_true, all_scores)

        latencies = all_latencies[name]
        mean_lat = float(np.mean(latencies)) if latencies else float("nan")
        median_lat = float(np.median(latencies)) if latencies else float("nan")

        results[name] = MultiMachineResult(
            model_name=name,
            per_machine=per_model_per_machine[name],
            global_precision=precision,
            global_recall=recall,
            global_f1=f1,
            global_auc_roc=auc.auc_roc,
            global_auc_pr=auc.auc_pr,
            n_events_total=n_events_total,
            n_events_detected=n_events_detected[name],
            mean_latency=mean_lat,
            median_latency=median_lat,
        )

    if verbose:
        print("=== Global results across all machines ===")
        for name, r in results.items():
            print(f"{name}:")
            print(f"  Global precision/recall/F1: {r.global_precision:.4f} / "
                  f"{r.global_recall:.4f} / {r.global_f1:.4f}")
            print(f"  Global AUC-ROC: {r.global_auc_roc:.4f}")
            print(f"  Global AUC-PR:  {r.global_auc_pr:.4f}")
            print(f"  Events detected: {r.n_events_detected}/{r.n_events_total} "
                  f"({r.n_events_detected / r.n_events_total:.1%})" if r.n_events_total else "")
            print(f"  Detection latency (detected events only): "
                  f"mean={r.mean_latency:.1f} timesteps, median={r.median_latency:.1f} timesteps")

        print("\n=== Accuracy vs. earliness, side by side ===")
        print(f"{'Model':<20}{'AUC-PR':>10}{'Recall':>10}{'Mean latency':>16}")
        for name, r in results.items():
            recall_events = r.n_events_detected / r.n_events_total if r.n_events_total else 0.0
            print(f"{name:<20}{r.global_auc_pr:>10.4f}{recall_events:>10.1%}{r.mean_latency:>16.1f}")

    return results


if __name__ == "__main__":
    # Point these at your actual SMD folders. If they don't exist relative
    # to wherever you run this script from, fix the paths below rather than
    # running with the defaults.
    TRAIN_DIR = "train"
    TEST_DIR = "test"
    LABEL_DIR = "test_label"

    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR) and os.path.isdir(LABEL_DIR)):
        cwd = os.getcwd()
        print(
            f"Could not find one or more of: {TRAIN_DIR}/, {TEST_DIR}/, {LABEL_DIR}/ "
            f"relative to the current working directory ({cwd}).\n"
            f"Either run this script from the folder containing those three "
            f"directories, or edit TRAIN_DIR/TEST_DIR/LABEL_DIR above to the "
            f"correct paths.\n"
        )
        print(f"Here's what's actually in {cwd}:")
        for entry in sorted(os.listdir(cwd)):
            full = os.path.join(cwd, entry)
            print(f"  {'[dir] ' if os.path.isdir(full) else '[file]'} {entry!r}")
    else:
        run_multi_machine(
            train_dir=TRAIN_DIR,
            test_dir=TEST_DIR,
            label_dir=LABEL_DIR,
        )