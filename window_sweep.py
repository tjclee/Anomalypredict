import numpy as np
import matplotlib.pyplot as plt

from run_smd_evaluation import run_multi_machine


def sweep_window_sizes(
    train_dir: str,
    test_dir: str,
    label_dir: str,
    window_sizes: list = (125, 150, 175, 200,),
    stride: int = 1,
    pca_n_components: int = 5,
    plot_path: str = "window_size_sweep.png",
):
    rows = []  # (window_size, model_name, events_detected, n_events_total, recall_events, f1, auc_pr, mean_latency, median_latency)

    for ws in window_sizes:
        print(f"\n########## window_size = {ws} ##########")
        results = run_multi_machine(
            train_dir=train_dir,
            test_dir=test_dir,
            label_dir=label_dir,
            window_size=ws,
            stride=stride,
            pca_n_components=pca_n_components,
            verbose=False,  # sweep prints its own compact summary at the end
        )
        for name, r in results.items():
            recall_events = r.n_events_detected / r.n_events_total if r.n_events_total else 0.0
            rows.append((
                ws, name, r.n_events_detected, r.n_events_total,
                recall_events, r.global_f1, r.global_auc_pr,
                r.mean_latency, r.median_latency,
            ))
            print(f"  {name:<20} events {r.n_events_detected:>4}/{r.n_events_total:<4} "
                  f"({recall_events:>5.1%})  F1={r.global_f1:.3f}  AUC-PR={r.global_auc_pr:.3f}  "
                  f"mean_lat={r.mean_latency:>6.1f}")

    print("\n\n=== Window size sweep summary ===")
    header = f"{'window':>8}{'model':<20}{'events':>14}{'F1':>8}{'AUC-PR':>9}{'mean_lat':>10}"
    print(header)
    print("-" * len(header))
    for ws, name, n_det, n_tot, recall_events, f1, auc_pr, mean_lat, median_lat in rows:
        events_str = f"{n_det}/{n_tot} ({recall_events:.0%})"
        print(f"{ws:>8}{name:<20}{events_str:>14}{f1:>8.3f}{auc_pr:>9.3f}{mean_lat:>10.1f}")

    plot_window_sweep(rows, plot_path)

    return rows


def plot_window_sweep(rows, plot_path: str = "window_size_sweep.png"):
    """
    Grouped bar chart (one bar per model per window size) for F1, AUC-PR,
    and event recall, stacked into three subplots, saved to plot_path.
    """
    window_sizes = sorted(set(r[0] for r in rows))
    model_names = sorted(set(r[1] for r in rows))

    # index rows by (window_size, model_name) for easy lookup
    lookup = {(ws, name): row for row in rows for ws, name in [(row[0], row[1])]}

    metrics = [
        ("recall_events", 4, "Event recall"),
        ("f1", 5, "Global F1"),
        ("auc_pr", 6, "Global AUC-PR"),
    ]

    n_models = len(model_names)
    bar_width = 0.8 / max(n_models, 1)
    x = np.arange(len(window_sizes))

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 3.2 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, (_, col_idx, title) in zip(axes, metrics):
        for i, name in enumerate(model_names):
            values = [lookup[(ws, name)][col_idx] for ws in window_sizes]
            ax.bar(x + i * bar_width, values, width=bar_width, label=name)
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. window size")
        ax.grid(axis="y", alpha=0.3)

    axes[-1].set_xticks(x + bar_width * (n_models - 1) / 2)
    axes[-1].set_xticklabels([str(ws) for ws in window_sizes])
    axes[-1].set_xlabel("window_size")
    axes[0].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved sweep histogram to {plot_path}")


if __name__ == "__main__":
    import os

    TRAIN_DIR, TEST_DIR, LABEL_DIR = "train", "test", "test_label"
    if not (os.path.isdir(TRAIN_DIR) and os.path.isdir(TEST_DIR) and os.path.isdir(LABEL_DIR)):
        print(f"Could not find train/, test/, test_label/ in {os.getcwd()}. "
              f"Edit TRAIN_DIR/TEST_DIR/LABEL_DIR above or run from the right folder.")
    else:
        sweep_window_sizes(TRAIN_DIR, TEST_DIR, LABEL_DIR)