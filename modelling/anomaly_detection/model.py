import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import pandas as pd


def get_unique_values(df: pd.DataFrame, exclude_status: str = "Correct") -> Tuple[
    np.ndarray, np.ndarray, List[str], Dict[str, int]]:
    """
    Extract unique aisles, timesteps, statuses, and map statuses to indices.

    Args:
        df (pd.DataFrame): Input warehouse data.
        exclude_status (str): status to exclude from anomaly detection.

    Returns:
        Tuple containing:
            - aisles (np.ndarray): Unique aisle names.
            - timesteps (np.ndarray): Unique timesteps.
            - statuses (List[str]): Relevant status categories.
            - status_to_idx (Dict[str, int]): Mapping from status to index.
    """
    aisles: np.ndarray = df['aisle'].unique()
    timesteps: np.ndarray = df['timestep'].unique()
    statuses: List[str] = [s for s in df['status'].unique() if s != exclude_status]
    status_to_idx: Dict[str, int] = {status: i for i, status in enumerate(statuses)}
    return aisles, timesteps, statuses, status_to_idx


def compute_distributions(df: pd.DataFrame, aisles: np.ndarray, timesteps: np.ndarray,
                          statuses: List[str], status_to_idx: Dict[str, int]) -> np.ndarray:
    """
    Compute probability distributions per aisle per timestep.

    Args:
        df (pd.DataFrame): Warehouse data.
        aisles (np.ndarray): Array of aisle names.
        timesteps (np.ndarray): Array of timesteps.
        statuses (List[str]): status categories.
        status_to_idx (Dict[str,int]): Mapping from status to index.

    Returns:
        distributions (np.ndarray): Shape (num_aisles, num_timesteps, num_statuses)
    """
    num_categories: int = len(statuses)
    distributions: np.ndarray = np.zeros((len(aisles), len(timesteps), num_categories))

    for i, aisle in enumerate(aisles):
        for t, timestep in enumerate(timesteps):
            subset: pd.DataFrame = df[(df['aisle'] == aisle) & (df['timestep'] == timestep)]
            counts: np.ndarray = np.zeros(num_categories)
            for status in subset['status']:
                if status in status_to_idx:
                    counts[status_to_idx[status]] += 1
            distributions[i, t] = counts / counts.sum() if counts.sum() > 0 else np.zeros(num_categories)
    return distributions


def compute_anomaly_scores(distributions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean, std, and per-class per-aisle anomaly scores (z-scores).

    Args:
        distributions (np.ndarray): Probability distributions per aisle/timestep/status.

    Returns:
        mean_dist (np.ndarray): Mean distribution per aisle.
        std_dist (np.ndarray): Standard deviation per aisle.
        per_class_anomaly_scores (np.ndarray): Z-score anomaly scores.
    """
    mean_dist: np.ndarray = distributions.mean(axis=1)
    std_dist: np.ndarray = distributions.std(axis=1) + 1e-6  # avoid division by zero
    per_class_anomaly_scores: np.ndarray = np.abs(distributions - mean_dist[:, np.newaxis, :]) / std_dist[:, np.newaxis,
                                                                                                 :]
    return mean_dist, std_dist, per_class_anomaly_scores


def detect_anomalies(per_class_anomaly_scores: np.ndarray, statuses: List[str],
                     class_percentiles: Dict[str, int]) -> np.ndarray:
    """
    Detect anomalies based on percentile thresholds per class.

    Args:
        per_class_anomaly_scores (np.ndarray): Z-score anomaly scores.
        statuses (List[str]): status categories.
        class_percentiles (Dict[str,int]): Percentile thresholds per status.

    Returns:
        per_class_anomalies (np.ndarray): Boolean array indicating anomalies.
    """
    per_class_anomalies: np.ndarray = np.zeros_like(per_class_anomaly_scores, dtype=bool)
    for idx, status in enumerate(statuses):
        percentile: int = class_percentiles.get(status, 99)
        threshold: float = np.percentile(per_class_anomaly_scores[:, :, idx], percentile)
        per_class_anomalies[:, :, idx] = per_class_anomaly_scores[:, :, idx] > threshold
    return per_class_anomalies


def save_distributions(distributions: np.ndarray, aisles: np.ndarray, timesteps: np.ndarray,
                       statuses: List[str], file_path: str = "distributions.npz") -> None:
    """
    Save distributions and metadata to file for later inference.

    Args:
        distributions (np.ndarray): Computed distributions.
        aisles (np.ndarray): aisle names.
        timesteps (np.ndarray): Timesteps.
        statuses (List[str]): status categories.
        file_path (str): Path to save the .npz file.
    """
    np.savez(file_path, distributions=distributions, aisles=aisles, timesteps=timesteps, statuses=statuses)
    print(f"Distributions saved to {file_path}")


def plot_anomalies(per_class_anomaly_scores: np.ndarray, per_class_anomalies: np.ndarray,
                   aisles: np.ndarray, timesteps: np.ndarray, statuses: List[str],
                   class_percentiles: Dict[str, int]) -> None:
    """
    Plot heatmaps of anomaly scores with anomaly markers, using different colormaps per status.

    Args:
        per_class_anomaly_scores (np.ndarray): Z-score anomaly scores.
        per_class_anomalies (np.ndarray): Boolean array indicating anomalies.
        aisles (np.ndarray): Array of aisle names.
        timesteps (np.ndarray): Array of timesteps.
        statuses (List[str]): status categories.
        class_percentiles (Dict[str,int]): Percentile thresholds per status.
    """
    colormaps: List[str] = ['Reds', 'Blues', 'Greens', 'Oranges', 'Purples', 'Greys']
    num_categories: int = len(statuses)
    num_cols: int = 2
    num_rows: int = int(np.ceil(num_categories / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(5 * num_cols, 4 * num_rows), sharey='row')
    axes = axes.flatten()

    for idx, status in enumerate(statuses):
        ax = axes[idx]
        cmap = colormaps[idx % len(colormaps)]
        sns.heatmap(per_class_anomaly_scores[:, :, idx],
                    xticklabels=timesteps,
                    yticklabels=aisles,
                    cmap=cmap,
                    cbar_kws={'label': 'Anomaly Score'}, ax=ax)
        ax.set_title(f'status: {status}, Threshold {class_percentiles.get(status, 99)}%')
        ax.set_xlabel('Timestep')
        if idx % num_cols == 0:
            ax.set_ylabel('aisle')

        for i in range(len(aisles)):
            for t in range(len(timesteps)):
                if per_class_anomalies[i, t, idx]:
                    ax.scatter([t + 0.5], [i + 0.5], marker='x', c='black')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    Main workflow: ingest data, compute anomaly scores, detect anomalies, save distributions, plot results.
    """
    df: pd.DataFrame = pd.read_parquet("../../data/processed_data/timeseries_data.parquet")

    aisles, timesteps, statuses, status_to_idx = get_unique_values(df)
    distributions: np.ndarray = compute_distributions(df, aisles, timesteps, statuses, status_to_idx)

    # Save distributions for inference
    save_distributions(distributions, aisles, timesteps, statuses)

    _, _, per_class_anomaly_scores = compute_anomaly_scores(distributions)

    class_percentiles: Dict[str, int] = {
        'Ignored': 99,
        'Warning': 95,
        'Error': 98,
        'Obstructed': 99
    }
    class_percentiles = {k: v for k, v in class_percentiles.items() if k in statuses}

    per_class_anomalies: np.ndarray = detect_anomalies(per_class_anomaly_scores, statuses, class_percentiles)
    plot_anomalies(per_class_anomaly_scores, per_class_anomalies, aisles, timesteps, statuses, class_percentiles)


if __name__ == "__main__":
    main()
