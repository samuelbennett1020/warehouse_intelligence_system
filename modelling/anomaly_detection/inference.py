import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def load_distributions(file_path: str = "distributions.npz") -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Load saved distributions and metadata from a .npz file.

    Args:
        file_path (str): Path to the saved .npz file.

    Returns:
        Tuple containing:
            - distributions (np.ndarray): Saved distributions array of shape (num_aisles, num_timesteps, num_statuses).
            - aisles (np.ndarray): Array of aisle names.
            - timesteps (np.ndarray): Array of timesteps.
            - statuses (List[str]): List of status categories.
    """
    data = np.load(file_path, allow_pickle=True)
    distributions: np.ndarray = data['distributions']
    aisles: np.ndarray = data['aisles']
    timesteps: np.ndarray = data['timesteps']
    statuses: List[str] = data['statuses'].tolist()
    return distributions, aisles, timesteps, statuses


def infer_anomalies(
    df: pd.DataFrame,
    saved_distributions_file: str = "distributions.npz",
    class_percentiles: Dict[str, int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Perform anomaly detection on a new dataframe using precomputed distributions.

    Args:
        df (pd.DataFrame): New warehouse data with columns ['Aisle', 'timestep', 'Location', 'status'].
        saved_distributions_file (str): Path to the saved distributions file (.npz).
        class_percentiles (Dict[str, int], optional): Percentile thresholds per status for detecting anomalies.

    Returns:
        Tuple containing:
            - per_class_anomaly_scores (np.ndarray): Z-score anomaly scores (shape: num_aisles x num_timesteps x num_statuses).
            - per_class_anomalies (np.ndarray): Boolean array indicating detected anomalies.
            - aisles (np.ndarray): Array of aisles used in inference.
            - timesteps (np.ndarray): Array of timesteps used in inference.
            - statuses (List[str]): List of statuses used in inference.
    """
    if class_percentiles is None:
        class_percentiles = {'Ignored': 99, 'Warning': 95, 'Error': 98, 'Obstructed': 99}

    # Load saved distributions and metadata
    distributions, aisles, timesteps, statuses = load_distributions(saved_distributions_file)
    status_to_idx: Dict[str, int] = {status: i for i, status in enumerate(statuses)}

    # Compute distributions for the new df
    new_distributions: np.ndarray = np.zeros_like(distributions)
    for i, aisle in enumerate(aisles):
        for t, timestep in enumerate(timesteps):
            subset: pd.DataFrame = df[(df['aisle'] == aisle) & (df['timestep'] == timestep)]
            counts: np.ndarray = np.zeros(len(statuses))
            for status in subset['status']:
                if status in status_to_idx:
                    counts[status_to_idx[status]] += 1
            new_distributions[i, t] = counts / counts.sum() if counts.sum() > 0 else np.zeros(len(statuses))

    # Compute mean and std from saved distributions
    mean_dist: np.ndarray = distributions.mean(axis=1)
    std_dist: np.ndarray = distributions.std(axis=1) + 1e-6

    # Compute anomaly scores (z-scores)
    per_class_anomaly_scores: np.ndarray = np.abs(new_distributions - mean_dist[:, np.newaxis, :]) / std_dist[:, np.newaxis, :]

    # Detect anomalies based on thresholds
    per_class_anomalies: np.ndarray = np.zeros_like(per_class_anomaly_scores, dtype=bool)
    for idx, status in enumerate(statuses):
        percentile: int = class_percentiles.get(status, 99)
        threshold: float = np.percentile(per_class_anomaly_scores[:, :, idx], percentile)
        per_class_anomalies[:, :, idx] = per_class_anomaly_scores[:, :, idx] > threshold

    return per_class_anomaly_scores, per_class_anomalies, aisles, timesteps, statuses


# ------------------- Example Usage -------------------

if __name__ == "__main__":

    """
    An example script to show how to take a dataframe and infer whether an anomaly has occured
    """

    # Example new data, would have to be
    df_new = pd.DataFrame({
        'aisle': ['AN 2', 'AN 2', 'AN 2', 'AN 2', 'AN 2'],
        'timestep': [0, 1, 0, 1, 0],
        'location': ['L1', 'L2', 'L3', 'L4', 'L5'],
        'status': ['Warning', 'Error', 'Correct', 'Ignored', 'Error']
    })

    scores, anomalies, aisles, timesteps, statuses = infer_anomalies(df_new)

    print("Aisles:", aisles)
    print("Timesteps:", timesteps)
    print("statuses:", statuses)
    print("Anomaly Scores:\n", scores)
    print("Detected Anomalies:\n", anomalies)
