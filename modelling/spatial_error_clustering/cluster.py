import pandas as pd
import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
from typing import List


def scale_features(df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    """
    Standardize numerical features.

    Args:
        df (pd.DataFrame): DataFrame containing features to scale.
        feature_cols (List[str]): Columns to scale.

    Returns:
        np.ndarray: Scaled feature matrix.
    """
    X = df[feature_cols].to_numpy()
    return StandardScaler().fit_transform(X)


def perform_hdbscan(
    X_scaled: np.ndarray,
    min_cluster_size: int = 100,
    min_samples: int = 10
) -> np.ndarray:
    """
    Perform HDBSCAN clustering.

    Args:
        X_scaled (np.ndarray): Scaled feature matrix.
        min_cluster_size (int): Minimum cluster size.
        min_samples (int): Minimum samples for a core point.

    Returns:
        np.ndarray: Cluster labels for each sample.
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method='eom',
        metric='euclidean'
    )
    return clusterer.fit_predict(X_scaled)
