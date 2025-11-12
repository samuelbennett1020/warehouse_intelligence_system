import warnings
from typing import List, Tuple
import pandas as pd

from modelling.spatial_error_clustering.utils import load_config, get_warehouse_image
from modelling.spatial_error_clustering.cluster import scale_features, perform_hdbscan
from modelling.spatial_error_clustering.visuals import plot_class_overlay


warnings.simplefilter(action='ignore', category=FutureWarning)


# ---------- Feature computation functions ----------

def compute_midpoints(df: pd.DataFrame, axes: List[str] = ['x','y','z']) -> pd.DataFrame:
    """
    Compute midpoint coordinates for min/max axes.

    Args:
        df (pd.DataFrame): DataFrame with 'min_' and 'max_' columns for each axis.
        axes (List[str], optional): List of axes to compute. Defaults to ['x','y','z'].

    Returns:
        pd.DataFrame: DataFrame with new midpoint columns added.
    """
    for axis in axes:
        df[axis] = (df[f'min_{axis}'] + df[f'max_{axis}']) / 2
    return df


def compute_class_proportions(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute class counts and class proportions per location.

    Args:
        df (pd.DataFrame): DataFrame containing 'location' and 'status'.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - class_proportions: DataFrame with normalized proportions per location.
            - class_counts: DataFrame with raw counts per location.
    """
    class_counts = df.groupby(['location','status']).size().unstack(fill_value=0)
    class_counts.columns = [str(c) for c in class_counts.columns]
    class_proportions = class_counts.div(class_counts.sum(axis=1), axis=0).reset_index()
    return class_proportions, class_counts


def compute_mean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean x, y, z coordinates per location.

    Args:
        df (pd.DataFrame): DataFrame containing 'x', 'y', 'z', and 'location'.

    Returns:
        pd.DataFrame: DataFrame with mean coordinates per location.
    """
    return df.groupby('location')[['x','y','z']].mean().reset_index()


def merge_features(coords: pd.DataFrame, class_proportions: pd.DataFrame) -> pd.DataFrame:
    """
    Merge mean coordinates with class proportions safely.

    Args:
        coords (pd.DataFrame): DataFrame with mean coordinates per location.
        class_proportions (pd.DataFrame): DataFrame with class proportions per location.

    Returns:
        pd.DataFrame: Merged DataFrame containing coordinates and class proportions.
    """
    return pd.merge(coords, class_proportions, on='location', how='inner')

# ---------- Main workflow ----------


def main(config_path: str = "../../config/clustering_config.yaml") -> None:
    """
    Main workflow for warehouse clustering and visualization.

    Steps:
        1. Load configuration from YAML file.
        2. Load warehouse data via ingest_data().
        3. Compute midpoint coordinates.
        4. Compute class counts and proportions.
        5. Compute mean coordinates per location.
        6. Merge features into a single DataFrame.
        7. Scale features and perform HDBSCAN clustering.
        8. Summarize clusters.
        9. Visualize locations highlighting dominant classes.

    Args:
        config_path (str, optional): Path to configuration YAML. Defaults to 'config.yaml'.

    Returns:
        None
    """
    # Load configuration
    config = load_config(config_path)

    # Load and preprocess data
    df = pd.read_parquet(config.get('data_path'))
    df = compute_midpoints(df)

    # Compute class proportions and coordinates
    class_proportions, class_counts = compute_class_proportions(df)
    coords = compute_mean_coordinates(df)
    features = merge_features(coords, class_proportions)

    # Check required columns
    feature_cols = ['x','y','z'] + [c for c in class_counts.columns]
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Perform clustering
    X_scaled = scale_features(features, feature_cols)
    cluster_params = config.get("clustering", {})
    features['cluster'] = perform_hdbscan(
        X_scaled,
        min_cluster_size=cluster_params.get("min_cluster_size", 100),
        min_samples=cluster_params.get("min_samples", 10)
    )

    # Summarize clusters
    cluster_summary = features.groupby('cluster')[class_counts.columns].mean()
    cluster_summary['dominant_class'] = cluster_summary.idxmax(axis=1)
    print("Cluster summaries:")
    print(cluster_summary)

    # Map cluster ID → dominant class
    cluster_to_class = cluster_summary['dominant_class'].to_dict()
    class_colors = {
        'Error': 'red',
        'Ignored': 'blue',
        'Warning': 'orange',
        'Obstructed': 'grey',
        'Correct': None
    }

    # Create a column in features for the dominant class
    features['dominant_class'] = features['cluster'].map(cluster_to_class)
    features['color'] = features['dominant_class'].map(class_colors)

    # Visualization
    highlight_config = config.get("highlight", {})
    default_class = highlight_config.get("default_class", "Correct")
    image_path = get_warehouse_image(config)

    plot_class_overlay(
        features,
        image_path,
        class_colors,
        default_class=default_class
    )


if __name__ == "__main__":
    main()
