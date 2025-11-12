import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from matplotlib.lines import Line2D


def plot_class_overlay(
        features: pd.DataFrame,
        image_path: Path,
        class_colors: dict[str, str],
        default_class: str = "Correct"
) -> None:
    """
    Plot warehouse locations, highlighting the class with highest proportion above threshold.

    Args:
        features (pd.DataFrame): DataFrame with x, y coordinates and class proportions.
        image_path (Path): Path to warehouse PNG image.
        default_class (str): Default primary class to highlight if above threshold.

    Returns:
        None
    """
    img = mpimg.imread(image_path)
    x_max, y_max = features['x'].max(), features['y'].max()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, extent=[1., x_max, 0, y_max + 5.], origin='upper', alpha=0.5)

    # Background grey points
    ax.scatter(features['x'], features['y'], color='lightgrey', s=10, alpha=0.3)

    mask = features['dominant_class'] != default_class

    # Optionally, you can also ignore noise points
    mask = mask & (features['cluster'] != -1)

    # Subset the features DataFrame
    features_wrong = features.loc[mask].copy()

    ax.scatter(
        features_wrong['x'],
        features_wrong['y'],
        c=features_wrong['color'],  # highlight wrong clusters
        s=20,
        alpha=0.8,
        edgecolors='k'
    )

    wrong_classes = features_wrong['dominant_class'].unique()
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=cls,
                              markerfacecolor=class_colors[cls], markersize=8)
                       for cls in wrong_classes]

    ax.legend(handles=legend_elements, title='Dominant Class')

    ax.set_xlabel("x midpoint")
    ax.set_ylabel("y midpoint")
    ax.set_title(f"Clusters where Dominant Class ≠ {default_class}")
    plt.tight_layout()
    plt.show()
