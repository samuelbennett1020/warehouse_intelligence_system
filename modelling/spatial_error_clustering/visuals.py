import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from typing import List


def assign_display_class(df: pd.DataFrame, class_cols: List[str], threshold: float, default_class: str) -> pd.Series:
    """
    Determine which class to display for each location based on threshold.
    If the default_class proportion < threshold, display the next highest class.

    Args:
        df (pd.DataFrame): DataFrame with class proportion columns.
        class_cols (List[str]): Columns representing class proportions.
        threshold (float): Minimum proportion to display a class.
        default_class (str): Primary class to highlight if above threshold.

    Returns:
        pd.Series: Series of class names to display per row.
    """

    def pick_class(row):
        if row[default_class] >= threshold:
            return default_class
        # Otherwise pick the class with highest proportion
        return row[class_cols].idxmax()

    return df.apply(pick_class, axis=1)


def plot_class_overlay(
        features: pd.DataFrame,
        image_path: Path,
        class_cols: List[str],
        threshold: float = 0.8,
        default_class: str = "Correct"
) -> None:
    """
    Plot warehouse locations, highlighting the class with highest proportion above threshold.

    Args:
        features (pd.DataFrame): DataFrame with x, y coordinates and class proportions.
        image_path (Path): Path to warehouse PNG image.
        class_cols (List[str]): Columns representing class proportions.
        threshold (float): Proportion threshold to highlight.
        default_class (str): Default primary class to highlight if above threshold.

    Returns:
        None
    """
    img = mpimg.imread(image_path)
    x_max, y_max = features['x'].max(), features['y'].max()

    # Assign display class
    features['display_class'] = assign_display_class(features, class_cols, threshold, default_class)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img, extent=[1., x_max, 0, y_max + 5.], origin='upper', alpha=0.5)

    # Background grey points
    ax.scatter(features['x'], features['y'], color='lightgrey', s=10, alpha=0.3)

    # Overlay by class
    classes = features['display_class'].unique()
    colors = plt.cm.get_cmap("Paired", len(classes))

    for i, cls in enumerate(classes):
        mask = features['display_class'] == cls
        ax.scatter(
            features.loc[mask, 'x'],
            features.loc[mask, 'y'],
            color=colors(i),
            s=20,
            alpha=0.8,
            label=f"{cls}"
        )

    ax.set_xlabel("x midpoint")
    ax.set_ylabel("y midpoint")
    ax.set_title(f"Warehouse Locations by Dominant Class, {default_class} Threshold: {100*threshold} %")
    ax.legend()
    plt.tight_layout()
    plt.show()
