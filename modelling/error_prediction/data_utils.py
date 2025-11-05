import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple
import random
import torch


def load_and_preprocess(parquet_path: str, target_col: str,
                        location_col: str = "location") -> Tuple[pd.DataFrame, list, list, dict, dict, list]:
    """
    Load dataset and automatically infer numeric/categorical features, encode target, scale numeric features.

    Args:
        parquet_path: Path to the CSV/Parquet file.
        target_col: Name of target column.
        location_col: Name of the location column.

    Returns:
        Tuple containing:
        - DataFrame
        - numeric_features
        - categorical_features
        - cat_encoders
        - target_mapping
        - cat_cardinalities
    """
    df = pd.read_parquet(parquet_path)

    # Automatically detect numeric and categorical columns
    numeric_features = [c for c in df.select_dtypes(include=['float64', 'int64']).columns
                        if c != target_col and c != location_col]
    categorical_features = [c for c in df.select_dtypes(include=['object', 'category']).columns
                            if c != location_col]

    # Fill missing categorical values and factorize
    cat_encoders = {}
    for col in categorical_features:
        df[col] = df[col].astype(str)
        codes, uniques = pd.factorize(df[col])

        # Shift by +1 so 0 = reserved for <UNK>
        df[col] = codes + 1

        # Store encoder with <UNK> explicitly at index 0
        cat_encoders[col] = {'<UNK>': 0, **{val: i + 1 for i, val in enumerate(uniques)}}

    # Encode target
    codes, uniques = pd.factorize(df[target_col])
    df[target_col] = codes
    target_mapping = {val: i for i, val in enumerate(uniques)}

    # Scale numeric features
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Embedding cardinalities
    cat_cardinalities = [df[c].max() + 1 for c in categorical_features]
    return df, numeric_features, categorical_features, cat_encoders, target_mapping, cat_cardinalities


def train_val_test_split_by_location(df: pd.DataFrame, location_col: str = 'location',
                                     test_size: float = 0.3, val_ratio: float = 0.5, random_state: int = 42
                                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train, validation, and test sets by unique location.

    Args:
        df (pd.DataFrame): Input dataframe.
        location_col (str): Column representing location.
        test_size (float): Fraction of data to use for test+validation.
        val_ratio (float): Fraction of remaining for validation.
        random_state (int): Random seed.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: train, val, test splits.
    """
    locations = df[location_col].unique()
    train_locs, val_test_locs = train_test_split(locations, test_size=test_size, random_state=random_state)
    val_locs, test_locs = train_test_split(val_test_locs, test_size=val_ratio, random_state=random_state)

    df_train = df[df[location_col].isin(train_locs)].copy()
    df_val = df[df[location_col].isin(val_locs)].copy()
    df_test = df[df[location_col].isin(test_locs)].copy()

    return df_train, df_val, df_test


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch for reproducible results.

    Args:
        seed (int): Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
