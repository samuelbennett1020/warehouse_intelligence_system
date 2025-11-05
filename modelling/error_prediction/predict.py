import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)


def prepare_sequence(df_seq: pd.DataFrame,
                     numeric_features: List[str],
                     categorical_features: List[str],
                     cat_encoders: Dict[str, Dict],
                     cat_cardinalities: List[int],
                     seq_len: int = 10,
                     device: torch.device = torch.device('cpu')
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a DataFrame sequence to model-ready tensors for prediction.

    Args:
        df_seq (pd.DataFrame): DataFrame containing the sequence (length = seq_len).
        numeric_features (List[str]): List of numeric feature names.
        categorical_features (List[str]): List of categorical feature names.
        cat_encoders (Dict[str, Dict]): Mapping from category to integer index for each categorical feature.
        cat_cardinalities (List[int]): Cardinalities of categorical features.
        seq_len (int): Sequence length (default 10).
        device (torch.device): Device to move tensors to.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (X_num_tensor, X_cat_tensor, cat_mask_tensor)
    """
    # Make a copy to avoid SettingWithCopyWarning
    df_seq = df_seq.copy()

    # Process categorical features
    for col in categorical_features:
        df_seq[col] = df_seq[col].astype(str).fillna('<UNK>')

    # Encode categorical features
    X_cat = []
    for col in categorical_features:
        enc = cat_encoders[col]
        codes = df_seq[col].map(lambda x: enc.get(x, 0)).values
        X_cat.append(codes)
    X_cat = np.stack(X_cat, axis=1).astype('int64')

    # Numeric features
    X_num = df_seq[numeric_features].values.astype('float32')

    # Mask for categorical embeddings (0 = padding/<UNK>)
    cat_mask = np.where(X_cat == 0, 0.0, 1.0).astype('float32')

    # Convert to tensors
    X_num_tensor = torch.tensor(X_num, dtype=torch.float32).unsqueeze(0).to(device)
    X_cat_tensor = torch.tensor(X_cat, dtype=torch.long).unsqueeze(0).to(device)
    cat_mask_tensor = torch.tensor(cat_mask, dtype=torch.float32).unsqueeze(0).to(device)

    # Clamp categorical indices
    for i, n_cat in enumerate(cat_cardinalities):
        X_cat_tensor[:, :, i] = torch.clamp(X_cat_tensor[:, :, i], 0, n_cat-1)

    return X_num_tensor, X_cat_tensor, cat_mask_tensor


def predict_next(model: torch.nn.Module,
                 X_num_tensor: torch.Tensor,
                 X_cat_tensor: torch.Tensor,
                 cat_mask_tensor: torch.Tensor,
                 target_mapping: Dict[int, str],
                 should_log: bool = True,
                ) -> int:
    """
    Predict the next timestep label given a prepared sequence.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        X_num_tensor (torch.Tensor): Numeric features tensor.
        X_cat_tensor (torch.Tensor): Categorical features tensor.
        cat_mask_tensor (torch.Tensor): Categorical mask tensor.
        target_mapping (Dict[int, str]): Mapping from class indices to labels.

    Returns:
        str: Predicted label.
    """
    model.eval()
    with torch.no_grad():
        preds = model(X_num_tensor, X_cat_tensor, cat_mask_tensor)
        predicted_idx = preds.argmax(1).item()
        predicted_label = list(target_mapping.keys())[list(target_mapping.values()).index(predicted_idx)]

    if should_log:
        logging.info(f"Predicted next label: {predicted_label}")
    return predicted_label
