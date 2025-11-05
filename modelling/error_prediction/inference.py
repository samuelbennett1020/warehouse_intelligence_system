import torch
import pandas as pd
import yaml
import logging

from modelling.error_prediction.model import LSTMClassifier
from modelling.error_prediction.predict import prepare_sequence, predict_next
from modelling.error_prediction.data_utils import load_and_preprocess

logging.basicConfig(level=logging.INFO)


def load_model(checkpoint_path: str,
               n_numeric: int,
               cat_cardinalities: list,
               embedding_dim: int,
               hidden_dim: int,
               n_classes: int,
               device: torch.device
              ) -> torch.nn.Module:
    """Load a trained LSTM model from a checkpoint."""
    model = LSTMClassifier(n_numeric, cat_cardinalities, embedding_dim, hidden_dim, n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {checkpoint_path}")
    return model


def infer_sequence(df_seq: pd.DataFrame,
                   model: torch.nn.Module,
                   numeric_features: list,
                   categorical_features: list,
                   cat_encoders: dict,
                   cat_cardinalities: list,
                   target_mapping: dict,
                   seq_len: int,
                   device: torch.device
                  ) -> str:
    """Run inference on a single sequence DataFrame."""
    X_num_tensor, X_cat_tensor, cat_mask_tensor = prepare_sequence(
        df_seq, numeric_features, categorical_features, cat_encoders, cat_cardinalities,
        seq_len=seq_len, device=device
    )
    predicted_label = predict_next(model, X_num_tensor, X_cat_tensor, cat_mask_tensor, target_mapping)
    return predicted_label


def batch_inference(df: pd.DataFrame,
                    model: torch.nn.Module,
                    numeric_features: list,
                    categorical_features: list,
                    cat_encoders: dict,
                    cat_cardinalities: list,
                    target_mapping: dict,
                    seq_len: int,
                    location_col: str = "location",
                    time_column: str = "timestep",
                    device: torch.device = torch.device("cpu"),
                    output_path: str = None
                   ) -> pd.DataFrame:
    """
    Run inference for all locations in a DataFrame and optionally save predictions to CSV.

    Args:
        df (pd.DataFrame): DataFrame containing multiple locations and timesteps.
        model (nn.Module): Trained model.
        numeric_features (list): Numeric feature names.
        categorical_features (list): Categorical feature names.
        cat_encoders (dict): Categorical encoders.
        cat_cardinalities (list): Cardinalities for embeddings.
        target_mapping (dict): Mapping from class index to label.
        seq_len (int): Sequence length.
        location_col (str): Column name for locations.
        time_column (str): Column name for time steps.
        device (torch.device): Device to run inference on.
        output_path (str): Path to save predictions CSV. If None, no file is saved.

    Returns:
        pd.DataFrame: DataFrame with location and predicted label.
    """
    predictions = []

    for loc in df[location_col].unique():
        df_seq = df[df[location_col] == loc].sort_values(time_column).tail(seq_len)
        if len(df_seq) < seq_len:
            logging.warning(f"Not enough data for location {loc}, skipping.")
            continue
        label = infer_sequence(df_seq, model, numeric_features, categorical_features,
                               cat_encoders, cat_cardinalities, target_mapping, seq_len, device)
        predictions.append({"location": loc, "predicted_label": label})

    predictions_df = pd.DataFrame(predictions)

    if output_path:
        predictions_df.to_csv(output_path, index=False)
        logging.info(f"Predictions saved to {output_path}")

    return predictions_df


if __name__ == "__main__":
    # --- Load config ---
    with open("../../config/error_prediction_config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load data and preprocessing ---
    df, numeric_features, categorical_features, cat_encoders, target_mapping, cat_cardinalities = \
        load_and_preprocess(cfg['data']['path'], cfg['target']['column'],
                            location_col=cfg['data'].get('location_col', 'location'))

    # --- Load trained model ---
    n_numeric = len(numeric_features)
    n_classes = len(target_mapping)
    model = load_model(cfg['training']['checkpoint_path'], n_numeric, cat_cardinalities,
                       embedding_dim=cfg['model']['embedding_dim'],
                       hidden_dim=cfg['model']['hidden_dim'],
                       n_classes=n_classes,
                       device=device)

    # --- Run batch inference and save to CSV ---
    output_csv_path = "predictions.csv"
    predictions_df = batch_inference(df, model, numeric_features, categorical_features,
                                     cat_encoders, cat_cardinalities, target_mapping,
                                     seq_len=cfg['data']['seq_len'],
                                     location_col=cfg['data'].get('location_col', 'location'),
                                     time_column=cfg['data'].get('time_column', 'timestep'),
                                     device=device,
                                     output_path=output_csv_path)

    logging.info("Batch inference completed:")
    logging.info(predictions_df)
