import torch
import yaml
import logging
from torch.utils.data import DataLoader

from modelling.error_prediction.data_utils import load_and_preprocess, train_val_test_split_by_location, set_seed
from modelling.error_prediction.dataset import SlidingWindowDataset
from modelling.error_prediction.model import LSTMClassifier
from modelling.error_prediction.train_functions import train_model
from modelling.error_prediction.evaluate import evaluate_model, one_vs_all_accuracy
from modelling.error_prediction.predict import prepare_sequence, predict_next

logging.basicConfig(level=logging.INFO)


def main(config_path: str = "../../config/error_prediction_config.yaml"):

    # --- Load configuration ---
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    logging.info("Configuration loaded.")

    SEED = cfg.get('seed', 42)
    set_seed(SEED)
    logging.info(f"Random seed set to {SEED}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # --- Load and preprocess data ---
    df, numeric_features, categorical_features, cat_encoders, target_mapping, cat_cardinalities = \
        load_and_preprocess(cfg['data']['path'], cfg['target']['column'])
    logging.info("Data loaded and preprocessed.")

    # --- Train/val/test split ---
    df_train, df_val, df_test = train_val_test_split_by_location(df)
    logging.info("Data split into train/val/test sets.")

    # --- Datasets and loaders ---
    train_dataset = SlidingWindowDataset(df_train, numeric_features, categorical_features, seq_len=cfg['data']['seq_len'])
    val_dataset = SlidingWindowDataset(df_val, numeric_features, categorical_features, seq_len=cfg['data']['seq_len'])
    test_dataset = SlidingWindowDataset(df_test, numeric_features, categorical_features, seq_len=cfg['data']['seq_len'])

    train_loader = DataLoader(train_dataset, batch_size=cfg['data']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg['data']['batch_size'], shuffle=False)
    logging.info("DataLoaders created.")

    # --- Initialize model ---
    n_numeric = len(numeric_features)
    n_classes = len(target_mapping)
    model = LSTMClassifier(n_numeric, cat_cardinalities, embedding_dim=cfg['model']['embedding_dim'],
                           hidden_dim=cfg['model']['hidden_dim'], n_classes=n_classes).to(device)
    logging.info("Model initialized.")

    # --- Train model ---
    model = train_model(model, train_loader, val_loader, cat_cardinalities, device,
                        lr=cfg['training']['lr'], epochs=cfg['training']['epochs'],
                        patience=cfg['training']['patience'], checkpoint_path="best_model.pth",
                        use_tensorboard=True)
    logging.info("Training completed.")

    # --- Evaluate model ---
    y_true, y_pred = evaluate_model(model, test_loader, cat_cardinalities, device, cfg['target']['label_mapping'])
    one_vs_all_accuracy(y_true, y_pred, target_label='correct', class_lookup=cfg['target']['label_mapping'])
    logging.info("Evaluation completed.")

    # --- Example: next-step prediction ---
    last_seq = df_test.groupby('location').tail(cfg['data']['seq_len'])
    X_num_tensor, X_cat_tensor, cat_mask_tensor = prepare_sequence(
        last_seq, numeric_features, categorical_features, cat_encoders, cat_cardinalities,
        seq_len=cfg['data']['seq_len'], device=device
    )
    predicted_label = predict_next(model, X_num_tensor, X_cat_tensor, cat_mask_tensor, cfg['target']['label_mapping'])
    logging.info(f"Example next-step prediction: {predicted_label}")


if __name__ == "__main__":
    main()
