import copy
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)


def clamp_categorical_features(X_cat: torch.Tensor, cat_cardinalities: List[int]) -> torch.Tensor:
    """
    Clamp categorical features to be within valid range based on cardinalities.

    Args:
        X_cat (torch.Tensor): Categorical features tensor of shape (batch, seq_len, num_cats).
        cat_cardinalities (List[int]): List of cardinalities for each categorical feature.

    Returns:
        torch.Tensor: Clamped categorical features.
    """
    for i, n_cat in enumerate(cat_cardinalities):
        X_cat[:, :, i] = torch.clamp(X_cat[:, :, i], 0, n_cat - 1)
    return X_cat


def train_one_epoch(model: nn.Module, loader: 'torch.utils.data.DataLoader',
                    criterion: nn.Module, optimizer: optim.Optimizer,
                    cat_cardinalities: List[int], device: torch.device) -> tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): PyTorch model.
        loader (DataLoader): Data loader for training data.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer.
        cat_cardinalities (List[int]): Categorical feature cardinalities.
        device (torch.device): Device to run the training on.

    Returns:
        tuple: Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for (X_num, X_cat, cat_mask), y in loader:
        X_num, X_cat, cat_mask, y = X_num.to(device), X_cat.to(device), cat_mask.to(device), y.to(device)
        X_cat = clamp_categorical_features(X_cat, cat_cardinalities)

        optimizer.zero_grad()
        preds = model(X_num, X_cat, cat_mask)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds.argmax(1) == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate_model(model: nn.Module, loader: 'torch.utils.data.DataLoader',
                   criterion: nn.Module, cat_cardinalities: List[int],
                   device: torch.device) -> tuple[float, float]:
    """
    Evaluate the model on a validation set.

    Args:
        model (nn.Module): PyTorch model.
        loader (DataLoader): Data loader for validation data.
        criterion (nn.Module): Loss function.
        cat_cardinalities (List[int]): Categorical feature cardinalities.
        device (torch.device): Device to run the evaluation on.

    Returns:
        tuple: Average validation loss and accuracy.
    """
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for (X_num, X_cat, cat_mask), y in loader:
            X_num, X_cat, cat_mask, y = X_num.to(device), X_cat.to(device), cat_mask.to(device), y.to(device)
            X_cat = clamp_categorical_features(X_cat, cat_cardinalities)

            preds = model(X_num, X_cat, cat_mask)
            loss = criterion(preds, y)
            val_loss += loss.item()
            val_correct += (preds.argmax(1) == y).sum().item()
            val_total += y.size(0)

    avg_val_loss = val_loss / len(loader)
    val_accuracy = val_correct / val_total
    return avg_val_loss, val_accuracy


def log_metrics(writer: Optional[SummaryWriter], train_loss: float, train_acc: float,
                val_loss: float, val_acc: float, epoch: int) -> None:
    """
    Log metrics to TensorBoard if writer is provided.

    Args:
        writer (Optional[SummaryWriter]): TensorBoard writer object.
        train_loss (float): Training loss.
        train_acc (float): Training accuracy.
        val_loss (float): Validation loss.
        val_acc (float): Validation accuracy.
        epoch (int): Current epoch number.
    """
    if writer:
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)


def train_model(model: nn.Module, train_loader: 'torch.utils.data.DataLoader',
                val_loader: 'torch.utils.data.DataLoader',
                cat_cardinalities: List[int], device: torch.device,
                lr: float = 1e-3, epochs: int = 50, patience: int = 3,
                checkpoint_path: str = "best_model.pth", use_tensorboard: bool = True
                ) -> nn.Module:
    """
    Train a PyTorch model with early stopping and optional TensorBoard logging.

    Args:
        model (nn.Module): PyTorch model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        cat_cardinalities (List[int]): List of categorical feature cardinalities.
        device (torch.device): Device to run the training on.
        lr (float): Learning rate.
        epochs (int): Maximum number of epochs.
        patience (int): Early stopping patience.
        checkpoint_path (str): File path to save the best model.
        use_tensorboard (bool): Whether to log metrics to TensorBoard.

    Returns:
        nn.Module: Model loaded with the best validation weights.
    """
    writer: Optional[SummaryWriter] = SummaryWriter() if use_tensorboard else None
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # Train & Validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cat_cardinalities, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, cat_cardinalities, device)

        logging.info(f"Epoch {epoch + 1}/{epochs} | "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2%} | "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2%}")

        # TensorBoard logging
        log_metrics(writer, train_loss, train_acc, val_loss, val_acc, epoch)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, checkpoint_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_model_state:
        model.load_state_dict(best_model_state)
    if writer:
        writer.close()
    return model
