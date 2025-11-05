import torch
import torch.nn as nn
from typing import List


class LSTMClassifier(nn.Module):
    """
    LSTM classifier for mixed numeric and categorical time series features.
    """
    def __init__(self, n_numeric: int, cat_cardinalities: List[int], embedding_dim: int = 8,
                 hidden_dim: int = 64, n_classes: int = 5):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=n_cat, embedding_dim=embedding_dim, padding_idx=0)
            for n_cat in cat_cardinalities
        ])
        self.n_emb_total = embedding_dim * len(cat_cardinalities)
        self.lstm_input_dim = n_numeric + self.n_emb_total

        self.lstm = nn.LSTM(input_size=self.lstm_input_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, cat_mask: torch.Tensor = None) -> torch.Tensor:
        embs = [emb(x_cat[:, :, i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(embs, dim=2)
        if cat_mask is not None:
            mask = cat_mask.unsqueeze(-1)
            mask = mask.repeat(1, 1, 1, x_emb.shape[-1] // mask.shape[2])
            x_emb = x_emb * mask.reshape_as(x_emb)
        x = torch.cat([x_num, x_emb], dim=2)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
