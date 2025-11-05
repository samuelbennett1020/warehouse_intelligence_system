import torch
import pandas as pd
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


class SlidingWindowDataset(Dataset):
    """
    Dataset for creating sliding window sequences for time series data.
    """
    def __init__(self, df: pd.DataFrame, numeric_features: List[str], categorical_features: List[str],
                 target_col: str = 'status', seq_len: int = 10, location_col: str = 'location', time_column: str = 'timestep'):
        self.seq_len = seq_len
        df = df.sort_values([location_col, time_column]).reset_index(drop=True)
        locs = df[location_col].values
        X_num_all = df[numeric_features].values.astype('float32')
        X_cat_all = df[categorical_features].values.astype('int64')
        y_all = df[target_col].values.astype('int64')

        self.X_num, self.X_cat, self.cat_mask, self.y = [], [], [], []

        unique_locs, loc_starts = np.unique(locs, return_index=True)
        loc_ends = np.append(loc_starts[1:], len(locs))

        for start, end in zip(loc_starts, loc_ends):
            length = end - start
            if length < seq_len + 1:
                continue
            for i in range(length - seq_len):
                seq_num = X_num_all[start+i:start+i+seq_len]
                seq_cat = X_cat_all[start+i:start+i+seq_len]
                target = y_all[start+i+seq_len]
                mask = np.where(seq_cat == 0, 0.0, 1.0).astype('float32')

                self.X_num.append(torch.tensor(seq_num, dtype=torch.float32))
                self.X_cat.append(torch.tensor(seq_cat, dtype=torch.long))
                self.cat_mask.append(torch.tensor(mask, dtype=torch.float32))
                self.y.append(torch.tensor(target, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        return (self.X_num[idx], self.X_cat[idx], self.cat_mask[idx]), self.y[idx]
