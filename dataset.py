import pandas as pd
import torch
from torch.utils.data import Dataset


class ContrastiveVectorDataset(Dataset):
    def __init__(self, pkl_path, negative_sample_size=10):
        raw_data = pd.read_pickle(pkl_path)  # List of (q_vec, [kw_vecs])

        self.data = []
        self.keyword_pool = []

        for q_vec, kw_vecs in raw_data:
            q_tensor = torch.tensor(q_vec, dtype=torch.float32)
            kw_tensors = [torch.tensor(kw, dtype=torch.float32).squeeze() for kw in kw_vecs]
            self.data.append((q_tensor, kw_tensors))
            self.keyword_pool.extend(kw_tensors)

        self.keyword_pool = torch.stack(self.keyword_pool)
        self.negative_sample_size = negative_sample_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q_vec, pos_kw_vecs = self.data[idx]
        pos_vec = pos_kw_vecs[torch.randint(0, len(pos_kw_vecs), (1,)).item()]

        # negative sampling
        neg_indices = torch.randint(0, len(self.keyword_pool), (self.negative_sample_size,))
        neg_kw_vecs = self.keyword_pool[neg_indices]

        return q_vec, pos_vec, neg_kw_vecs
