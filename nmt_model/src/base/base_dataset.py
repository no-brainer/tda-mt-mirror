from typing import List, Tuple, Dict, Optional

import numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, examples: List[Tuple[str, str]], tokenizer, split: str, limit: Optional[int] = None):
        self.tokenizer = tokenizer

        n_train = int(len(examples) * 0.9)
        if split == "train":
            examples = examples[:n_train]
        else:
            examples = examples[n_train:]

        self.dataset_size = len(examples)
        if limit is not None:
            self.dataset_size = min(self.dataset_size, limit)
            idx = np.random.choice(len(examples), size=self.dataset_size, replace=False)
            self.examples = [examples[i] for i in idx]
        else:
            self.examples = examples

    def __getitem__(self, idx: int) -> Dict:
        src_sent, trg_sent = self.examples[idx]
        return {
            "src_text": src_sent,
            "trg_text": trg_sent,
            "src_enc": self.tokenizer.encode_src(src_sent),
            "trg_enc": self.tokenizer.encode_trg(trg_sent),
        }

    def __len__(self) -> int:
        return self.dataset_size
