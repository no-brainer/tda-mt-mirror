from typing import List, Tuple, Dict, Optional

import numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, examples: List[Tuple[str, str]], tokenizer, max_length: int = 512, limit: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.dataset_size = len(examples)
        if limit is not None:
            self.dataset_size = min(self.dataset_size, limit)
            idx = np.random.choice(len(examples), size=self.dataset_size, replace=False)
            self.examples = [examples[i] for i in idx]
        else:
            self.examples = examples

    def __getitem__(self, idx: int) -> Dict:
        src_sent, trg_sent = self.examples[idx]

        src_sent = src_sent.strip()
        trg_sent = trg_sent.strip()

        src_enc = self.tokenizer.encode_src(src_sent)[:self.max_length]
        trg_enc = self.tokenizer.encode_trg(trg_sent)[:self.max_length]

        return {
            "src_text": src_sent,
            "trg_text": trg_sent,
            "src_enc": torch.LongTensor(src_enc),
            "trg_enc": torch.LongTensor(trg_enc),
        }

    def __len__(self) -> int:
        return self.dataset_size
