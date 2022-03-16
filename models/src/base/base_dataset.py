import numpy as np
import torch


class BaseDataset(torch.utils.data.Dataset):

    def __init__(self, examples, tokenizer, limit=None):
        self.tokenizer = tokenizer

        self.dataset_size = len(self.examples)
        if limit is not None:
            self.dataset_size = min(self.dataset_size, limit)
            idx = np.random.choice(len(examples), size=self.dataset_size, replace=False)
            self.examples = [examples[i] for i in idx]
        else:
            self.examples = examples

    def __getitem__(self, idx):
        src_sent, trg_sent = self.examples[idx]
        src_enc, trg_enc = self.tokenizer(src_sent, trg_sent)
        return {
            "src_text": src_sent,
            "trg_text": trg_sent,
            "src_enc": src_enc,
            "trg_enc": trg_enc,
        }

    def __len__(self):
        return self.dataset_size
