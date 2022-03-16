import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        return self.tokenizer(self.examples[idx])

    def __len__(self):
        return self.dataset_size
