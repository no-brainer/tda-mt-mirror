import torch

from src.base.base_dataset import BaseDataset


class BucketSampler(torch.utils.data.Sampler):

    def __init__(self, datasource: BaseDataset, n_bins: int, batch_size: int):
        self.batch_size = batch_size
        self.n_bins = n_bins

        tokenizer = datasource.tokenizer
        self.indices = sorted(
            list(range(len(datasource))),
            key=lambda idx: len(tokenizer.encode_src(datasource[idx][0]))
        )

        self.bin_sampler = torch.utils.data.RandomSampler(range(n_bins))

        self.bin_size = len(self.indices) // n_bins
        self.n_batches = n_bins * (self.bin_size // batch_size)

    def __iter__(self):
        for bin_idx in self.bin_sampler:
            start_idx = self.bin_size * bin_idx
            end_idx = start_idx + self.bin_size
            sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(self.indices[start_idx: end_idx]),
                batch_size=self.batch_size,
                drop_last=True
            )
            for batch in sampler:
                yield batch

    def __len__(self):
        return self.n_batches
