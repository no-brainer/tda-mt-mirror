from typing import Optional

from src.base import BaseDataset


class CustomDataset(BaseDataset):

    def __init__(self, src_path: str, trg_path: str, tokenizer, max_length: int = 512, limit: Optional[int] = None):
        examples = []
        with open(src_path, "r") as src_file, \
                open(trg_path, "r") as trg_file:
            for i, (src_sent, trg_sent) in enumerate(zip(src_file, trg_file), 1):
                examples.append((src_sent, trg_sent))

        super(CustomDataset, self).__init__(examples, tokenizer, max_length, limit)
