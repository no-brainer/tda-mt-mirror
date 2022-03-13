import torch
from torch.nn.utils.rnn import pad_sequence


class PadCollatorFn:

    def __init__(self, padding_value=0, padded_keys=None):
        self.padding_value = padding_value
        self.padded_keys = padded_keys
        if padded_keys is None:
            self.padded_keys = ["src", "ref"]

    def __call__(self, batch):
        result = dict()

        for key in batch[0].keys():
            key_values = [obj[key] for obj in batch]
            if key not in self.padded_keys:
                result[key] = key_values
                continue

            result[key] = pad_sequence(key_values, batch_first=True, padding_value=self.padding_value)
            result[f"{key}_length"] = torch.LongTensor([len(enc_seq) for enc_seq in key_values])

        return result
