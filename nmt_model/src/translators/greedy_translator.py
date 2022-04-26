from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence

from src.base import BaseTranslator


class GreedyTranslator(BaseTranslator):

    @torch.no_grad()
    def translate(self, src_sent: str) -> str:
        prediction = [self.bos_id]
        src_encoded = torch.as_tensor(self.tokenizer.encode_src(src_sent), dtype=torch.long)

        batch = dict(
            src_enc=src_encoded.unsqueeze(0).to(self.device),
            src_enc_length=torch.as_tensor([src_encoded.size(-1)], dtype=torch.long),
        )
        for _ in range(self.max_length):
            batch["trg_enc"] = torch.as_tensor([prediction], dtype=torch.long).to(self.device)
            batch["trg_enc_length"] = torch.as_tensor([len(prediction)], dtype=torch.long)

            output = self.model(**batch)

            next_value = output[0, -1].argmax().item()
            prediction.append(next_value)
            if next_value == self.eos_id:  # end of string
                break

        prediction = prediction[1:len(prediction) - 1]
        return self.tokenizer.decode_trg(prediction)

    @torch.no_grad()
    def translate_batch(self, src_sents: List[str]) -> List[str]:
        predictions = [[self.bos_id] for _ in range(len(src_sents))]
        src_encoded = self.tokenizer.encode_src(src_sents)
        batch = dict(
            src_enc=pad_sequence(
                [torch.as_tensor(src_enc, dtype=torch.long) for src_enc in src_encoded],
                batch_first=True, padding_value=self.pad_id
            ).to(self.device),
            src_enc_length=torch.as_tensor([len(src_enc_sent) for src_enc_sent in src_encoded], dtype=torch.long)
        )
        for i in range(self.max_length):
            batch["trg_enc"] = torch.as_tensor(predictions, dtype=torch.long).to(self.device)
            trg_lengths = [self._safe_index(trg_sent, self.eos_id, len(trg_sent)) for trg_sent in predictions]
            batch["trg_enc_length"] = torch.as_tensor(trg_lengths, dtype=torch.long)
            if torch.all(batch["trg_enc_length"] - 1 < i):
                break

            output = self.model(**batch)

            next_values = output[:, -1].argmax(dim=-1).cpu().tolist()
            for j, next_value in enumerate(next_values):
                predictions[j].append(next_value)

        for i in range(len(predictions)):
            pred_length = self._safe_index(predictions[i], self.eos_id, len(predictions[i]))
            predictions[i] = predictions[i][1:pred_length]

        return self.tokenizer.decode_trg(predictions)
