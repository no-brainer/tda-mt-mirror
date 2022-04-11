import torch

from src.base import BaseTranslator


class GreedyTranslator(BaseTranslator):

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

        return self.tokenizer.decode_trg(prediction)
