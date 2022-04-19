import torch
import torch.nn.functional as F

from src.base import BaseTranslator, BaseModel, BaseTokenizer


class BeamSearchTranslator(BaseTranslator):

    def __init__(self, model: BaseModel, tokenizer: BaseTokenizer, device: str, bos_id: int = 2, eos_id: int = 3,
                 max_length: int = 512, beam_size: int = 5, temperature: float = 1.):
        self.beam_size = beam_size
        self.temperature = temperature

        super(BeamSearchTranslator, self).__init__(model, tokenizer, device, bos_id, eos_id, max_length)

    def translate(self, src_sent: str) -> str:
        src_encoded = torch.as_tensor(self.tokenizer.encode_src(src_sent), dtype=torch.long)

        top_beams = [
            ([self.bos_id], 0.),
        ]

        batch = {
            "src_enc": src_encoded.unsqueeze(0).to(self.device),
            "src_enc_length": torch.as_tensor([src_encoded.size(-1)], dtype=torch.long)
        }
        for i in range(self.max_length):
            trgs, beam_probs = list(zip(*top_beams))
            trg_lengths = [self._safe_index(trg, self.eos_id, len(trg)) for trg in trgs]

            if len(top_beams) != batch["src_enc"].size(0):
                batch["src_enc"] = torch.tile(batch["src_enc"], (self.beam_size, 1))
                batch["src_enc_length"] = torch.tile(batch["src_enc_length"], (self.beam_size,))

            batch["trg_enc"] = torch.as_tensor(trgs, dtype=torch.long).to(self.device)
            batch["trg_enc_length"] = torch.as_tensor(trg_lengths, dtype=torch.long)

            if torch.all(batch["trg_enc_length"] <= i):
                break

            outputs = self.model(**batch)
            outputs = outputs.detach().cpu()

            top_beams = []
            for trg, prev_prob, output in zip(trgs, beam_probs, outputs):
                log_probs = F.log_softmax(output[-1] / self.temperature, dim=-1)
                new_values = torch.topk(log_probs, self.beam_size)

                for j in range(self.beam_size):
                    new_trg = trg.copy()
                    new_trg.append(new_values.indices[j].item())

                    prob = prev_prob + new_values.values[j].item()
                    top_beams.append((new_trg, prob))

            top_beams.sort(key=lambda x: x[1], reverse=True)
            top_beams = top_beams[:self.beam_size]

        prediction = top_beams[0][0]
        prediction_length = self._safe_index(prediction, self.eos_id, len(prediction))
        prediction = prediction[1:prediction_length - 1]

        return self.tokenizer.decode_trg(prediction)
