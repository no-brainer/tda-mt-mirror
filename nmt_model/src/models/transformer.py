import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import PositionalEncoding
from src.base import BaseModel


class NMTTransformer(BaseModel):

    def __init__(
            self,
            trg_vocab_size: int,
            src_vocab_size: int,
            d_model: int,
            nhead: int,
            num_encoder_layers: int,
            num_decoder_layers: int,
            dim_feedforward: int,
            dropout_enc: float = 0.1,
            dropout_transformer: float = 0.1,
            *args,
            **kwargs):
        super().__init__()

        activation = kwargs.get("activation", "gelu")
        padding_idx = kwargs.get("padding_idx", 0)

        self.trg_embs = nn.Embedding(trg_vocab_size, d_model, padding_idx=padding_idx)
        emb_type = kwargs.get("emb_type", "shared")
        if emb_type == "shared":
            self.src_embs = self.trg_embs
        else:
            self.src_embs = nn.Embedding(src_vocab_size, d_model, padding_idx=padding_idx)

        max_length = kwargs.get("max_length", 512)
        self.pos_enc = PositionalEncoding(d_model, dropout_enc, max_len=max_length)

        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout_transformer,
            activation, batch_first=True
        )
        self.decoder = nn.Linear(d_model, trg_vocab_size)

    @staticmethod
    def _length_mask(max_size: int, lengths: torch.LongTensor) -> torch.Tensor:
        return torch.arange(max_size)[None, :] >= lengths.view(-1, 1)

    def forward(self, src_enc, trg_enc, src_enc_length, trg_enc_length, *args, **kwargs):
        src_emb = self.pos_enc(self.src_embs(src_enc))
        trg_emb = self.pos_enc(self.trg_embs(trg_enc))

        device = src_enc.device

        trg_mask = self.transformer.generate_square_subsequent_mask(trg_emb.size(1))
        trg_mask = trg_mask.to(device)

        src_padding_mask = self._length_mask(src_emb.size(1), src_enc_length).to(device)
        trg_padding_mask = self._length_mask(trg_emb.size(1), trg_enc_length).to(device)

        out = self.transformer(
            src_emb,
            trg_emb,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
        )
        return self.decoder(out)
