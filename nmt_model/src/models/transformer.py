import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import PositionalEncoding
from src.base import BaseModel


class NMTTransformer(BaseModel):

    def __init__(
            self,
            input_vocab_size: int,
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
        padding_idx = kwargs.get("padding_idx", input_vocab_size)

        self.embs = nn.Embedding(input_vocab_size + 1, d_model, padding_idx=padding_idx)
        self.pos_enc = PositionalEncoding(d_model, dropout_enc)

        custom_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, activation=activation,
                                       dropout=dropout_transformer, batch_first=True),
            num_encoder_layers
        )
        custom_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, activation=activation,
                                       dropout=dropout_transformer, batch_first=True),
            num_decoder_layers
        )

        self.transformer = nn.Transformer(
            d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout_transformer,
            activation, batch_first=True, custom_encoder=custom_encoder, custom_decoder=custom_decoder
        )
        self.decoder = nn.Linear(d_model, input_vocab_size + 1)

    @staticmethod
    def _length_mask(max_size: int, lengths: torch.LongTensor) -> torch.Tensor:
        return torch.arange(max_size)[None, :] >= lengths.view(-1, 1)

    def forward(self, src_encoded, trg_encoded, *args, **kwargs):
        src_emb = self.pos_enc(self.embs(src_encoded))
        trg_emb = self.pos_enc(self.embs(trg_encoded))

        device = src_encoded.device

        src_mask = self.transformer.generate_square_subsequent_mask(src_emb.size(1))
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_emb.size(1))
        src_mask, trg_mask = src_mask.to(device), trg_mask.to(device)

        src_padding_mask = self._length_mask(src_emb.size(1), kwargs["src_length"])
        src_padding_mask = src_padding_mask.to(device)
        trg_padding_mask = self._length_mask(trg_emb.size(1), kwargs["trg_length"])
        trg_padding_mask = trg_padding_mask.to(device)

        out = self.transformer(
            src_emb,
            trg_emb,
            src_mask=src_mask,
            tgt_mask=trg_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=trg_padding_mask,
        )
        return self.decoder(out)
