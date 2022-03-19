from typing import List, Union

from tokenizers import Tokenizer


class BaseTokenizer:
    """
    idk, might want different tokenizer combinations later
    """
    def __init__(self, src_tokenizer: Tokenizer, trg_tokenizer: Tokenizer):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    @staticmethod
    def _encode(tokenizer: Tokenizer, sent: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        is_flat = isinstance(sent, str)
        if is_flat:
            sent = [sent]

        results = []
        for result in tokenizer.encode_batch(sent):
            results.append(result.ids)

        if is_flat:
            results = results[0]

        return results

    @staticmethod
    def _decode(tokenizer: Tokenizer, enc_sent: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        if isinstance(enc_sent[0], int):
            return tokenizer.decode(enc_sent)
        return tokenizer.decode_batch(enc_sent)

    def encode_src(self, sent):
        return BaseTokenizer._encode(self.src_tokenizer, sent)

    def encode_trg(self, sent):
        return BaseTokenizer._encode(self.trg_tokenizer, sent)

    def decode_src(self, enc_sent):
        return BaseTokenizer._decode(self.src_tokenizer, enc_sent)

    def decode_trg(self, enc_sent):
        return BaseTokenizer._decode(self.trg_tokenizer, enc_sent)
