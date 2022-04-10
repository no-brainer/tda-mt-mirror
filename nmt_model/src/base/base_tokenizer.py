from typing import List, Union

import youtokentome as yttm


class BaseTokenizer:
    """
    idk, might want different tokenizer combinations later
    """
    def __init__(self, src_tokenizer: yttm.BPE, trg_tokenizer: yttm.BPE):
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    @staticmethod
    def _encode(tokenizer: yttm.BPE, sent: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        is_flat = isinstance(sent, str)
        if is_flat:
            sent = [sent]

        results = tokenizer.encode(sent, output_type=yttm.OutputType.ID, bos=True, eos=True)
        if is_flat:
            results = results[0]

        return results

    @staticmethod
    def _decode(tokenizer: yttm.BPE, enc_sent: Union[List[int], List[List[int]]]) -> Union[str, List[str]]:
        is_flat = isinstance(enc_sent[0], int)
        if is_flat:
            enc_sent = [enc_sent]

        results = tokenizer.decode(enc_sent)
        if is_flat:
            results = results[0]

        return results

    def encode_src(self, sent):
        return BaseTokenizer._encode(self.src_tokenizer, sent)

    def encode_trg(self, sent):
        return BaseTokenizer._encode(self.trg_tokenizer, sent)

    def decode_src(self, enc_sent):
        return BaseTokenizer._decode(self.src_tokenizer, enc_sent)

    def decode_trg(self, enc_sent):
        return BaseTokenizer._decode(self.trg_tokenizer, enc_sent)
