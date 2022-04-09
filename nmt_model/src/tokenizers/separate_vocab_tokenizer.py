import os
from typing import Dict

from src.base import BaseTokenizer
from src.tokenizers.utils import prepare_tokenizer


class SeparateVocabTokenizer(BaseTokenizer):

    def __init__(self, save_path: str, src_model_args: Dict, trg_model_args: Dict):
        src_tokenizer = prepare_tokenizer(os.path.join(save_path, "src"), src_model_args)
        trg_tokenizer = prepare_tokenizer(os.path.join(save_path, "trg"), trg_model_args)
        super(SeparateVocabTokenizer, self).__init__(src_tokenizer, trg_tokenizer)
