from typing import Dict

from src.base import BaseTokenizer
from src.tokenizers.utils import prepare_tokenizer


class SharedVocabTokenizer(BaseTokenizer):

    def __init__(self, save_path: str, model_args: Dict):
        tokenizer = prepare_tokenizer(save_path, model_args)
        super(SharedVocabTokenizer, self).__init__(tokenizer, tokenizer)
