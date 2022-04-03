from typing import Dict

from src.base import BaseTokenizer
from src.tokenizers.utils import prepare_tokenizer


class SharedVocabTokenizer(BaseTokenizer):

    def __init__(self, model_name: str, save_path: str, model_args: Dict, trainer_args: Dict):
        assert model_name in ["BPE", "WordPiece"]

        tokenizer = prepare_tokenizer(save_path, model_name, model_args, trainer_args)
        super(SharedVocabTokenizer, self).__init__(tokenizer, tokenizer)
