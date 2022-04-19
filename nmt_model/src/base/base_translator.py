from abc import abstractmethod
from typing import Optional, List

from src.base import BaseModel, BaseTokenizer


class BaseTranslator:

    def __init__(self, model: BaseModel, tokenizer: BaseTokenizer, device: str, bos_id: int = 2, eos_id: int = 3,
                 pad_id: int = 0, max_length: int = 512):
        self.max_length = max_length
        self.eos_id = eos_id
        self.bos_id = bos_id
        self.pad_id = pad_id

        self.device = device

        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def translate(self, src_sent: str) -> str:
        raise NotImplementedError

    def translate_batch(self, src_sents: List[str]) -> List[str]:
        predictions = []
        for src_sent in src_sents:
            predictions.append(self.translate(src_sent))
        return predictions

    @staticmethod
    def _safe_index(arr: List, value: int, default_value: Optional[int] = None):
        try:
            result = arr.index(value)
        except ValueError:
            result = default_value
        return result
