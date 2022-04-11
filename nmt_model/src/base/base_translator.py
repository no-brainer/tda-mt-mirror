from abc import abstractmethod

from src.base import BaseModel, BaseTokenizer


class BaseTranslator:

    def __init__(self, model: BaseModel, tokenizer: BaseTokenizer, device: str, bos_id: int = 2, eos_id: int = 3, max_length: int = 512):
        self.max_length = max_length
        self.eos_id = eos_id
        self.bos_id = bos_id

        self.device = device

        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def translate(self, src_sent: str) -> str:
        raise NotImplementedError
