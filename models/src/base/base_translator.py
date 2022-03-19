class BaseTranslator:

    def __init__(self, model, tokenizer, device, max_length=512):
        self.max_length = max_length

        self.device = device

        self.model = model
        self.tokenizer = tokenizer

    def translate(self, src_sent: str) -> str:
        raise NotImplementedError
