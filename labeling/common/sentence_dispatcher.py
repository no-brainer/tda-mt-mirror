import linecache
import logging
import os


class SentencePairDispatcher:
    def __init__(self, corpus_path, src_lang="eng", trg_lang="rus"):
        self.corpus_path = corpus_path
        self.src_lang = src_lang
        self.trg_lang = trg_lang

        self.data_format = "unknown"
        basename = os.path.basename(self.corpus_path).lower()
        if "tatoeba" in basename:
            self.data_format = "tatoeba"
        elif "wikimatrix" in basename:
            self.data_format = "wikimatrix"
        logging.info(f"Dataset format is {self.data_format}")

    def __call__(self, line_idx):
        logging.info(f"Extracting sentence pair from line {line_idx}")
        line = linecache.getline(self.corpus_path, line_idx + 1)
        parts = list(map(str.strip, line.split("\t")))
        return getattr(self, "extract_" + self.data_format)(line_idx, parts)

    def extract_tatoeba(self, line_idx, parts):
        if len(parts) != 4:
            logging.warning(f"Expected 4 parts, got {len(parts)} (line {line_idx})")
            return "", ""
        if sorted([self.src_lang, self.trg_lang]) != sorted(parts[:2]):
            logging.warning(f"Expected languages {self.src_lang}-{self.tgt_lang}, got {parts[0]}-{parts[1]} (line {line_idx})")
            return "", ""
        result = parts[2:]
        if parts[0] != self.src_lang:
            result[0], result[1] = result[1], result[0]
        return result
    
    def extract_wikimatrix(self, line_idx, parts):
        if len(parts) < 3:
            logging.warning(f"Expected 3 parts, got {len(parts)} (line {line_idx})")
            return "", ""
        result = parts[1:]
        if self.src_lang > self.trg_lang:
            result[0], result[1] = result[1], result[0]
        return result
    
    def extract_unknown(self):
        return "", ""
