import os
from typing import Optional
import zipfile

import gdown

from src.base import BaseDataset


class YandexDataset(BaseDataset):

    GDRIVE_ID = "1C7X1EzMmxgUbNJD6yZVz8sEKd8v8E2hx"

    def __init__(self, src_lang: str, trg_lang: str, datapath: str, tokenizer, limit: Optional[int] = None):
        if src_lang not in ["en", "ru"] or trg_lang not in ["en", "ru"]:
            raise ValueError(
                "This dataset only supports the following languages: en, ru"
            )

        corpus_path = os.path.join(datapath, "yandex-corpus")
        if not os.path.exists(corpus_path):
            YandexDataset._download_data(corpus_path)

        src_datapath = os.path.join(corpus_path, f"corpus.en_ru.1m.{src_lang}")
        trg_datapath = os.path.join(corpus_path, f"corpus.en_ru.1m.{trg_lang}")

        examples = []
        with open(src_datapath, "r") as src_file, \
                open(trg_datapath, "r") as trg_file:
            for i, (src_sent, trg_sent) in enumerate(zip(src_file, trg_file), 1):
                if limit is not None and i >= limit:
                    break
                examples.append((src_sent, trg_sent))

        super(YandexDataset, self).__init__(examples, tokenizer, limit)

    @staticmethod
    def _download_data(corpus_path: str):
        print("Downloading Yandex Dataset...")
        zip_path = os.path.join(corpus_path, "yandex-corpus.zip")
        gdown.download(
            id=YandexDataset.GDRIVE_ID,
            output=os.path.join(zip_path)
        )
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(corpus_path)

        os.remove(zip_path)
        print("Download is complete!")
