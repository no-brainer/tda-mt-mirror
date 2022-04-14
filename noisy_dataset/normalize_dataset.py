import argparse
import unicodedata
import os

from spacy.lang.de import German
from spacy.lang.en import English
from spacy.lang.ru import Russian


os.environ["LC_ALL"] = "en_US.utf8"


def select_tokenizer(lang):
    if lang == "de":
        tokenizer = German()
    elif lang == "en":
        tokenizer = English()
    elif lang == "ru":
        tokenizer = Russian()
    else:
        raise ValueError(f"Unknown language: {lang}")

    return tokenizer


def main(args):
    tokenizer = select_tokenizer(args.language)
    with open(args.input_dataset, "r") as in_file, \
            open(args.output_dataset, "w") as out_file:
        for sent in in_file:
            sent = sent.strip()
            sent = unicodedata.normalize("NFC", sent)
            sent = sent.lower()

            doc = tokenizer(sent.strip())
            tokens = [token.text for token in doc]
            sent = " ".join(tokens)

            out_file.write(sent)
            out_file.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("language", type=str, choices=["de", "en", "ru"])
    parser.add_argument("input_dataset", type=str)
    parser.add_argument("output_dataset", type=str)

    script_args = parser.parse_args()

    main(script_args)
