import argparse
import unicodedata
import os


os.environ["LC_ALL"] = "en_US.utf8"


def main(args):
    with open(args.input_dataset, "r") as in_file, \
            open(args.output_dataset, "w") as out_file:
        for sent in in_file:
            sent = sent.decode("utf-8")
            sent = unicodedata.normalize("NFC", sent)
            sent = sent.lower()
            sent = sent.encode("utf-8")
            out_file.write(sent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_dataset", type=str)
    parser.add_argument("output_dataset", type=str)

    script_args = parser.parse_args()

    main(script_args)
