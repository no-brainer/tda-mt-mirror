import csv
import logging


logger = logging.getLogger(__name__)


def split_tsv(input_filename, tgt_filename, src_filename, tgt_lang, src_lang):
    """
    Expected format:
        tgt_lang    src_lang    sentence in tgt lang    sentence in src lang
    OR
        src_lang    tgt_lang    sentence in src lang    sentence in tgt lang
    """
    tgt_file = open(tgt_filename, "w")
    src_file = open(src_filename, "w")

    lines_used = 0
    with open(input_filename, "r") as tsv:
        for i, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
            if len(line) != 4:
                logger.warning(f"Incorrect amount of values in file {input_filename} (line {i})")
                continue
            if tgt_lang not in line[:2] or src_lang not in line[:2]:
                logger.warning(f"Wrong language labels in file {input_filename} (line {i})")
                continue

            out_files = [tgt_file, src_file]
            if line[0] != tgt_lang:
                out_files[0], out_files[1] = out_files[1], out_files[0]

            for k in [0, 1]:
                out_files[k].write(line[2 + k])
                out_files[k].write("\n")

            lines_used += 1

    return lines_used


def tsv_sentence_pairs(input_filename, tgt_lang, src_lang):
    with open(input_filename, "r") as tsv:
        for i, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
            if tgt_lang not in line[:2] or src_lang not in line[:2]:
                logger.warning(f"Wrong language labels in file {input_filename} (line {i})")
                continue
            if line[0] == tgt_lang:
                yield line[2], line[3]
            else:
                yield line[3], line[2]
