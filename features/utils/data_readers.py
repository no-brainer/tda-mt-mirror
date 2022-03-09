import csv
import json
import logging
import os


logger = logging.getLogger(__name__)


# Tools for Tatoeba format
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


def tatoeba_sentence_pairs(input_filename, tgt_lang, src_lang):
    with open(input_filename, "r") as tsv:
        for i, line in enumerate(csv.reader(tsv, dialect="excel-tab")):
            if tgt_lang not in line[:2] or src_lang not in line[:2]:
                logger.warning(f"Wrong language labels in file {input_filename} (line {i})")
                continue
            result = [i, line[2], line[3]]
            if line[0] != tgt_lang:
                result[1], result[2] = result[2], result[1]
            yield result


# WikiMatrix tools
def wikimatrix_sentence_pairs(input_filename, tgt_lang, src_lang, thresh=1.05):
    lang1, lang2 = os.path.basename(input_filename).split(".")[1].split("-")
    with open(input_filename, "r") as tsv:
        for i, line in enumerate(csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE)):
            if len(line) < 3:
                logger.warning(f"Incorrect amount of values in file {input_filename} (line {i}, {len(line)} values)")
                continue
            if float(line[0]) <= thresh:
                logger.warning(f"Margin score {float(line[0]):.4f} for line {i} is less than threshold")
                continue
            result = [i, line[1], line[2]]
            if not tgt_lang.startswith(lang1):
                result[1], result[2] = result[2], result[1]
            yield result


# Common
def tsv_sentence_pairs(input_filename, tgt_lang, src_lang, batch_size=1):
    gen = None
    if input_filename.find("tatoeba") != -1:
        gen = tatoeba_sentence_pairs
    elif input_filename.find("WikiMatrix") != -1:
        gen = wikimatrix_sentence_pairs
    else:
        logger.error(f"Unknown dataset: {input_filename}")
        raise ValueError("Unknown dataset")

    buffer = [[], [], []]
    for val in gen(input_filename, tgt_lang, src_lang):
        for i in range(len(buffer)):
            buffer[i].append(val[i])
        
        if len(buffer[0]) >= batch_size:
            yield buffer
            buffer = [[], [], []]


def wmt19_qe_reader(input_path):
    used_fields = {
        "src": "src", "mt": "tr", "pe": "ref", "hter": "hter"
    }

    # every value is in its own file
    input_files = dict()
    for filename in os.listdir(input_path):
        ext = filename.rsplit(".", maxsplit=1)[-1]
        if ext in used_fields.keys():
            full_path = os.path.join(input_path, filename)
            input_files[ext] = open(full_path, "r")

    if len(input_files) != len(used_fields):
        raise ValueError(
            f"But input path {input_path}. "
            f"Only found files with extensions {', '.join(input_files.keys())}, "
            f"out of the required {', '.join(used_fields.keys())}"
        )

    idx = 0
    while True:
        data = dict(line_idx=idx)
        has_reached_eof = False
        for field in used_fields.keys():
            value = input_files[field].readline()
            if len(value) == 0:
                has_reached_eof = True
                break

            value = value.strip()
            if field == "hter":
                value = float(value)
            data[used_fields[field]] = value

        if has_reached_eof:
            break

        yield data
        idx += 1

    for input_file in input_files.values():
        input_file.close()


# translation readers
def wikihades(input_path):
    with open(input_path, "r") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            data["line_idx"] = i
            data["text"] = data["replaced"].replace("==", "")
            data.pop("replaced")
            yield data


def wmt19_format(input_path):
    with open(input_path, "r") as input_file:
        for i, line in enumerate(input_file):
            yield {
                "line_idx": i,
                "text": line.strip(),
            }


def scarecrow_format(input_path):
    with open(input_path, "r") as input_file:
        for row in csv.reader(input_file):
            if row[0] == "id":  # header
                continue

            yield {
                "line_idx": int(row[0]),
                "text": " ".join([row[2], row[3]]),
            }


def custom_dataset_format(input_path):
    with open(input_path, "r") as input_file:
        for i, row in enumerate(csv.reader(input_file, dialect="excel-tab")):
            if row[0] == "sentence":
                continue

            yield {
                "line_idx": i,
                "text": row[0],
            }
