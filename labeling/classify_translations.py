import argparse
import csv
from collections import defaultdict
import curses
import logging
import os
import sys
import textwrap

import sacrebleu

sys.path.append("../nmt_model")
import src.tokenizers
from src.utils import parse_config, init_obj


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="./manual_labeling.log", 
    level=logging.INFO,
)


def center_horiz(stdscr, y, line, attr=None):
    x = curses.COLS // 2 - len(line) // 2
    if attr is None:
        stdscr.addstr(y, x, line)
    else:
        stdscr.addstr(y, x, line, attr)


def wrap_text(text, max_width):
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        line_parts = textwrap.wrap(line, width=max_width - 1)
        new_lines.extend(line_parts)
    
    return "\n".join(new_lines)


class LabelingManager:

    def __init__(self, args):
        self.trg_datapath = args.trg_datapath
        self.translations_datapath = args.translations_datapath
        self.labels_datapath = args.labels_datapath

        training_config = parse_config(args.config_path)
        self.tokenizer = init_obj(src.tokenizers, training_config["tokenizer"])

    def __call__(self, stdscr):
        curses.curs_set(0)
        is_exiting = self._show_help(stdscr)
        if is_exiting:
            return
        offset, total = self._load_data()
        self._run_labeling(stdscr, offset, total)

    def _load_data(self):
        total = 0
        with open(self.trg_datapath, "r") as in_file:
            for _ in in_file:
                total += 1

        offset = 0
        if os.path.exists(self.labels_datapath):
            with open(self.labels_datapath, "r") as in_file:
                csv_reader = csv.reader(in_file)
                for row in csv_reader:
                    offset = max(offset, int(row[0]) + 1)

        logging.info(f"Already labeled: {offset}")
        return offset, total

    def _show_help(self, stdscr):
        stdscr.clear()

        center_horiz(stdscr, 1, "Input data\n\n", curses.A_BOLD)
        stdscr.addstr(f"Targets: {os.path.abspath(self.trg_datapath)}\n"
                      f"Translations: {os.path.abspath(self.translations_datapath)}\n")

        center_horiz(stdscr, 7, "Keypress legend\n\n", curses.A_BOLD)
        stdscr.addstr("a - hallucination\n"
                      "d - good translation\n"
                      "s - skip\n\n"
                      "any other key - exit")

        center_horiz(stdscr, 15, "Press space to start or any other key to exit")

        stdscr.refresh()

        label = stdscr.getkey()
        return label != " "

    @staticmethod
    def _count_ngrams(tokens, ngram_order):
        max_count = 0
        cnt = defaultdict(int)
        for idx in range(len(tokens) - ngram_order):
            mark = "_".join(map(str, tokens[idx: idx + ngram_order]))
            cnt[mark] += 1
            if cnt[mark] > max_count:
                max_count = cnt[mark]
        return max_count

    def _run_labeling(self, stdscr, offset, total):
        writer = csv.writer(open(self.labels_datapath, "a+"))

        with open(self.trg_datapath, "r") as trg_file, \
                open(self.translations_datapath, "r") as tr_file:

            for _ in range(offset):
                trg_file.readline()
                tr_file.readline()

            bleu_metric = sacrebleu.BLEU(effective_order=True)
            stdscr.clear()
            for trg_sent, tr_sent in zip(trg_file, tr_file):
                stdscr.erase()

                stdscr.addstr(f"[{offset:7d}/{total}]\n\n")

                rows, cols = stdscr.getmaxyx()
                stdscr.addstr(f"Target:\n{wrap_text(trg_sent, cols)}\n\n")
                stdscr.addstr(f"Translated:\n{wrap_text(tr_sent, cols)}\n\n")

                bleu_score = bleu_metric.sentence_score(tr_sent, [trg_sent])
                stdscr.addstr(f"BLEU: {bleu_score.score:.6f}\n\n")

                if bleu_score.score > 60.:
                    stdscr.addstr(f"High BLEU - possibly good?\n\n")

                tokens = self.tokenizer.encode_trg(tr_sent)
                is_hal = False
                for ngram_order, count_thresh in [(1, 8), (2, 5), (3, 3)]:
                    max_ngram_count = self._count_ngrams(tokens, ngram_order)
                    if max_ngram_count >= count_thresh:
                        is_hal = True

                if is_hal:
                    stdscr.addstr(f"Many repeating ngrams - possibly hallucination\n\n")

                stdscr.refresh()
                keypress = stdscr.getkey()
                if keypress == "a":
                    label = "hallucination"
                elif keypress == "d":
                    label = "good"
                elif keypress == "s":
                    label = None
                else:
                    break

                if label is not None:
                    writer.writerow([offset, label])
                offset += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Manual labeling for hallucinations"
    )
    parser.add_argument("trg_datapath", type=str)
    parser.add_argument("translations_datapath", type=str)
    parser.add_argument("labels_datapath", type=str)
    parser.add_argument("config_path", type=str)

    script_args = parser.parse_args()

    labeling_manager = LabelingManager(script_args)

    curses.wrapper(labeling_manager)
