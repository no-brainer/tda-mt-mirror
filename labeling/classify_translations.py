import argparse
import csv
import curses
import linecache
import logging
import os
import textwrap

import pandas as pd


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename="./manual_labeling.log", 
    encoding="utf-8", 
    level=logging.INFO,
)


parser = argparse.ArgumentParser(
    description="Manual labeling for translations"
)
parser.add_argument("score_path", type=str)
parser.add_argument("corpus_path", type=str)
parser.add_argument("labeled_path", type=str)

parser.add_argument("--bleu_thresh", type=float, default=1.1, required=False)
parser.add_argument("--src_gold_thresh", type=float, default=0.0, required=False)
parser.add_argument("--src_tr_thresh", type=float, default=0.0, required=False)
parser.add_argument("--gold_tr_thresh", type=float, default=0.0, required=False)

args = parser.parse_args()


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
        line = linecache.getline(args.corpus_path, line_idx + 1)
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
        raise NotImplementedError()
    
    def extract_unknown(self):
        return ("", "")


def horiz_center_line(stdscr, y, line, attr=None):
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

def show_help(stdscr):
    stdscr.clear()

    horiz_center_line(stdscr, 1, "Input data\n\n", curses.A_BOLD)
    stdscr.addstr(f"Corpus: {os.path.abspath(args.corpus_path)}\n"
                  f"Scores: {os.path.abspath(args.score_path)}\n"
                  f"Labels: {os.path.abspath(args.labeled_path)}")

    horiz_center_line(stdscr, 7, "Keypress legend\n\n", curses.A_BOLD)
    stdscr.addstr("a - hallucination\n"
                  "d - loss of context\n"
                  "w - good translation\n"
                  "s - skip\n\n"
                  "any other key - exit")

    horiz_center_line(stdscr, 15, "Press space to start or any other key to exit")

    stdscr.refresh()

    label = stdscr.getkey()
    return label != " "

def load_data():
    if os.path.exists(args.labeled_path):
        labeled_idx = pd.read_csv(args.labeled_path, index_col=0).index
    else:
        labeled_idx = pd.Index([])

    logging.info(f"Already labeled: {len(labeled_idx)}")
    scores_df = pd.read_csv(args.score_path, index_col=0, sep="\t")
    logging.info(f"Dataset contains {len(scores_df)} sentences")
    scores_df = scores_df[scores_df["bleu"] <= args.bleu_thresh]
    logging.info(f"Sentences after bleu filter: {len(scores_df)}")
    for filter_name in ["src_gold", "src_tr", "gold_tr"]:
        thresh = getattr(args, filter_name + "_thresh")
        col_name = "cosine_" + filter_name
        scores_df = scores_df[scores_df[col_name] >= thresh]
        logging.info(f"Sentences after {filter_name} filter: {len(scores_df)}")
    scores_df.drop(labeled_idx, inplace=True, errors="ignore")

    return len(labeled_idx), scores_df

def labeling_loop(stdscr, labeled_cnt, unlabeled_df):
    pair_dispatcher = SentencePairDispatcher(args.corpus_path)
    total = labeled_cnt + len(unlabeled_df)

    with open(args.labeled_path, "a+") as label_csv:
        writer = csv.writer(label_csv)
        if os.fstat(label_csv.fileno()).st_size == 0:
            writer.writerow(["line_idx", "label"])
    
        stdscr.clear()
        for row in unlabeled_df.itertuples():
            tr = row.translation
            
            src, gold = pair_dispatcher(row.Index)

            stdscr.erase()

            stdscr.addstr(f"[{labeled_cnt:7d}/{total}]\n\n")

            rows, cols = stdscr.getmaxyx()
            stdscr.addstr(f"Source:\n{wrap_text(src, cols)}\n\n")
            stdscr.addstr(f"Gold:\n{wrap_text(gold, cols)}\n\n")
            stdscr.addstr(f"Translated:\n{wrap_text(tr, cols)}")

            stdscr.refresh()
            keypress = stdscr.getkey()
            if keypress == "a":
                label = "hallucination"
            elif keypress == "d":
                label = "context"
            elif keypress == "w":
                label = "good"
            elif keypress == "s":
                continue
            else:
                break
            
            labeled_cnt += 1
            writer.writerow([row.Index, label])
        
def main(stdscr):
    curses.curs_set(0)
    is_exiting = show_help(stdscr)
    if is_exiting:
        return
    labeled_cnt, unlabeled_df = load_data()
    labeling_loop(stdscr, labeled_cnt, unlabeled_df)

curses.wrapper(main)