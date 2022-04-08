"""
Follows RU scheme from https://arxiv.org/pdf/2104.06683.pdf
"""
import argparse
import linecache

import numpy as np


def count_sentence_pairs(corpus_filepath):
    cnt = 0
    with open(corpus_filepath, "r") as corpus_file:
        for _ in corpus_file:
            cnt += 1

    return cnt


def build_corpus(src_indices, trg_indices, corpus_src_filepath, corpus_trg_filepath,
                 src_output_filepath, trg_output_filepath):
    with open(src_output_filepath, "w") as src_output_file, \
            open(trg_output_filepath, "w") as trg_output_file:
        for src_idx, trg_idx in zip(src_indices, trg_indices):
            src_sent = linecache.getline(corpus_src_filepath, src_idx + 1)
            src_output_file.write(src_sent)

            trg_sent = linecache.getline(corpus_trg_filepath, trg_idx + 1)
            trg_output_file.write(trg_sent)


def main(args):
    num_sentences = count_sentence_pairs(args.corpus_src_filepath)

    rng = np.random.default_rng(args.seed)

    selected_indices = rng.choice(
        num_sentences,
        size=args.n_clean_pairs + args.n_unique_sources,
        replace=False
    )
    clean_indices = selected_indices[:args.n_clean_pairs]
    noisy_indices = selected_indices[args.n_clean_pairs:]

    noisy_targets = rng.choice(
        num_sentences,
        size=args.n_unique_sources * args.targets_per_source,
        replace=False
    )

    src_indices = np.concatenate([clean_indices, np.repeat(noisy_indices, args.targets_per_source)])
    trg_indices = np.concatenate([clean_indices, noisy_targets])

    permutation = rng.permutation(len(src_indices))
    src_indices = src_indices[permutation]
    trg_indices = trg_indices[permutation]

    build_corpus(src_indices, trg_indices, args.corpus_src_filepath, args.corpus_trg_filepath,
                 args.src_output_filepath, args.trg_output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("corpus_src_filepath", type=str)
    parser.add_argument("corpus_trg_filepath", type=str)

    parser.add_argument("src_output_filepath", type=str)
    parser.add_argument("trg_output_filepath", type=str)

    parser.add_argument("n_clean_pairs", type=int)
    parser.add_argument("n_unique_sources", type=int)
    parser.add_argument("targets_per_source", type=int)

    parser.add_argument("--seed", "-s", type=int, default=42)

    script_args = parser.parse_args()

    main(script_args)
