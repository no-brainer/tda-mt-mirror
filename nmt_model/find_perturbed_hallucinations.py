"""
Follows Raunak's extension of hallucination detection by Lee
"""
import argparse
from collections import Counter

import sacrebleu
import torch

from src.models import NMTTransformer
import src.tokenizers
import src.translators
from src.utils import parse_config, set_seed, init_obj


class AdjustedBLEU(sacrebleu.BLEU):
    NGRAM_ORDER = 2
    WEIGHTS = [1., 0.8]

    @staticmethod
    def extract_ngrams(line, min_order=1, max_order=NGRAM_ORDER) -> Counter:
        ngrams = Counter()
        tokens = line.split()
        for n, weight in enumerate(AdjustedBLEU.WEIGHTS, min_order):
            for i in range(0, len(tokens) - n + 1):
                ngram = ' '.join(tokens[i: i + n])
                ngrams[ngram] += weight

        return ngrams


def check_with_perturbations(perturbation_tokens, translator, src_sent, trg_sent, hallucination_thresh, batch_size=16):
    bleu_metric = AdjustedBLEU(effective_order=True)

    src_tokens = translator.tokenizer.encode_src(src_sent)
    src_tokens = src_tokens[1:]  # remove bos token

    new_sents = []
    for token in perturbation_tokens:
        pert_src_sent = translator.tokenizer.decode_src([translator.bos_id, token] + src_tokens)
        new_sents.append(pert_src_sent)

    for start_idx in range(0, len(new_sents), batch_size):
        batch = new_sents[start_idx:start_idx + batch_size]
        translations = translator.translate_batch(batch)

        for translation in translations:
            adj_bleu = bleu_metric.sentence_score(translation, [trg_sent])
            adj_bleu = adj_bleu.score
            if adj_bleu < hallucination_thresh:
                return True, translation

    return False, None


def detect_hallucinations(perturbation_tokens, translator, src_datapath, trg_datapath, out_datapath,
                          perturbation_thresh, hallucination_thresh):
    bleu_metric = AdjustedBLEU(effective_order=True)
    with open(src_datapath, "r") as src_file, \
            open(trg_datapath, "r") as trg_file, \
            open(out_datapath, "w") as out_file:

        for src_sent, trg_sent in zip(src_file, trg_file):
            translation = translator.translate(src_sent)

            adj_bleu = bleu_metric.sentence_score(translation, [trg_sent])
            adj_bleu = adj_bleu.score
            is_hallucination = False
            perturbed_hallucination = None
            if adj_bleu > perturbation_thresh:
                is_hallucination, perturbed_hallucination = check_with_perturbations(
                    perturbation_tokens,
                    translator,
                    src_sent,
                    trg_sent,
                    hallucination_thresh
                )

            if perturbed_hallucination is None:
                perturbed_hallucination = translation

            label = "1" if is_hallucination else "0"
            out_file.write(f"{perturbed_hallucination},{label}\n")


def select_perturbation_tokens(tokenizer, src_datapath, n_tokens):
    cnt = Counter()
    with open(src_datapath, "r") as in_file:

        for line in in_file:
            token_ids = tokenizer.encode_src(line.strip())
            cnt.update(token_ids[1: len(token_ids) - 1])

    most_common_tokens = list(map(lambda x: x[0], cnt.most_common(n_tokens)))
    return most_common_tokens


def main(args):
    training_config = parse_config(args.config_path)

    device = "cpu"
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda"
    print("Inference on", device)

    set_seed(training_config["seed"])

    saved_data = torch.load(args.checkpoint_path, map_location="cpu")
    model = NMTTransformer(**training_config["model"])
    model.load_state_dict(saved_data["state_dict"])
    model = model.to(device)

    tokenizer = init_obj(src.tokenizers, training_config["tokenizer"])
    translator = src.translators.GreedyTranslator(model, tokenizer, device, args.bos_id, args.eos_id, args.pad_id,
                                                  args.max_length)

    model.eval()

    perturbation_tokens = select_perturbation_tokens(tokenizer, args.src_datapath, args.perturbation_tokens)
    print("Most common tokens:")
    decoded_pert_tokens = tokenizer.decode_src([[idx] for idx in perturbation_tokens])
    print(*decoded_pert_tokens)

    detect_hallucinations(perturbation_tokens, translator, args.src_datapath, args.trg_datapath, args.out_datapath,
                          args.perturbation_thresh, args.hallucination_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("src_datapath", type=str)
    parser.add_argument("trg_datapath", type=str)
    parser.add_argument("out_datapath", type=str)

    parser.add_argument("--config_path", "-c", type=str, required=True)
    parser.add_argument("--not_use_cuda", dest="use_cuda", action="store_false")

    parser.add_argument("--perturbation_tokens", "-p", type=int, default=100)

    parser.add_argument("--max_length", "-l", type=int, default=512)
    parser.add_argument("--bos_id", type=int, default=2)
    parser.add_argument("--eos_id", type=int, default=3)
    parser.add_argument("--pad_id", type=int, default=0)

    parser.add_argument("--perturbation_thresh", type=float, default=9.)
    parser.add_argument("--hallucination_thresh", type=float, default=1.)

    script_args = parser.parse_args()

    main(script_args)
