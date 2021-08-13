import argparse
import csv

from easynmt import EasyNMT
from sacrebleu import sentence_bleu
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer
import torch

from utils.data import tsv_sentence_pairs


parser = argparse.ArgumentParser(
    description="Compute translation quality for individual sentence pairs"
)
parser.add_argument("input_path", type=str)
parser.add_argument("output_path", type=str)
args = parser.parse_args()

OUTPUT_PATH = args.output_path
DATASET_PATH = args.input_path

TGT_LANG = "rus"
SRC_LANG = "eng"

MODEL_TYPE = "opus-mt"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG[:2]}-{TGT_LANG[:2]}"

BEAM_SIZE = 1
BATCH_SIZE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = EasyNMT(MODEL_TYPE)
translator = model.translator.load_model(MODEL_NAME)

labse = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)

with open(OUTPUT_PATH, "w") as output_file:
    tsv_writer = csv.writer(output_file, dialect="excel-tab")
    tsv_writer.writerow(["line_idx", "translation", "bleu", "cosine_src_gold", "cosine_src_tr", "cosine_gold_tr"])
    for line_idx, tgt_sentence_batch, src_sentence_batch in tsv_sentence_pairs(DATASET_PATH, TGT_LANG, SRC_LANG, BATCH_SIZE):
        translations = model.translate(
            src_sentence_batch, 
            source_lang=SRC_LANG[:2],
            target_lang=TGT_LANG[:2],
            beam_size=BEAM_SIZE,
        )

        src_embs = labse.encode(src_sentence_batch)
        tgt_gold_embs = labse.encode(tgt_sentence_batch)
        tgt_translated_embs = labse.encode(translations)

        cosine_dist_src_gold = paired_cosine_distances(src_embs, tgt_gold_embs)
        cosine_dist_src_translated = paired_cosine_distances(src_embs, tgt_translated_embs)
        cosine_dist_gold_translated = paired_cosine_distances(tgt_gold_embs, tgt_translated_embs)

        for i in range(BATCH_SIZE):
            bleu = sentence_bleu(translations[i], [tgt_sentence_batch[i]]).score / 100
            
            tsv_writer.writerow([
                line_idx[i], 
                translations[i], 
                bleu,
                cosine_dist_src_gold[i],
                cosine_dist_src_translated[i],
                cosine_dist_gold_translated[i],
            ])