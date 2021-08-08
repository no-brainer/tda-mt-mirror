import argparse
import csv

from easynmt import EasyNMT
from sacrebleu import sentence_bleu
from scipy.spatial.distance import cosine
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = EasyNMT(MODEL_TYPE)
translator = model.translator.load_model(MODEL_NAME)

labse = SentenceTransformer("sentence-transformers/LaBSE", device=DEVICE)

with open(OUTPUT_PATH, "w") as output_file:
    tsv_writer = csv.writer(output_file, dialect="excel-tab")
    tsv_writer.writerow(["line_idx", "translation", "cosine", "bleu"])
    for line_idx, tgt_sentence, src_sentence in tsv_sentence_pairs(DATASET_PATH, TGT_LANG, SRC_LANG):
        translation = model.translate(
            [src_sentence], 
            source_lang=SRC_LANG[:2],
            target_lang=TGT_LANG[:2],
            beam_size=BEAM_SIZE,
        )[0]

        sentence_embs = labse.encode([tgt_sentence, translation])
        similarity = cosine(*sentence_embs)

        bleu = sentence_bleu(translation, [tgt_sentence]).score

        tsv_writer.writerow([line_idx, translation, similarity, bleu])