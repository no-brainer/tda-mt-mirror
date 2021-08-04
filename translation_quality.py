import csv

from easynmt import EasyNMT
from sacrebleu import sentence_bleu
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from utils.data import tsv_sentence_pairs


OUTPUT_PATH = "./data/translation.txt"
DATASET_PATH = "./data/train.txt"

TGT_LANG = "rus"
SRC_LANG = "eng"

MODEL_TYPE = "opus-mt"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG[:2]}-{TGT_LANG[:2]}"

BEAM_SIZE = 1


model = EasyNMT(MODEL_TYPE)
translator = model.translator.load_model(MODEL_NAME)

labse = SentenceTransformer("sentence-transformers/LaBSE")

with open(OUTPUT_PATH, "w") as output_file:
    tsv_writer = csv.writer(output_file, dialect="excel-tab")
    tsv_writer.writerow(["translation", "cosine", "bleu"])
    for tgt_sentence, src_sentence in tsv_sentence_pairs(DATASET_PATH, TGT_LANG, SRC_LANG):
        translation = model.translate(
            [src_sentence], 
            source_lang=SRC_LANG[:2],
            target_lang=TGT_LANG[:2],
            beam_size=BEAM_SIZE,
        )

        sentence_embs = labse.encode([tgt_sentence, translation])
        similarity = cosine(*sentence_embs)

        bleu = sentence_score(translation, [tgt_sentence]).score

        tsv_writer.writerow([translation, similarity, bleu])