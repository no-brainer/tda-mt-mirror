import csv

from easynmt import EasyNMT
from sacrebleu import sentence_bleu
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

from utils.data import tsv_sentence_pairs


OUTPUT_PATH = "./data/translation.txt"
DATASET_PATH = "./data/train.txt"

TGT_LANG = "ru"
SRC_LANG = "en"

MODEL_TYPE = "opus-mt"
MODEL_NAME = f"Helsinki-NLP/opus-mt-{SRC_LANG}-{TGT_LANG}"

BEAM_SIZE = 1


model = EasyNMT(MODEL_TYPE)
translator = model.translator.load_model(MODEL_NAME)

labse = SentenceTransformer("sentence-transformers/LaBSE")

with open(OUTPUT_PATH, "w") as output_file:
    tsv_writer = csv.writer(output_file, delimeter="excel-tab")
    tsv_writer.writerow(["translation", "cosine", "bleu"])
    for tgt_sentence, src_sentence in tsv_sentence_pairs(DATASET_PATH, TGT_LANG, SRC_LANG):
        translation = model.translate(
            [src_sentence], 
            source_lang=SRC_LANG,
            target_lang=TGT_LANG,
            beam_size=BEAM_SIZE,
        )

        sentence_embs = labse.encode([tgt_sentence, translation])
        similarity = cosine(*sentence_embs)

        bleu = sentence_score(translation, [tgt_sentence]).score

        tsv_writer.writerow([translation, similarity, bleu])