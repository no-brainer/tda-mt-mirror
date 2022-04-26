from collections import Counter

from rouge_score import rouge_scorer
import sacrebleu
from scipy.spatial import distance
from sentence_transformers import SentenceTransformer


__all__ = [
    "LabseCosineDistanceFunctor",
    "RougeMetricFunctor",
    "AdjustedBleuFunctor",
    "compute_bleu",
]


class LabseCosineDistanceFunctor:

    def __init__(self, device):
        self.labse_model = SentenceTransformer("sentence-transformers/LaBSE", device=device)

    def __call__(self, data, field1, field2):
        sent1_emb = self.labse_model.encode(data[field1], convert_to_numpy=True)
        sent2_emb = self.labse_model.encode(data[field2], convert_to_numpy=True)
        # distance.cosine returns cosine distance which is equal to 1 - cosine similarity
        return 1 - distance.cosine(sent1_emb, sent2_emb)


class RougeMetricFunctor:

    def __init__(self, rouge_type):
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)

    def __call__(self, data):
        scores = self.scorer.score(data["tr"], data["ref"])
        return list(scores.values())[0]


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


class AdjustedBleuFunctor:
    """
    Proposed for hallucination filtering in https://clarafy.github.io/research/neurips_irasl_2018.pdf
    """
    def __init__(self, smooth_method="exp", smooth_value=None):
        args = sacrebleu.Namespace(
            smooth_method=smooth_method, smooth_value=smooth_value, force=False,
            short=False, lc=False, tokenize=sacrebleu.DEFAULT_TOKENIZER
        )

        self.metric = AdjustedBLEU(args)

    def __call__(self, data):
        return self.metric.sentence_score(data["tr"], [data["ref"]]).score / 100


def compute_bleu(data):
    return sacrebleu.sentence_bleu(data["tr"], [data["ref"]]).score / 100
