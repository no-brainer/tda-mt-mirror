import argparse
import logging
import os

from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from dagshub import dagshub_logger
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from common.random_search import Objective


# constants
SEED = 42
EXP_NAME = "catboost for bleu prediction"
BASE_PARAMS = {
    "random_state": SEED, 
    "task_type": "GPU" if get_gpu_device_count() != 0 else "CPU",
    "devices": "0",
}
SEARCH_SPACE = [
    ("iterations", "discrete_uniform", (50, 1500, 10)),
    ("learning_rate", "loguniform", (1e-4, 1)),
    ("depth", "int", (2, 6)),
]
KFOLD_PARAMS = {
    "n_splits": 5,
    "random_state": SEED,
    "shuffle": True,
}

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("features_path", type=str)
parser.add_argument("target_path", type=str)
parser.add_argument("n_trials", type=int)
parser.add_argument("--metrics_path", default="../metrics.csv", type=str)
parser.add_argument("--hparam_path", default="../params.yml", type=str)
parser.add_argument("--log_path", default="./bleu_prediction.log", type=str)
parser.add_argument("--checkpoint_path", default=None, type=str)
args = parser.parse_args()

# logger
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename=args.log_path, 
    level=logging.INFO,
)


def prep_features(features_path):
    X = pd.read_csv(features_path, sep="\t", index_col=0)
    return X

def prep_target(target_path):
    df = pd.read_csv(target_path, sep="\t", index_col=0)
    return df.bleu

def get_eval_func(X, y):
    X = X.values
    y = y.values
    kf = KFold(**KFOLD_PARAMS)

    def func(model):
        logging.info("Starting evaluation")
        r2s = []
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            train_data = Pool(data=X[train_index], label=y[train_index])
            test_data = Pool(data=X[test_index], label=y[test_index])
            model.fit(train_data)
            preds = model.predict(test_data)
            score = r2_score(y[test_index], preds)
            logging.info(f"Fold {i}: r2_score is {score:.6f}")
            r2s.append(score)
        r2s = np.array(r2s)
        return r2s.mean(), None

    return func


if __name__ == "__main__":
    logging.info(f"Loading features from {os.path.abspath(args.features_path)}")
    X = prep_features(args.features_path)
    logging.info(f"Loading target from {os.path.abspath(args.target_path)}")
    y = prep_target(args.target_path)
    X = X.loc[y.index]

    study = optuna.create_study(direction="maximize", study_name=EXP_NAME)
    obj = Objective(CatBoostRegressor, get_eval_func(X, y), SEARCH_SPACE, BASE_PARAMS)
    study.optimize(obj, n_trials=args.n_trials)
    hparams = study.best_params
    metrics = dict(r2=study.best_value)

    logging.info("Logging data to dagshub")
    with dagshub_logger(metrics_path=args.metrics_path, hparams_path=args.hparam_path) as logger:
        logger.log_metrics(metrics)
        logger.log_hyperparams(model=hparams)
        logger.log_hyperparams(random_seed=SEED)
        logger.log_hyperparams(experiment_type=bleu_prediction)
        logger.log_hyperparams(features_path=args.features_path)
        logger.log_hyperparams(target_path=args.target_path)
    
    logging.info("Finished")
