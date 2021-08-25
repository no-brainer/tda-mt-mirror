import os
import pickle

import optuna

class Objective:
    def __init__(self, model_cls, eval_func, search_space, base_params=None, checkpoint_folder=None, exp_name=None):
        self.model_cls = model_cls
        self.search_space = search_space
        self.base_params = dict() if base_params is None else base_params
        self.eval_func = eval_func
        self.checkpoint_folder = checkpoint_folder
        self.exp_name = "exp" if exp_name is None else exp_name

    def __call__(self, trial):
        params = dict(self.base_params)
        for name, param_type, param_bounds in self.search_space:
            suggest_func = getattr(trial, "suggest_" + param_type)
            params[name] = suggest_func(name, *param_bounds)
        
        score, model = self.eval_func(self.model_cls(**params))
        if self.checkpoint_folder is not None and model is not None:
            checkpoint_path = os.path.join(self.checkpoint_folder, f"{self.exp_name}_trial{trial.number}")
            pickle.dump(model, open(checkpoint_path, "wb"))

        return score
