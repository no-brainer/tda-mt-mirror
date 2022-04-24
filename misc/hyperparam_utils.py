def restore_params(best_params, search_space):
    full_params = best_params.copy()
    for name, params in search_space.items():
        if params[0] == "const":
            full_params[name] = params[1]

    return full_params


class OptimizationObjective:

    def __init__(self, X, y, kf, model_cls, search_space, scoring, scoring_type="binary"):
        self.X = X
        self.y = y

        self.kf = kf

        self.model_cls = model_cls
        self.search_space = search_space

        assert scoring_type in ["binary", "probs"], "Invalid scoring type"
        self.scoring_type = scoring_type
        self.scoring = scoring

    def _prep_model_params(self, trial):
        model_params = dict()
        for name, params in self.search_space.items():
            suggestion_type = params[0]
            if suggestion_type == "const":
                value = params[1]
            else:
                suggest_value = getattr(trial, f"suggest_{suggestion_type}")
                value = suggest_value(name, *params[1:])

            model_params[name] = value

        return model_params

    def _get_scores(self, estimator):
        scores = []
        for train_index, test_index in self.kf.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            estimator = estimator.fit(X_train, y_train)

            if self.scoring_type == "binary":
                preds = estimator.predict(X_test)
            else:
                preds = estimator.predict_proba(X_test)

            scores.append(self.scoring(y_test, preds))

        return np.mean(scores), np.std(scores)

    def __call__(self, trial):
        model_params = self._prep_model_params(trial)
        estimator = self.model_cls(**model_params)

        scores_mean, scores_std = self._get_scores(estimator)
        return scores_mean
