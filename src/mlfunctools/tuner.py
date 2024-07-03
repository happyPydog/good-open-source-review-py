"""Hyperparameter tuning module."""

from typing import Self

import mlflow
import optuna
import xgboost as xgb

from mlfunctools.metrics import classifier_metrics


class Tuner:
    """Abstract class for hyperparameter tuning."""

    def __init__(self) -> None:
        self.study = optuna.create_study(direction="maximize")

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to optimize."""
        raise NotImplementedError

    def tune(self, n_trials: int = 10, timeout: int = 60) -> Self:
        """Tunes parameters of the XGBoost model."""
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout)
        return self

    @property
    def best_params(self) -> dict:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value


class XGBoostTuner(Tuner):
    """XGBoost hyperparameter tuner."""

    def __init__(self, X, y, X_eval, y_eval) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.X_eval = X_eval
        self.y_eval = y_eval

        self.eval_set = [(X, y), (X_eval, y_eval)]

    def objective(self, trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            params = {
                "max_depth": trial.suggest_int("max_depth", 2, 10),
                "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1e-1
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "gamma": trial.suggest_float("gamma", 0.0, 1.0),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 10
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
            }

            model = xgb.XGBClassifier(**params)

            model.fit(self.X, self.y, eval_set=self.eval_set, verbose=False)

            # prediction
            y_pred_proba = model.predict_proba(self.X_eval)[:, 1]
            y_pred = model.predict(self.X_eval)

            # compute metrics
            metrics = classifier_metrics(self.y_eval, y_pred, y_pred_proba)

            mlflow.log_params(params)

        return metrics["pr_auc"]
