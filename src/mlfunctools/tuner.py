"""Hyperparameter tuning module."""

from collections import Counter
from typing import Any, Self

import mlflow
import optuna
import plotly.graph_objects as go
import xgboost as xgb
from optuna.visualization import (  # plot_intermediate_values, # ! Need to setup pruning feature, so comment out
    plot_contour,
    plot_edf,
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_rank,
    plot_slice,
    plot_timeline,
)

from mlfunctools.callback import LogLossCallback
from mlfunctools.metrics import classifier_metrics
from mlfunctools.types import ArrayLike, OptunaCallbackFuncType

vis_funcs: dict[str, Any] = {
    "plot_contour": plot_contour,
    "plot_edf": plot_edf,
    # "plot_intermediate_values": plot_intermediate_values, # ! Need to setup pruning feature, so comment out
    "plot_optimization_history": plot_optimization_history,
    "plot_parallel_coordinate": plot_parallel_coordinate,
    "plot_param_importances": plot_param_importances,
    "plot_rank": plot_rank,
    "plot_slice": plot_slice,
    "plot_timeline": plot_timeline,
}

_has_params_vis_funcs = {
    "plot_contour",
    "plot_parallel_coordinate",
    "plot_param_importances",
    "plot_slice",
    "plot_rank",
}


class Tuner:
    """Abstract class for hyperparameter tuning."""

    def __init__(self, seed: int, direction: str = "maximize") -> None:
        self.seed = seed
        self.study = optuna.create_study(
            direction=direction, sampler=optuna.samplers.TPESampler(seed=seed)
        )

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function to optimize."""
        raise NotImplementedError

    def tune(
        self,
        n_trials: int = 10,
        timeout: int = 60,
        callbacks: list[OptunaCallbackFuncType] | None = None,
    ) -> Self:
        """Tunes parameters of the XGBoost model."""
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            callbacks=callbacks,
        )
        return self

    @property
    def best_trial(self) -> optuna.trial.FrozenTrial:
        return self.study.best_trial

    @property
    def best_params(self) -> dict:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value


class XGBoostTuner(Tuner):
    """XGBoost hyperparameter tuner."""

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        X_eval: ArrayLike,
        y_eval: ArrayLike,
        seed: int = 42,
    ) -> None:
        super().__init__(seed=seed)
        self.X = X
        self.y = y
        self.X_eval = X_eval
        self.y_eval = y_eval

        self.eval_set = [(X, y), (X_eval, y_eval)]

    def objective(self, trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            run = mlflow.active_run()
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

            model = xgb.XGBClassifier(
                **params,
                callbacks=[LogLossCallback(self.X_eval, self.y_eval, run=run)],
            )

            model.fit(self.X, self.y, eval_set=self.eval_set, verbose=False)

            # prediction
            y_pred_proba = model.predict_proba(self.X_eval)[:, 1]
            y_pred = model.predict(self.X_eval)

            # compute metrics
            metrics = classifier_metrics(self.y_eval, y_pred, y_pred_proba)

            mlflow.log_params(params)

        return metrics["pr_auc"]

    @property
    def prevalence_pos_label(self) -> float:
        counter: Counter = Counter(self.y)
        return counter[1] / len(self.y)


def analysis_param(
    study: optuna.study.Study,
    vis_funcs: dict[str, Any] = vis_funcs,
    params: list[str] | None = None,
    display: bool = True,
) -> list[go.Figure]:
    """Visualize the parameter analysis."""
    figs = []
    for key, func in vis_funcs.items():
        if key in _has_params_vis_funcs and params is not None:
            fig = func(study, params=params)
        else:
            fig = func(study)

        if display:
            fig.show()

        figs.append(fig)

    return figs
