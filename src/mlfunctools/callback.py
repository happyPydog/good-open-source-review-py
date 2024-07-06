import collections
import pickle
import tempfile

import matplotlib.pyplot as plt
import mlflow
import mlflow.entities
import optuna
import plotly.graph_objects as go
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from mlfunctools.metrics import classifier_metrics
from mlfunctools.types import ArrayLike


class MlflowCallbackMinxin:

    _client = mlflow.tracking.MlflowClient()

    @property
    def client(self) -> mlflow.tracking.MlflowClient:
        return self._client


class LogLossCallback(xgb.callback.TrainingCallback, MlflowCallbackMinxin):

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        run: mlflow.entities.Run,
    ) -> None:
        super().__init__()
        self.X = xgb.DMatrix(X)
        self.y = y
        self.run = run
        self.history: dict[str, list[float]] = collections.defaultdict(list)

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ) -> bool:
        metrics = {}
        # ! assume the order of the eval_set = [(X_train, y_train), (X_eval, y_eval)]
        if "validation_0" in evals_log:
            metrics["train_logloss"] = evals_log["validation_0"]["logloss"][-1]
        if "validation_1" in evals_log:
            metrics["eval_logloss"] = evals_log["validation_1"]["logloss"][-1]

        for key, value in metrics.items():
            self.history[key].append(value)

        mlflow.log_metrics(metrics=metrics, step=epoch)
        return False

    def after_training(self, model: xgb.Booster):
        self._plot_training_loss()
        return model

    def _plot_training_loss(self):
        if not self.history:
            return

        fig = go.Figure()

        if "train_logloss" in self.history:

            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.history["train_logloss"]))),
                    y=self.history["train_logloss"],
                    mode="lines+markers",
                    name="training_loss",
                )
            )

        if "eval_logloss" in self.history:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(self.history["eval_logloss"]))),
                    y=self.history["eval_logloss"],
                    mode="lines+markers",
                    name="eval_loss",
                )
            )

        fig.update_layout(
            title="Training and Evaluation Log Loss Over Epochs",
            xaxis_title="Epoch",
            yaxis_title="Log Loss",
            legend=dict(x=0.8, y=1.2),
        )

        fig_name = "training_loss.png"
        self.client.log_figure(self.run.info.run_id, fig, fig_name)


class ClassifierMetricsCallback(
    xgb.callback.TrainingCallback, MlflowCallbackMinxin
):
    """Log common classifier metrics after each iteration."""

    def __init__(
        self,
        X: ArrayLike,
        y: ArrayLike,
        run: mlflow.entities.Run,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.X = xgb.DMatrix(X)
        self.y = y
        self.run = run
        self.threshold = threshold
        self.history: dict[str, list[float]] = collections.defaultdict(list)

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ) -> bool:
        y_pred_proba = model.predict(self.X)
        y_pred = self._predict(y_pred_proba)

        metrics = classifier_metrics(self.y, y_pred, y_pred_proba)

        mlflow.log_metrics(metrics=metrics, step=epoch)

        for key, value in metrics.items():
            self.history[key].append(value)

        return False

    def after_training(self, model: xgb.Booster):
        y_pred_proba = model.predict(self.X)
        y_pred = self._predict(y_pred_proba)

        self._plot_pr_auc_curve(y_pred_proba)
        self._plot_roc_auc_curve(y_pred_proba)
        self._plot_confusion_matrix(y_pred)

        return model

    def _predict(self, y_pred_proba):
        return (y_pred_proba > self.threshold).astype(int)

    def _plot_pr_auc_curve(self, y_pred_proba):
        pr_display = PrecisionRecallDisplay.from_predictions(
            self.y, y_pred_proba
        )
        fig, ax = plt.subplots()
        pr_display.plot(ax=ax)
        fig_name = "pr_auc_curve.png"
        self.client.log_figure(self.run.info.run_id, fig, fig_name)

    def _plot_roc_auc_curve(self, y_pred_proba):
        roc_display = RocCurveDisplay.from_predictions(self.y, y_pred_proba)
        fig, ax = plt.subplots()
        roc_display.plot(ax=ax)
        fig_name = "roc_auc_curve.png"
        self.client.log_figure(self.run.info.run_id, fig, fig_name)

    def _plot_confusion_matrix(self, y_pred):
        cm_display = ConfusionMatrixDisplay.from_predictions(self.y, y_pred)
        fig, ax = plt.subplots()
        cm_display.plot(ax=ax, cmap=plt.cm.Blues)
        fig_name = "confusion_matrix.png"
        self.client.log_figure(self.run.info.run_id, fig, fig_name)


class OptunaCheckPointCallback:

    def __init__(self, n_trials: int, recorded_step: int = 10) -> None:
        self.n_trials = n_trials
        self.recorded_step = recorded_step

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if (
            trial.number % self.recorded_step == 0
            or trial.number == self.n_trials - 1
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                study.trials_dataframe().to_csv(
                    f"{tmpdir}/trials.csv", index=False
                )
                with open(f"{tmpdir}/sampler.pkl", "wb") as fout:
                    pickle.dump(study.sampler, fout)

                mlflow.log_artifacts(tmpdir, artifact_path="optuna")
