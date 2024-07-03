import collections

import mlflow
import mlflow.entities
import plotly.graph_objects as go
import xgboost as xgb

from mlfunctools.metrics import classifier_metrics


class ClassifierMetricsCallback(xgb.callback.TrainingCallback):
    """Log common classifier metrics after each iteration."""

    def __init__(self, X, y, run: mlflow.entities.Run) -> None:
        self.X = xgb.DMatrix(X)
        self.y = y
        self.run = run
        self.client = mlflow.tracking.MlflowClient()
        self.history: dict[str, list[float]] = collections.defaultdict(list)

        super().__init__()

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ) -> bool:
        y_pred_proba = model.predict(self.X)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # log common classifier metrics
        metrics = classifier_metrics(self.y, y_pred, y_pred_proba)

        # log loss
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
        fig = go.Figure()

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
