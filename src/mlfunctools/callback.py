import mlflow
import xgboost as xgb

from mlfunctools.metrics import classifier_metrics


class ClassifierMetricsCallback(xgb.callback.TrainingCallback):
    """Compute common metrics after each iteration."""

    def __init__(self, X, y) -> None:
        self.X = xgb.DMatrix(X)
        self.y = y

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ) -> bool:
        y_pred_proba = model.predict(self.X)
        y_pred = (y_pred_proba > 0.5).astype(int)
        metrics = classifier_metrics(self.y, y_pred, y_pred_proba)
        mlflow.log_metrics(metrics=metrics, step=epoch)
        return False
