# import mlflow
# import xgboost as xgb

# from mlfunctools.metrics import common_metrics


# class MlflowCommonMetricsCallback(xgb.callback.TrainingCallback):
#     """Compute common metrics after each iteration."""

#     def __init__(self, X, y) -> None:
#         self.X = X
#         self.y = y
#         self.metrics = {
#             "accuracy": [],
#             "precision": [],
#             "recall": [],
#             "f1": [],
#             "roc_auc": [],
#             "pr_auc": [],
#             "log_loss": [],
#         }

#     def after_iteration(
#         self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
#     ) -> bool:
#         y_pred = model.predict(self.X)
#         y_pred_proba = model.predict_proba(self.X)[:, 1]
#         metrics = common_metrics(self.y, y_pred, y_pred_proba)

#         for key, value in metrics.items():
#             self.metrics[key].append(value)

#         return False

#     def get_metrics(self):
#         return self.metrics
