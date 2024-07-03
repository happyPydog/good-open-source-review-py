"""The example of tuning hyperparameters of XGBoost.
Also, log the trial in mlflow.
"""

import mlflow
import pandas as pd
import xgboost as xgb
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from mlfunctools.metrics import common_metrics
from mlfunctools.mlflow import mlflow_run
from mlfunctools.tuner import XGBoostTuner


def get_titanic_dataset():
    dataset = fetch_openml(name="titanic", version=1, as_frame=True)
    df = dataset.frame

    df = df.drop(columns=["boat", "body", "home.dest"])
    df = df.dropna()

    X = df.drop(columns=["survived"])
    y = pd.DataFrame(df["survived"].astype(int))

    for col in X.select_dtypes(include=["category", "object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    for col in X.select_dtypes(include=["int64"]).columns:
        X[col] = X[col].astype("float64")
    return X, y


class MlflowCommonMetricsCallback(xgb.callback.TrainingCallback):
    """Compute common metrics after each iteration."""

    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y
        self.metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "roc_auc": [],
            "pr_auc": [],
            "log_loss": [],
        }

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: dict[str, dict]
    ) -> bool:
        y_pred = model.predict(self.X)
        y_pred_proba = model.predict_proba(self.X)[:, 1]
        metrics = common_metrics(self.y, y_pred, y_pred_proba)
        metrics["log_loss"] = evals_log["train"]["logloss"][epoch]

        for key, value in metrics.items():
            self.metrics[key].append(value)
            mlflow.log_metrics(metrics=self.metrics, step=epoch)

        return False

    def get_metrics(self):
        return self.metrics


@mlflow_run(run_name="training")
def main():

    X, y = get_titanic_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(
        f"{X_train.shape = }, {X_test.shape = }, {y_train.shape = }, {y_test.shape = }"
    )

    tuner = XGBoostTuner(
        X=X_train,
        y=y_train,
        X_eval=X_test,
        y_eval=y_test,
    )

    tuner.tune()

    model = xgb.XGBClassifier(
        **tuner.best_params,
        callbacks=[MlflowCommonMetricsCallback(X_train, y_train)],
    )
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    metrics = common_metrics(y_test, y_pred, y_pred_proba)

    mlflow.log_params(tuner.best_params)
    mlflow.log_metrics(metrics)

    # log dataset
    input_X = mlflow.data.from_pandas(X_train)
    input_y = mlflow.data.from_pandas(y_train)
    mlflow.log_input(input_X, "X", tags={"data": "train"})
    mlflow.log_input(input_y, "y", tags={"data": "train"})

    # log model
    signature = mlflow.models.infer_signature(X_test, y_pred)
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="xgboost_titanic",
    )


if __name__ == "__main__":
    main()
