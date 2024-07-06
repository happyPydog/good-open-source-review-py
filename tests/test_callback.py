import os

import mlflow
import numpy as np
import pytest
import xgboost as xgb

from mlfunctools.callback import (
    ClassifierMetricsCallback,
    LogLossCallback,
    MlflowCallbackMinxin,
    OptunaCheckPointCallback,
)
from mlfunctools.tuner import XGBoostTuner


@pytest.fixture
def mlflow_client():
    return mlflow.tracking.MlflowClient()


@pytest.fixture
def data():
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    return X, y


def train(X, y, callback):
    model = xgb.XGBClassifier(callbacks=[callback])
    model.fit(
        X,
        y,
        eval_set=[(X, y), (X, y)],
        verbose=False,
    )


def test_MlflowCallbackMinxin():
    assert MlflowCallbackMinxin._client is not None
    assert MlflowCallbackMinxin().client is not None


class MLflowCallbackTestMinxin:

    def setup_method(self, method):
        self.run = mlflow.start_run()

    def teardown_method(self, method):
        mlflow.end_run()

    def should_exist_artifacts(self, file_path):
        artifacts_path = self.get_artifacts_path()
        assert file_path in os.listdir(artifacts_path)

    def get_artifacts_path(self):
        pre_path = "mlruns/0"
        return f"{pre_path}/{self.run.info.run_id}/artifacts"


class TestLogLossCallback(MLflowCallbackTestMinxin):

    def test_LogLossCallback(self, data):
        X, y = data

        callback = LogLossCallback(X, y, self.run)

        train(X, y, callback)

        # Check if the attributes are set
        assert callback.X is not None
        assert callback.y is not None
        assert callback.run is not None
        assert callback.history is not None
        assert "train_logloss" in callback.history
        assert "eval_logloss" in callback.history

        # Check if training loss figure is saved
        self.should_exist_artifacts("training_loss.png")


class TestClassifierMetricsCallback(MLflowCallbackTestMinxin):

    def test_ClassifierMetricsCallback(self, mlflow_client, data):
        X, y = data

        callback = ClassifierMetricsCallback(X, y, self.run)

        train(X, y, callback)

        # Check if the attributes are set
        assert callback.X is not None
        assert callback.y is not None
        assert callback.run is not None
        assert callback.threshold is not None
        assert callback.history is not None

        # Check metrics are logged
        assert "pr_auc" in callback.history
        assert "roc_auc" in callback.history
        assert "f1" in callback.history
        assert "accuracy" in callback.history
        assert "precision" in callback.history
        assert "recall" in callback.history

        # Check metrics are logged in mlflow
        assert (
            mlflow_client.get_run(self.run.info.run_id).data.metrics.keys()
            == callback.history.keys()
        )

        # Check if figure is saved
        self.should_exist_artifacts("pr_auc_curve.png")
        self.should_exist_artifacts("roc_auc_curve.png")
        self.should_exist_artifacts("confusion_matrix.png")


class TestOptunaCheckPointCallback(MLflowCallbackTestMinxin):

    def test_OptunaCheckPointCallback(self, data):
        X, y = data

        n_trials = 10
        recorded_step = 1
        callback = OptunaCheckPointCallback(
            n_trials=n_trials, recorded_step=recorded_step
        )

        tuner = XGBoostTuner(
            X=X,
            y=y,
            X_eval=X,
            y_eval=y,
        )

        tuner.tune(n_trials=n_trials, callbacks=[callback])

        # Check if the attributes are set
        assert callback.n_trials == n_trials
        assert callback.recorded_step == recorded_step

        # # Check if study checkpoint is saved
        self.should_exist_artifacts("optuna")
