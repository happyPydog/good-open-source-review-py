import numpy as np

from mlfunctools.metrics import classifier_metrics


def test_classifier_metrics():
    y_true = [1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 1, 1, 0, 0]
    y_pred_proba = [0.9, 0.1, 0.8, 0.7, 0.2, 0.6]
    metrics = classifier_metrics(y_true, y_pred, y_pred_proba)

    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
