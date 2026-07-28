import numpy as np

from src.evaluation import evaluate


class TestEvaluate:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])
        metrics = evaluate(y_true, y_pred, y_proba, model_name="test")
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_no_positive_predictions_handles_zero_division(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 0, 0])
        y_proba = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = evaluate(y_true, y_pred, y_proba, model_name="test")
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        y_proba = np.array([0.1, 0.9, 0.6, 0.8])
        metrics = evaluate(y_true, y_pred, y_proba, model_name="test")
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2
