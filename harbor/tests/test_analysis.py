import pytest
from harbor.schema.data import Dataset
from harbor.analysis import get_roc_curve, get_roc_curve_with_uncertainty
import numpy as np


def test_roc_curve(active_inactive_dataset):
    roc_curve = get_roc_curve(active_inactive_dataset, "test")
    assert len(roc_curve.fpr) == len(roc_curve.tpr)
    assert np.isclose(roc_curve.auc, 0.083, atol=0.01)
    assert roc_curve.model_id == "test"
    assert roc_curve.thresholds == [np.inf, 12, 11, 9]


def test_roc_curve_with_uncertainty(active_inactive_dataset):
    roc_curve = get_roc_curve_with_uncertainty(active_inactive_dataset, "test")
    assert len(roc_curve.fpr) == len(roc_curve.tpr)
    assert np.isclose(roc_curve.auc, 0.083, atol=0.01)
    assert roc_curve.model_id == "test"
    assert roc_curve.thresholds == [np.inf, 12, 11, 9]
    assert np.allclose(roc_curve.auc_ci, [0.0, 0.33], atol=0.01)
