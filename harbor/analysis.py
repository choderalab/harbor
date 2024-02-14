from harbor.data import ActiveInactiveDataset
from harbor.plot_schema import RocCurve, PrecisionRecallCurve, RocCurveUncertainty
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from pydantic import BaseModel, Field
import numpy as np


def get_ci_from_bootstraps(
    values: list[float], alpha: float = 0.95
) -> tuple[float, float]:
    """
    Calculate the confidence interval of a list of values
    """
    sorted_values = np.sort(values)
    n_values = len(sorted_values)
    lower_index = sorted_values[int(n_values * (1 - alpha))]
    upper_index = sorted_values[int(n_values * alpha)]
    return (lower_index, upper_index)


def get_roc_curve(dataset: ActiveInactiveDataset, model_id: str) -> RocCurve:
    fpr, tpr, thresholds = roc_curve(
        dataset.experimental_values, dataset.predicted_values
    )
    return RocCurve(
        id=model_id,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
        auc=roc_auc_score(dataset.experimental_values, dataset.predicted_values),
        dataset=dataset,
    )


def get_roc_curve_with_uncertainty(
    dataset: ActiveInactiveDataset, model_id: str, n_bootstraps: int = 1000
) -> RocCurveUncertainty:
    fpr, tpr, thresholds = roc_curve(
        dataset.experimental_values, dataset.predicted_values
    )
    aucs = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(
            range(len(dataset.experimental_values)),
            len(dataset.experimental_values),
            replace=True,
        )
        if len(np.unique(dataset.experimental_values[indices])) < 2:
            continue
        aucs.append(
            roc_auc_score(
                dataset.experimental_values[indices], dataset.predicted_values[indices]
            )
        )
    return_curve = RocCurveUncertainty(
        id=model_id,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
        auc=np.mean(aucs),
        auc_ci=get_ci_from_bootstraps(aucs),
        dataset=dataset,
    )
    # This is hacky, without this `dataset` will return None
    return_curve.dataset = dataset
    return return_curve
