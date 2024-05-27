from harbor.analysis.utils import get_ci_from_bootstraps
from harbor.schema.data import ActiveInactiveDataset
from harbor.plot_schema import RocCurve, RocCurveUncertainty
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


def get_roc_curve(dataset: ActiveInactiveDataset, model_id: str) -> RocCurve:
    fpr, tpr, thresholds = roc_curve(
        dataset.experimental_values, dataset.get_higher_is_better_values()
    )
    return RocCurve(
        model_id=model_id,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
        auc=roc_auc_score(
            dataset.experimental_values, dataset.get_higher_is_better_values()
        ),
    )


def get_roc_curve_with_uncertainty(
    dataset: ActiveInactiveDataset, model_id: str, n_bootstraps: int = 1000
) -> RocCurveUncertainty:
    fpr, tpr, thresholds = roc_curve(
        dataset.experimental_values, dataset.get_higher_is_better_values()
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
                dataset.experimental_values[indices],
                dataset.get_higher_is_better_values()[indices],
            )
        )
    return RocCurveUncertainty(
        model_id=model_id,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
        auc=np.mean(aucs),
        auc_ci=get_ci_from_bootstraps(aucs),
    )
