from harbor.data import ActiveInactiveDataset
from harbor.plot_schema import RocCurve, PrecisionRecallCurve, RocCurveUncertainty
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from pydantic import BaseModel, Field
import numpy as np


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

        fpr, tpr, _ = roc_curve(
            dataset.experimental_values[indices], dataset.predicted_values[indices]
        )
        aucs.append(auc(fpr, tpr))
    return RocCurveUncertainty(
        id=model_id,
        fpr=fpr.tolist(),
        tpr=tpr.tolist(),
        thresholds=thresholds.tolist(),
        auc=roc_auc_score(dataset.experimental_values, dataset.predicted_values),
        auc_uncertainty=np.std(aucs),
    )


class ModelEvaluator(BaseModel):
    """
    ModelEvaluator
    """

    dataset: ActiveInactiveDataset = Field(..., description="Dataset to evaluate")

    def get_precision_recall_curve(
        self, model_id: str, model_predictions: list[float]
    ) -> dict:
        precision, recall, _ = precision_recall_curve(
            self.dataset.experimental_values, model_predictions
        )
        return {
            "id": model_id,
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "auc": auc(recall, precision),
        }
