from harbor.data import ActiveInactiveDataset
from harbor.plot_schema import RocCurve, PrecisionRecallCurve
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from pydantic import BaseModel, Field


class ModelEvaluator(BaseModel):
    """
    ModelEvaluator
    """

    dataset: ActiveInactiveDataset = Field(..., description="Dataset to evaluate")

    def get_roc_curve(self, model_id: str) -> RocCurve:
        fpr, tpr, thresholds = roc_curve(
            self.dataset.experimental_values, self.dataset.predicted_values
        )
        print(thresholds)
        return RocCurve(
            id=model_id,
            fpr=fpr.tolist(),
            tpr=tpr.tolist(),
            auc=roc_auc_score(
                self.dataset.experimental_values, self.dataset.predicted_values
            ),
        )

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
