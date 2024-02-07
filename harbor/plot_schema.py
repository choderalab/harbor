from pydantic import BaseModel, Field
import numpy as np


class RocCurve(BaseModel):
    """
    ROC curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    fpr: list[float] = Field(..., description="False positive rate (x-axis)")
    tpr: list[float] = Field(..., description="True positive rate (y-axis)")
    thresholds: list[float] = Field(..., description="Thresholds")
    auc: float = 0.0


class RocCurveUncertainty(BaseModel):
    """
    ROC curve with uncertainty
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    fpr: list[float] = Field(..., description="False positive rate (x-axis)")
    tpr: list[float] = Field(..., description="True positive rate (y-axis)")
    thresholds: list[float] = Field(..., description="Thresholds")
    auc: float = 0.0
    auc_ci: tuple[float, float] = (0.0, 0.0)

    @property
    def auc_ci_lower(self) -> float:
        return self.auc_ci[0]

    @property
    def auc_ci_upper(self) -> float:
        return self.auc_ci[1]


class PrecisionRecallCurve(BaseModel):
    """
    Precision recall curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    precision: list[float] = Field(..., description="Precision (x-axis)")
    recall: list[float] = Field(..., description="Recall (y-axis)")
    auc: float = 0.0
