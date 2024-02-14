from pydantic import BaseModel, Field
import numpy as np
from harbor.data import ActiveInactiveDataset


class RocCurve(BaseModel):
    """
    ROC curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    dataset: ActiveInactiveDataset = Field(
        ..., description="The dataset used to generate this curve"
    )
    fpr: list[float] = Field(..., description="False positive rate (x-axis)")
    tpr: list[float] = Field(..., description="True positive rate (y-axis)")
    thresholds: list[float] = Field(..., description="Thresholds")
    auc: float = 0.0

    @property
    def auc_str(self) -> str:
        return f"{self.auc:.2f}"


class RocCurveUncertainty(BaseModel):
    """
    ROC curve with uncertainty
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    dataset: ActiveInactiveDataset = Field(
        ..., description="The dataset used to generate this curve"
    )
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

    @property
    def auc_str(self) -> str:
        return (
            f"{self.auc:.2f} [95%: {self.auc_ci_lower: .2f}, {self.auc_ci_upper:.2f}]"
        )


class PrecisionRecallCurve(BaseModel):
    """
    Precision recall curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    precision: list[float] = Field(..., description="Precision (x-axis)")
    recall: list[float] = Field(..., description="Recall (y-axis)")
    auc: float = 0.0
