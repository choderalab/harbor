from pydantic import BaseModel, Field


class RocCurve(BaseModel):
    """
    ROC curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    fpr: list[float] = Field(..., description="False positive rate (x-axis)")
    tpr: list[float] = Field(..., description="True positive rate (y-axis)")
    auc: float = 0.0


class PrecisionRecallCurve(BaseModel):
    """
    Precision recall curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")
    precision: list[float] = Field(..., description="Precision (x-axis)")
    recall: list[float] = Field(..., description="Recall (y-axis)")
    auc: float = 0.0
