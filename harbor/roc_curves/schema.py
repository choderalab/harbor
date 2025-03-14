import numpy as np
from pydantic import BaseModel, Field
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from harbor.analysis.utils import get_ci_from_bootstraps
from harbor.schema.data import ActiveInactiveDataset


class CurveBase(BaseModel):
    """
    Base class for a curve
    """

    id: str = Field(..., description="The name of the model assessed in this curve")

    def to_df(self):
        """
        Convert the curve to a pandas DataFrame
        """
        raise NotImplementedError


class RocCurve(BaseModel):
    """
    ROC curve
    """

    model_id: str = Field(
        ..., description="The name of the model assessed in this curve"
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

    model_id: str = Field(
        ..., description="The name of the model assessed in this curve"
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

    precision: list[float] = Field(..., description="Precision (x-axis)")
    recall: list[float] = Field(..., description="Recall (y-axis)")
    auc: float = 0.0


class RocCurvesPlotly(BaseModel):
    """
    Input schema for plotting roc curves
    """

    id: str = Field(..., description="The title to be given to the figure")
    roc_curves: list[RocCurve | RocCurveUncertainty] = Field(
        ..., description="The ROC curves to be plotted"
    )

    @property
    def roc_df(self) -> pd.DataFrame:
        """
        Convert the roc curves to a dataframe
        """
        dfs = []
        for roc_curve in self.roc_curves:
            fpr = roc_curve.fpr
            tpr = roc_curve.tpr
            dfs.append(
                pd.DataFrame(
                    {
                        "False Positive Rate": fpr,
                        "True Positive Rate": tpr,
                        "Protocol": roc_curve.model_id,
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)

    @property
    def summary_df(self):
        """
        Return a summary of the roc curves
        """
        dfs = []
        for roc_curve in self.roc_curves:
            dfs.append(
                pd.DataFrame(
                    {
                        "Protocol": [roc_curve.id],
                        "AUC": [roc_curve.auc_str],
                        "Number of Actives": [roc_curve.dataset.n_actives],
                        "Number of Inactives": [roc_curve.dataset.n_inactives],
                        "Number of Total": [len(roc_curve.dataset.experimental_values)],
                    }
                )
            )
        return pd.concat(dfs, ignore_index=True)


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
