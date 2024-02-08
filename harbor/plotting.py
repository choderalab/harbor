import plotly.express as px
import plotly.graph_objects as go
from pydantic import BaseModel, Field
from typing import Union
import pandas as pd
from harbor.plot_schema import RocCurve, PrecisionRecallCurve, RocCurveUncertainty


def get_plotly_df_from_roc_curves(
    roc_curves: list[Union[RocCurve, RocCurveUncertainty]]
):
    """
    Convert a list of ROC curves to a pandas DataFrame
    :param roc_curves:
    :return:
    """
    return pd.DataFrame(
        {
            "fpr": roc_curve.fpr,
            "tpr": roc_curve.tpr,
            "thresholds": roc_curve.thresholds,
        }
    )


class RocCurvesPlotly(BaseModel):
    """
    Input schema for plotting roc curves
    """

    id: str = Field(..., description="The title to be given to the figure")
    roc_curves: list[Union[RocCurve, RocCurveUncertainty]] = Field(
        ..., description="The ROC curves to be plotted"
    )

    @property
    def plotly_df(self):
        return get_plotly_df_from_roc_curves(self.roc_curves)
