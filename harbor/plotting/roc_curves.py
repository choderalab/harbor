from pydantic import BaseModel, Field
import pandas as pd
from harbor.plot_schema import RocCurve, RocCurveUncertainty


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
