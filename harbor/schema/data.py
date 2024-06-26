from pydantic import BaseModel, Field, model_validator
from enum import Enum
import pandas as pd, numpy as np

from harbor.schema.measurement_types import (
    ExperimentalMeasurementType,
    PredictedMeasurementType,
    IsActive,
)


class Molecule(BaseModel):
    """
    Molecule
    """

    id: str = Field(..., description="ID")
    smiles: str = Field(None, description="SMILES string")


class MoleculePropertyType(Enum):
    """
    MoleculePropertyType
    """

    pka = "pka"
    logp = "logp"
    solubility = "solubility"
    melting_point = "melting_point"


class Prediction(BaseModel):
    """
    Prediction
    """

    molecule: Molecule = Field(..., description="Molecule")
    type: PredictedMeasurementType = Field(..., description="Prediction type")
    value: float = Field(..., description="Prediction")


class Experiment(BaseModel):
    """
    Experiment
    """

    molecule: Molecule = Field(..., description="Molecule")
    type: ExperimentalMeasurementType = Field(..., description="Experiment type")
    value: float = Field(..., description="Experimental value")

    def to_active_inactive(self, threshold: float) -> "Experiment":
        if self.type.higher_is_better:
            return Experiment(
                molecule=self.molecule,
                type=IsActive,
                value=1 if self.value >= threshold else 0,
            )
        else:
            return Experiment(
                molecule=self.molecule,
                type=IsActive,
                value=1 if self.value <= threshold else 0,
            )


class Dataset(BaseModel):
    """
    Dataset
    """

    molecules: list[Molecule] = Field(..., description="Molecules")
    predictions: list[Prediction] = Field(..., description="Predictions")
    experiments: list[Experiment] = Field(..., description="Experiment")
    prediction_type: PredictedMeasurementType = Field(
        ..., description="Prediction type"
    )
    experiment_type: ExperimentalMeasurementType = Field(
        ..., description="Experiment type"
    )

    @property
    def predicted_values(self) -> np.ndarray:
        return np.array([p.value for p in self.predictions])

    def get_higher_is_better_values(self) -> np.ndarray:
        if self.prediction_type.higher_is_better:
            return self.predicted_values
        else:
            return np.array([-p.value for p in self.predictions])

    @property
    def experimental_values(self) -> np.ndarray:
        return np.array([e.value for e in self.experiments])

    @property
    def molecule_ids(self) -> np.ndarray:
        return np.array([p.molecule.id for p in self.predictions])

    def to_active_inactive(self, threshold: float) -> "ActiveInactiveDataset":
        experiments = [e.to_active_inactive(threshold) for e in self.experiments]
        return ActiveInactiveDataset(
            molecules=self.molecules,
            predictions=self.predictions,
            experiments=experiments,
            prediction_type=self.prediction_type,
            experiment_type=IsActive,
        )

    @model_validator(mode="after")
    def check_consistency(self):
        if not all(p.type == self.prediction_type for p in self.predictions):
            raise ValueError("Inconsistent prediction types")
        if not all(e.type == self.experiment_type for e in self.experiments):
            raise ValueError("Inconsistent experiment types")
        if not len(self.molecules) == len(self.predictions) == len(self.experiments):
            raise ValueError(
                f"Inconsistent number of "
                f"molecules({len(self.molecules)}), "
                f"predictions({len(self.predictions)}), and "
                f"experiments({len(self.experiments)})"
            )

    @classmethod
    def from_csv(
        cls,
        filename: str,
        id_column: str,
        experimental_data_column: str,
        prediction_column: str,
        prediction_type: PredictedMeasurementType,
        experiment_type: ExperimentalMeasurementType,
        smiles_column: str = None,
    ) -> "Dataset":
        df = pd.read_csv(filename)
        return cls.from_dataframe(
            df,
            id_column,
            experimental_data_column,
            prediction_column,
            prediction_type,
            experiment_type,
            smiles_column,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        id_column: str,
        experimental_data_column: str,
        prediction_column: str,
        prediction_type: PredictedMeasurementType,
        experiment_type: ExperimentalMeasurementType,
        smiles_column: str = None,
    ) -> "Dataset":
        molecules = []
        predictions = []
        experiments = []
        for row in df.iterrows():
            if row[1][id_column] is None:
                raise ValueError(f"ID is missing in {row}")
            if row[1][experimental_data_column] is None:
                raise ValueError(f"Experimental data is missing in {row}")
            if smiles_column and row[1][smiles_column] is None:
                raise ValueError(f"SMILES is missing in {row}")

            molecule = (
                Molecule(id=row[1][id_column], smiles=row[1][smiles_column])
                if smiles_column
                else Molecule(id=row[1][id_column])
            )
            molecules.append(molecule)
            predictions.append(
                Prediction(
                    molecule=molecule,
                    type=prediction_type,
                    value=row[1][prediction_column],
                )
            )
            experiments.append(
                Experiment(
                    molecule=molecule,
                    type=experiment_type,
                    value=row[1][experimental_data_column],
                )
            )
        return cls(
            molecules=molecules,
            predictions=predictions,
            experiments=experiments,
            prediction_type=prediction_type,
            experiment_type=experiment_type,
        )


class ActiveInactiveDataset(Dataset):
    """
    ActiveInactiveDataset
    """

    experiment_type: ExperimentalMeasurementType = Field(
        IsActive, frozen=True, description="Experiment type"
    )

    @property
    def n_actives(self) -> int:
        return int(np.sum(self.experimental_values))

    @property
    def n_inactives(self) -> int:
        return int(len(self.experimental_values) - self.n_actives)
