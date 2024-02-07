from pydantic import BaseModel, Field, field_validator
from enum import Enum
import pandas as pd


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


class ExperimentType(Enum):
    """
    ExperimentType
    """

    pic50 = "pic50"
    ic50 = "ic50"
    ki = "ki"
    kd = "kd"
    ec50 = "ec50"
    relative_activity = "relative_activity"
    relative_inhibition = "relative_inhibition"


class PredictionType(Enum):
    """
    PredictionType
    """

    docking = "docking"


class Prediction(BaseModel):
    """
    Prediction
    """

    molecule: Molecule = Field(..., description="Molecule")
    type: PredictionType = Field(..., description="Prediction type")
    value: float = Field(..., description="Prediction")


class Experiment(BaseModel):
    """
    Experiment
    """

    molecule: Molecule = Field(..., description="Molecule")
    type: ExperimentType = Field(..., description="Experiment type")


class Dataset(BaseModel):
    """
    Dataset
    """

    molecules: list[Molecule] = Field(..., description="Molecules")
    predictions: list[Prediction] = Field(..., description="Predictions")
    experiments: list[Experiment] = Field(..., description="Experiment")
    prediction_type: PredictionType = Field(..., description="Prediction type")
    experiment_type: ExperimentType = Field(..., description="Experiment type")

    @property
    def predicted_values(self):
        return [p.value for p in self.predictions]

    @property
    def experimental_values(self):
        return [self.experiment] * len(self.experiments)

    @property
    def molecule_ids(self):
        return [p.molecule.id for p in self.predictions]

    @field_validator("experiments")
    def check_experiments(cls, v):
        if len(v) != len(cls.molecules):
            raise ValueError("Number of experiments does not match number of molecules")
        if not all(e.type == cls.experiment_type for e in cls.experiments):
            raise ValueError("Experiment type does not match")
        return v

    @classmethod
    def from_csv(
        cls,
        filename: str,
        id_column: str,
        experimental_data_column: str,
        prediction_column: str,
        prediction_type: PredictionType = PredictionType.docking,
        experiment_type: ExperimentType = ExperimentType.pic50,
        smiles_column: str = None,
    ) -> "Dataset":
        df = pd.read_csv(filename)
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
            experiments.append(Experiment(molecule=molecule, type=experiment_type))
        return cls(
            molecules=molecules,
            predictions=predictions,
            experiments=experiments,
            prediction_type=prediction_type,
            experiment_type=experiment_type,
        )
