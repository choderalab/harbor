from harbor.schema.measurement_types import IsActive, DockingScore, pIC50
from harbor.schema.data import Dataset, ActiveInactiveDataset
import numpy as np


def test_dataset_loading_from_csv(molecule_csv, molecule_data):
    dataset = Dataset.from_csv(
        molecule_csv,
        "molecule_chembl_id",
        "pIC50",
        "glide_docking",
        prediction_type=DockingScore,
        experiment_type=pIC50,
        smiles_column="smiles",
    )
    assert np.array_equal(dataset.molecule_ids, molecule_data["molecule_chembl_id"])
    assert np.allclose(dataset.predicted_values, molecule_data["glide_docking"])
    assert np.allclose(dataset.experimental_values, molecule_data["pIC50"])


def test_dataset_loading_from_dataframe(molecule_csv, molecule_data):
    dataset = Dataset.from_dataframe(
        molecule_data,
        "molecule_chembl_id",
        "pIC50",
        "glide_docking",
        prediction_type=DockingScore,
        experiment_type=pIC50,
        smiles_column="smiles",
    )
    assert np.array_equal(dataset.molecule_ids, molecule_data["molecule_chembl_id"])
    assert np.allclose(dataset.predicted_values, molecule_data["glide_docking"])
    assert np.allclose(dataset.experimental_values, molecule_data["pIC50"])


def test_dataset_to_active_inactive(molecule_data):

    dataset = Dataset.from_dataframe(
        molecule_data,
        "molecule_chembl_id",
        "pIC50",
        "glide_docking",
        prediction_type=DockingScore,
        experiment_type=pIC50,
        smiles_column="smiles",
    )
    active_inactive = dataset.to_active_inactive(11.22)
    assert np.array_equal(active_inactive.experimental_values, [1, 1, 1, 0, 0])
    assert active_inactive.experiment_type == IsActive
    assert isinstance(active_inactive, ActiveInactiveDataset)
