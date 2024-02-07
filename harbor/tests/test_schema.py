import pytest
from harbor.data import Dataset, ActiveInactiveDataset
import numpy as np


def test_dataset_loading_from_csv(molecule_csv, molecule_data):
    dataset = Dataset.from_csv(
        molecule_csv, "molecule_chembl_id", "pIC50", "glide_docking"
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
    )
    assert np.array_equal(dataset.molecule_ids, molecule_data["molecule_chembl_id"])
    assert np.allclose(dataset.predicted_values, molecule_data["glide_docking"])
    assert np.allclose(dataset.experimental_values, molecule_data["pIC50"])


def test_dataset_to_active_inactive(molecule_data):
    from harbor.data import ExperimentType

    dataset = Dataset.from_dataframe(
        molecule_data, "molecule_chembl_id", "pIC50", "glide_docking"
    )
    active_inactive = dataset.to_active_inactive(11.22)
    assert np.array_equal(active_inactive.experimental_values, [1, 1, 1, 0, 0])
    assert active_inactive.experiment_type == ExperimentType.is_active
    assert isinstance(active_inactive, ActiveInactiveDataset)
