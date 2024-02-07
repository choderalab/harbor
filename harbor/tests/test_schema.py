import pytest


def test_dataset_loading_from_csv(molecule_csv, molecule_data):
    from harbor.data import Dataset

    dataset = Dataset.from_csv(
        molecule_csv, "molecule_chembl_id", "pIC50", "glide_docking"
    )
    assert dataset.molecule_ids == molecule_data["molecule_chembl_id"].tolist()
    assert dataset.predicted_values == molecule_data["glide_docking"].tolist()
    assert dataset.experimental_values == molecule_data["pIC50"].tolist()


def test_dataset_loading_from_dataframe(molecule_csv, molecule_data):
    from harbor.data import Dataset

    dataset = Dataset.from_dataframe(
        molecule_data,
        "molecule_chembl_id",
        "pIC50",
        "glide_docking",
    )
    assert dataset.molecule_ids == molecule_data["molecule_chembl_id"].tolist()
    assert dataset.predicted_values == molecule_data["glide_docking"].tolist()
    assert dataset.experimental_values == molecule_data["pIC50"].tolist()
