import pytest
import pandas as pd


@pytest.fixture(scope="session")
def molecule_data():
    return pd.DataFrame(
        {
            "molecule_chembl_id": [
                "CHEMBL63786",
                "CHEMBL35820",
                "CHEMBL53711",
                "CHEMBL66031",
                "CHEMBL53753",
            ],
            "smiles": [
                "Brc1cccc(Nc2ncnc3cc4ccccc4cc23)c1",
                "CCOc1cc2ncnc(Nc3cccc(Br)c3)c2cc1OCC",
                "CN(C)c1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
                "Brc1cccc(Nc2ncnc3cc4[nH]cnc4cc23)c1",
                "CNc1cc2c(Nc3cccc(Br)c3)ncnc2cn1",
            ],
            "pIC50": [11.522879, 11.221849, 11.221849, 11.096910, 11.096910],
            "glide_docking": [10, 11, 9, 11, 12],
        }
    )


@pytest.fixture(scope="session")
def molecule_csv(tmpdir_factory, molecule_data):
    file = tmpdir_factory.mktemp("data").join("molecule.csv")
    molecule_data.to_csv(file, index=False)
    return file


@pytest.fixture(scope="session")
def dataset(molecule_data):
    from harbor.data import Dataset

    return Dataset.from_dataframe(
        molecule_data,
        "molecule_chembl_id",
        "pIC50",
        "glide_docking",
        smiles_column="smiles",
    )


@pytest.fixture(scope="session")
def active_inactive_dataset(dataset):
    return dataset.to_active_inactive(11.22)
