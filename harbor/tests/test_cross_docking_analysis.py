import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import patch
from pathlib import Path

from harbor.analysis.cross_docking import (
    Evaluator,
    PoseSelector,
    DateSplit,
    ColumnFilter,
    SimilaritySplit,
    ScaffoldSplit,
    RandomSplit,
    Scorer,
    ScaffoldSplitOptions,
    BinaryEvaluation,
    DataFrameModel,
    DataFrameType,
    DockingDataModel,
    FractionGood,
    Settings,
    get_unique_structures_randomized_by_date,
)
import pandas as pd
import numpy as np
import itertools


@pytest.fixture(autouse=True)
def setup_random_state():
    """Ensure consistent random state for all tests."""
    np.random.seed(42)
    yield
    np.random.seed(None)  # Reset the random state after each test


@pytest.fixture()
def refs():
    """Sample reference structures fixture."""
    return [f"PDB{i}" for i in range(1, 10)]


@pytest.fixture()
def ligs():
    """Sample ligands fixture."""
    return [f"LIG_{i}" for i in range(1, 10)]


@pytest.fixture()
def ref_dataframe(refs):
    """Sample reference data fixture."""
    return pd.DataFrame(
        {
            "Reference_Structure": refs,
            "Ref_Data_1": [np.random.random() for ref in refs],
            "Date": [datetime.now() - timedelta(days=i) for i in range(len(refs))],
            "Ref_Scaffold": [
                np.random.choice([f"Scaffold_{i}" for i in range(5)]) for _ in refs
            ],
        }
    )


@pytest.fixture()
def lig_dataframe(ligs):
    """Sample ligand data fixture."""
    return pd.DataFrame(
        {
            "Query_Ligand": ligs,
            "Lig_Data_1": [np.random.random() for lig in ligs],
            "Query_Scaffold": [
                np.random.choice([f"Scaffold_{i}" for i in range(5)]) for _ in ligs
            ],
        }
    )


@pytest.fixture()
def pose_dataframe(refs, ligs):
    """Sample pose results fixture."""
    return pd.DataFrame.from_records(
        [
            {
                "Reference_Structure": ref,
                "Query_Ligand": lig,
                "RMSD": np.random.random() * 8,
                "Pose_ID": pose,
            }
            for ref, lig, pose in itertools.product(refs, ligs, range(0, 2))
        ]
    )


@pytest.fixture()
def ecfp_dataframe(refs, ligs):
    """Sample ECFP data fixture."""
    return pd.DataFrame.from_records(
        [
            {
                "Reference_Structure": ref,
                "Query_Ligand": lig,
                "Tanimoto": np.random.random(),
                "radius": radius,
                "bitsize": bitsize,
            }
            for ref, lig, radius, bitsize in itertools.product(
                refs, ligs, [2, 5], [2048]
            )
        ]
    )


@pytest.fixture()
def tanimotocombo_data(refs, ligs):
    """Sample TanimotoCombo data fixture."""
    return pd.DataFrame.from_records(
        [
            {
                "Reference_Structure": ref,
                "Query_Ligand": lig,
                "Tanimoto": np.random.random(),
                "Aligned": aligned,
            }
            for ref, lig, aligned in itertools.product(refs, ligs, ["True", "False"])
        ]
    )


@pytest.fixture()
def scaffold_dataframe(refs, ligs, ref_dataframe, lig_dataframe):
    """Sample scaffold data fixture."""
    return pd.DataFrame.from_records(
        [
            {
                "Reference_Structure": ref,
                "Query_Ligand": lig,
                "Tanimoto": (
                    1
                    if ref_dataframe[ref_dataframe["Reference_Structure"] == ref][
                        "Ref_Scaffold"
                    ].values[0]
                    == lig_dataframe[lig_dataframe["Query_Ligand"] == lig][
                        "Query_Scaffold"
                    ].values[0]
                    else 0
                ),
            }
            for ref, lig in itertools.product(refs, ligs)
        ]
    )


def test_column_filter(pose_dataframe):
    """Test the ColumnFilter class."""

    cf = ColumnFilter(
        column="RMSD",
        value=4.0,
        operator="le",
    )
    filtered_df = cf.filter(pose_dataframe)
    assert len(filtered_df) == 86  # 2 poses per reference-ligand pair
    assert all(filtered_df["RMSD"] <= 4.0)

    # Test with a different operator
    cf = ColumnFilter(
        column="RMSD",
        value=4.0,
        operator="ge",
    )
    filtered_df = cf.filter(pose_dataframe)
    assert len(filtered_df) == 76
    assert all(filtered_df["RMSD"] >= 4.0)


class TestDockingDataModel:

    @pytest.fixture()
    def dataframe_model(self, pose_dataframe):
        data = DataFrameModel(
            name="PoseData",
            type=DataFrameType.POSE,
            dataframe=pose_dataframe,
            key_columns=["Query_Ligand", "Reference_Structure", "Pose_ID"],
        )
        return data

    def test_serialization(self, dataframe_model, tmpdir):
        fp = dataframe_model.serialize(Path(tmpdir) / "test_pose_data")
        assert fp.with_suffix(".parquet").exists()
        assert fp.with_suffix(".json").exists()

        loaded = DataFrameModel.deserialize(fp)

        assert loaded == dataframe_model

    def test_merge(self, pose_dataframe, ref_dataframe, lig_dataframe):
        pose_df = DataFrameModel(
            name="PoseData",
            type=DataFrameType.POSE,
            dataframe=pose_dataframe,
            key_columns=["Query_Ligand", "Reference_Structure", "Pose_ID"],
        )
        ref_df = DataFrameModel(
            name="RefData",
            type=DataFrameType.REFERENCE,
            dataframe=ref_dataframe,
            key_columns=["Reference_Structure"],
        )
        lig_df = DataFrameModel(
            name="QueryData",
            type=DataFrameType.QUERY,
            dataframe=lig_dataframe,
            key_columns=["Query_Ligand"],
        )
        merged = DockingDataModel.from_models([pose_df, ref_df, lig_df])
        assert merged.dataframe.size == 1458

        assert set(merged.get_key_columns()) == {
            "Query_Ligand",
            "Reference_Structure",
            "Pose_ID",
        }

    @pytest.fixture()
    def docking_data_model(self, pose_dataframe, ref_dataframe, lig_dataframe):
        pose_df = DataFrameModel(
            name="PoseData",
            type=DataFrameType.POSE,
            dataframe=pose_dataframe,
            key_columns=["Query_Ligand", "Reference_Structure", "Pose_ID"],
        )
        ref_df = DataFrameModel(
            name="RefData",
            type=DataFrameType.REFERENCE,
            dataframe=ref_dataframe,
            key_columns=["Reference_Structure"],
        )
        lig_df = DataFrameModel(
            name="QueryData",
            type=DataFrameType.QUERY,
            dataframe=lig_dataframe,
            key_columns=["Query_Ligand"],
        )
        return DockingDataModel.from_models([pose_df, ref_df, lig_df])

    def test_docking_data_model_serialization(self, docking_data_model, tmpdir):
        fp = docking_data_model.serialize(Path(tmpdir) / "test_pose_data")
        assert fp.with_suffix(".parquet").exists()
        assert fp.with_suffix(".json").exists()

        loaded = DockingDataModel.deserialize(fp)

        assert loaded == docking_data_model

    def test_docking_data_model_methods(self, refs, docking_data_model):
        assert set(docking_data_model.get_unique_refs()) == set(refs)
        assert len(docking_data_model.get_unique_refs()) == 9

    def test_random_split(self, docking_data_model):
        rs = RandomSplit(n_reference_structures=5, reference_structure_column="Reference_Structure")

        results = rs.run(docking_data_model, bootstraps=10)
        unique_refs = [str(sorted(result.get_unique_refs())) for result in results]
        assert len(set(unique_refs)) > 1

    def test_date_split(self, docking_data_model):
        date_split = DateSplit(
            n_reference_structures=3,
            reference_structure_column="Reference_Structure",
            date_column="Date",
            randomize_by_n_days=3,
        )
        splits = date_split.run(docking_data_model, bootstraps=10)
        assert len(splits) == 10

        # make sure the splits are not all the same
        assert len(set([tuple(split.get_unique_refs()) for split in splits])) > 1

        # make sure the splits have the right number of structures
        assert all(len(split.get_unique_refs()) == 3 for split in splits)

    @pytest.mark.parametrize(
        "split_option,query_subset,ref_subset",
        [
            (ScaffoldSplitOptions.X_TO_Y, "Scaffold_1", "Scaffold_2"),
            (ScaffoldSplitOptions.X_TO_NOT_X, "Scaffold_1", None),
            (ScaffoldSplitOptions.NOT_X_TO_X, None, "Scaffold_1"),
            (ScaffoldSplitOptions.X_TO_ALL, "Scaffold_2", None),
            (ScaffoldSplitOptions.ALL_TO_X, None, "Scaffold_2"),
            (ScaffoldSplitOptions.X_TO_X, "Scaffold_4", None)
        ]
    )
    def test_x_to_y_scaffold_split(self, split_option, query_subset, ref_subset, docking_data_model, tmpdir):
        fp = docking_data_model.serialize(tmpdir / "test_file")
        loaded = DockingDataModel.deserialize(fp)
        """Test the ScaffoldSplit class."""
        scaffold_split = ScaffoldSplit(
            query_scaffold_id_column="Query_Scaffold",
            reference_scaffold_id_column="Ref_Scaffold",
            split_option=split_option,
            reference_scaffold_id_subset=[ref_subset] if ref_subset else None,
            query_scaffold_id_subset=[query_subset] if query_subset else None,
        )
        splits = scaffold_split.run(loaded)
        assert len(splits) == 1

        split_data = splits[0]

        combined_df = split_data.dataframe

        # check that the ref and query scaffolds are the only ones included
        if ref_subset:
            assert all(
                combined_df["Ref_Scaffold"].isin([ref_subset])
            ), f"Ref_Scaffold should only contain {ref_subset}"
        if query_subset:
            assert all(
                combined_df["Query_Scaffold"].isin([query_subset])
            ), f"Query_Scaffold should only contain {query_subset}"

        if split_option in (
                ScaffoldSplitOptions.X_TO_Y,
                ScaffoldSplitOptions.X_TO_NOT_X,
                ScaffoldSplitOptions.NOT_X_TO_X,
        ):
            # None of the scaffolds should match
            assert (
                    len(
                        combined_df[
                            combined_df["Query_Scaffold"] == combined_df["Ref_Scaffold"]
                            ]
                    )
                    == 0
            )
        elif split_option in ScaffoldSplitOptions.X_TO_ALL:
            # should have the same refs as before
            assert set(split_data.get_unique_refs()) == set(docking_data_model.get_unique_refs())
        elif split_option in ScaffoldSplitOptions.ALL_TO_X:
            # should have the same ligs as before
            assert set(split_data.get_unique_ligs()) == set(docking_data_model.get_unique_ligs())

    @pytest.mark.parametrize(
        "split_option,query_subset,ref_subset",
        [
            # X_TO_X requires at least one to be set and for it to be single
            (ScaffoldSplitOptions.X_TO_X, None, None),
            (ScaffoldSplitOptions.X_TO_X, ["Scaffold_1", "Scaffold_2"], None),

            # X_TO_NOT_X requires query subset
            (ScaffoldSplitOptions.X_TO_NOT_X, None, ["Scaffold_1"]),

            # NOT_X_TO_X requires reference subset
            (ScaffoldSplitOptions.NOT_X_TO_X, ["Scaffold_1"], None),

            # X_TO_Y requires both subsets and they can't overlap
            (ScaffoldSplitOptions.X_TO_Y, None, ["Scaffold_1"]),
            (ScaffoldSplitOptions.X_TO_Y, ["Scaffold_1"], None),
            (ScaffoldSplitOptions.X_TO_Y, ["Scaffold_1"], ["Scaffold_1"]),

            # ALL_TO_X requires reference subset
            (ScaffoldSplitOptions.ALL_TO_X, ["Scaffold_1"], None),

            # X_TO_ALL requires query subset
            (ScaffoldSplitOptions.X_TO_ALL, None, ["Scaffold_1"]),
        ]
    )
    def test_scaffold_split_incompatible_combinations(
            self, docking_data_model, split_option, query_subset, ref_subset
    ):
        """Test that ScaffoldSplit raises appropriate errors for incompatible combinations."""
        with pytest.raises(ValueError):
            scaffold_split = ScaffoldSplit(
                query_scaffold_id_column="Query_Scaffold",
                reference_scaffold_id_column="Ref_Scaffold",
                split_option=split_option,
                query_scaffold_id_subset=query_subset,
                reference_scaffold_id_subset=ref_subset,
            )
            scaffold_split.run(docking_data_model)


def test_settings():
    settings = Settings()
    settings.to_yaml_file("test.yaml")

    import numpy as np

    new_settings = Settings(n_per_split=np.arange(1, 21))
    s2 = Settings.from_yaml_file("test.yaml")


def test_create_evaluators_from_settings():
    settings = Settings.from_yaml_file("test.yaml")
    settings.n_reference_structures = [1]
    evs = settings.create_evaluators()
    assert len(evs) == 2


def test_fraction_good():
    fg = FractionGood(total=100, fraction=0.5, replicates=[0.5, 0.6, 0.7])
    assert fg.get_records() == {
        "Min": 0.5,
        "Max": 0.7,
        "CI_Upper": 0.7,
        "CI_Lower": 0.5,
        "Total": 100,
        "Fraction": 0.5,
    }

    with pytest.raises(ValueError):
        FractionGood(total=100, fraction=1.1, replicates=[0.5, 0.6, 0.7, 0.8])


def test_date_split():
    ds = DateSplit(
        reference_structure_column="help",
        n_per_split=100,
        date_dict={"s1": "2021-01-01"},
        randomize_by_n_days=1,
    )


def test_serialization():
    ev = Evaluator(
        pose_selector=PoseSelector(
            name="Default", variable="Pose_ID", number_to_return=1
        ),
        dataset_split=RandomSplit(reference_structure_column="help", n_per_split=100),
        scorer=Scorer(
            name="RMSD", variable="RMSD", higher_is_better=False, number_to_return=1
        ),
        evaluator=BinaryEvaluation(variable="RMSD", cutoff=2),
        groupby=["Query_Ligand"],
    )
    ev.to_json_file("test.json")
    ev2 = Evaluator.from_json_file("test.json")
    assert ev == ev2

    ev3 = Evaluator(
        pose_selector=PoseSelector(
            name="Default", variable="Pose_ID", number_to_return=1
        ),
        dataset_split=DateSplit(
            reference_structure_column="help",
            n_per_split=100,
            date_dict={"s1": "2021-01-01"},
            randomize_by_n_days=1,
        ),
        scorer=Scorer(
            name="RMSD", variable="RMSD", higher_is_better=False, number_to_return=1
        ),
        evaluator=BinaryEvaluation(variable="RMSD", cutoff=2),
        groupby=["Query_Ligand"],
    )
    ds = ev3.dataset_split
    ev3.to_json_file("test.json")
    ev4 = Evaluator.from_json_file("test.json")
    assert ev3 == ev4


@pytest.fixture
def sample_date_dict():
    """Sample data fixture with structures and their dates."""
    return {
        "structure1": "2023-01-01 00:00:00",
        "structure2": "2023-01-15 00:00:00",
        "structure3": "2023-02-01 00:00:00",
        "structure4": "2023-02-15 00:00:00",
        "structure5": "2023-03-01 00:00:00",
        "structure6": "2023-03-15 00:00:00",
        "structure7": "2023-04-01 00:00:00",
        "structure8": "2023-04-15 00:00:00",
    }


def get_dates(structures: list, date_dict) -> list:
    return [
        datetime.strptime(date_dict[structure], "%Y-%m-%d %H:%M:%S")
        for structure in structures
    ]


def test_unique_structures_randomized_by_date(sample_date_dict):
    """Test that the function returns a set of unique structures."""
    unique_structures = sample_date_dict.keys()
    date_dict = sample_date_dict
    for days in [1, 15, 30, 60]:
        for i in range(1, len(sample_date_dict.keys()) + 1):
            result = get_unique_structures_randomized_by_date(
                set(unique_structures),
                date_dict,
                n_structures_to_return=i,
                n_days_to_randomize=days,
            )
            # make sure the result is a set of unique structures with length n_structures_to_return
            assert len(result) == i
            assert isinstance(result, set)

            # make sure that all the returned structures are within the
            # date range of the input structures + n_days_to_randomize
            result_dates = get_dates(list(result), date_dict)
            structure_dates = get_dates(list(unique_structures), date_dict)

            for result in result_dates:
                assert any(
                    abs((result - structure_date).days) <= days
                    for structure_date in structure_dates
                ), f"Result {result} is not within {days} days of any input structure date."


def test_warns_when_no_structures(sample_date_dict):
    """Test that the function raises an error when no structures are provided."""
    unique_structures = set()

    date_dict = sample_date_dict
    with pytest.warns(UserWarning):
        get_unique_structures_randomized_by_date(
            unique_structures,
            date_dict,
            n_structures_to_return=1,
            n_days_to_randomize=30,
        )


def test_raises_value_error_when_missing_in_date_dict(sample_date_dict):
    """Test that the function raises a ValueError when a structure is missing in the date_dict."""
    unique_structures = {"structure10"}
    date_dict = sample_date_dict
    with pytest.raises(ValueError):
        get_unique_structures_randomized_by_date(
            unique_structures,
            date_dict,
            n_structures_to_return=1,
            n_days_to_randomize=30,
        )


def get_random_structure(structure, date_dict, timedelta=timedelta(days=365)) -> str:
    """
    Get a random structure within timedelta of the input structure.

    :param structure:
    :param date_dict:
    :param timedelta:
    :return:
    """
    date = date_dict[structure]
    date = datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    start_date = date - timedelta
    end_date = date + timedelta
    possible_structures = [
        key
        for key, value in date_dict.items()
        if start_date <= datetime.strptime(value, "%Y-%m-%d %H:%M:%S") <= end_date
    ]
    return np.random.choice(possible_structures)


def test_original_list_manipulation_naive():
    """Test the naive approach (original code snippet) could potentially produce duplicates."""
    date_dict = {f"structure{i}": f"2023-01-{i:02d} 00:00:00" for i in range(1, 10)}

    variable_split = ["structure1", "structure2", "structure3"]

    # Simulate the original approach, which could produce duplicates
    with patch("numpy.random.choice", return_value="structure1"):  # Force duplicates
        new_variable_split = [
            get_random_structure(
                structure,
                date_dict,
                timedelta=timedelta(days=30),
            )
            for structure in variable_split
        ]

        # This demonstrates the potential problem - duplicates would lead to a shorter list if deduplicated
        assert len(set(new_variable_split)) <= len(variable_split)
