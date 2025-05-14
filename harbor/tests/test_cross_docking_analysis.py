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
    RMSDScorer,
    POSITScorer,
    ScaffoldSplitOptions,
    BinaryEvaluation,
    DataFrameModel,
    DataFrameType,
    DockingDataModel,
    SuccessRate,
    get_unique_structures_randomized_by_date,
    EvaluatorFactory,
    Results,
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
    return [f"PDB{i}" for i in range(1, 500)]


@pytest.fixture()
def ligs():
    """Sample ligands fixture."""
    return [f"LIG_{i}" for i in range(1, 500)]


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
                "docking-confidence-POSIT": np.random.random(),
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
            for ref, lig, aligned in itertools.product(refs, ligs, [True, False])
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


@pytest.fixture()
def docking_data_model(
    pose_dataframe, ref_dataframe, lig_dataframe, tanimotocombo_data
):
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
    tcdf = DataFrameModel(
        name="TanimotoComboData",
        type=DataFrameType.CHEMICAL_SIMILARITY,
        dataframe=tanimotocombo_data,
        key_columns=["Query_Ligand", "Reference_Structure", "Aligned"],
    )
    return DockingDataModel.from_models([pose_df, ref_df, lig_df, tcdf])


def test_column_filter(pose_dataframe):
    """Test the ColumnFilter class."""

    cf = ColumnFilter(
        column="RMSD",
        value=4.0,
        operator="le",
    )
    filtered_df = cf.filter(pose_dataframe)
    assert len(filtered_df) == 248850  # 2 poses per reference-ligand pair
    assert all(filtered_df["RMSD"] <= 4.0)

    # Test with a different operator
    cf = ColumnFilter(
        column="RMSD",
        value=4.0,
        operator="ge",
    )
    filtered_df = cf.filter(pose_dataframe)
    assert len(filtered_df) == 249152
    assert all(filtered_df["RMSD"] >= 4.0)


class TestDataFrameModel:

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
        assert merged.dataframe.size == 4980020

        assert set(merged.get_key_columns()) == {
            "Query_Ligand",
            "Reference_Structure",
            "Pose_ID",
        }


class TestDockingDataModel:
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
        assert len(docking_data_model.get_unique_refs()) == 499


class TestSplits:

    def test_random_split(self, docking_data_model):
        rs = RandomSplit(
            n_reference_structures=5, reference_structure_column="Reference_Structure"
        )

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
            (ScaffoldSplitOptions.X_TO_X, "Scaffold_4", None),
        ],
    )
    def test_x_to_y_scaffold_split(
        self, split_option, query_subset, ref_subset, docking_data_model, tmpdir
    ):
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

        split_data = splits[0]
        assert isinstance(split_data, DockingDataModel)

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
            assert set(split_data.get_unique_refs()) == set(
                docking_data_model.get_unique_refs()
            )
        elif split_option in ScaffoldSplitOptions.ALL_TO_X:
            # should have the same ligs as before
            assert set(split_data.get_unique_ligs()) == set(
                docking_data_model.get_unique_ligs()
            )

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
        ],
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

    def test_pose_selector(self, docking_data_model):
        pose_selector = PoseSelector(
            name="Default", variable="Pose_ID", number_to_return=1
        )
        new_data = pose_selector.run(docking_data_model)
        assert all(new_data.dataframe["Pose_ID"] == 0)

    def test_similarity_split(self, docking_data_model):
        """Test that SimilaritySplit correctly filters data based on similarity thresholds."""
        # Test with threshold above which to include pairs
        ss_high = SimilaritySplit(
            n_reference_structures=3,
            similarity_column="Tanimoto",
            threshold=0.7,
            include_similar=True,
            higher_is_more_similar=True,
            query_ligand_column="Query_Ligand",
            groupby={"Aligned": True},
        )
        result_high = ss_high.run(docking_data_model)
        assert len(result_high) == 1  # Should return single split
        df_high = result_high[0].dataframe
        assert all(df_high["Tanimoto"] >= 0.7)
        print(ss_high.get_records())

        # Test with threshold below which to include pairs
        ss_low = SimilaritySplit(
            n_reference_structures=3,
            similarity_column="Tanimoto",
            threshold=0.3,
            include_similar=False,
            higher_is_more_similar=True,
            query_ligand_column="Query_Ligand",
            groupby={"Aligned": True},
        )
        result_low = ss_low.run(docking_data_model)
        assert len(result_low) == 1
        df_low = result_low[0].dataframe
        assert all(df_low["Aligned"] == True)
        assert all(df_low["Tanimoto"] < 0.3)

        # Test with bootstrapping
        ss_boot = SimilaritySplit(
            n_reference_structures=3,
            similarity_column="Tanimoto",
            threshold=0.5,
            include_similar=True,
            higher_is_more_similar=True,
            query_ligand_column="Query_Ligand",
            groupby={"Aligned": True},
        )
        result_boot = ss_boot.run(docking_data_model, bootstraps=5)
        assert len(result_boot) == 5

        # Check that different bootstraps generate different reference sets
        ref_sets = [set(split.get_unique_refs()) for split in result_boot]
        assert len(set(tuple(refs) for refs in ref_sets)) > 1

        # Test invalid threshold
        with pytest.raises(ValueError):
            ss_invalid = SimilaritySplit(
                n_reference_structures=3,
                similarity_column="Tanimoto",
                threshold=1.5,  # Invalid threshold > 1
            )
            ss_invalid.run(docking_data_model)


class TestEvaluator:
    @pytest.fixture()
    def pose_selector(self, docking_data_model):
        return PoseSelector(name="Default", variable="Pose_ID", number_to_return=1)

    @pytest.fixture()
    def random_split(self, docking_data_model):
        return RandomSplit(
            reference_structure_column=docking_data_model.get_ref_column(),
            n_reference_structures=5,
        )

    @pytest.fixture()
    def date_split(self, docking_data_model):
        return DateSplit(
            n_reference_structures=3,
            reference_structure_column="Reference_Structure",
            date_column="Date",
            randomize_by_n_days=3,
        )

    @pytest.fixture()
    def rmsd_scorer(self, docking_data_model):
        return RMSDScorer()

    @pytest.fixture()
    def posit_scorer(self, docking_data_model):
        return POSITScorer()

    @pytest.fixture()
    def binary_evaluator(self, docking_data_model):
        return BinaryEvaluation(variable="RMSD", cutoff=2)

    def test_evaluator(
        self,
        docking_data_model,
        refs,
        ligs,
        random_split,
        rmsd_scorer,
        binary_evaluator,
    ):
        ev = Evaluator(
            dataset_split=random_split,
            scorer=rmsd_scorer,
            evaluator=binary_evaluator,
            n_bootstraps=1000,
        )
        data_list = ev.run_dataset_split(docking_data_model)
        dfs = [data.dataframe[data.dataframe["Pose_ID"] == 0] for data in data_list]
        lig_total = len(ligs)
        fractions = []
        for df in dfs:
            filtered = df.sort_values("RMSD").groupby("Query_Ligand").head(1)
            total = len(filtered)
            assert lig_total == total
            fraction = filtered["RMSD"].apply(lambda x: x <= 2).sum() / total
            fractions.append(SuccessRate(total=lig_total, fraction=fraction))
        manual = SuccessRate.from_replicates(fractions)

        success_rate = ev.run(docking_data_model)
        assert isinstance(success_rate, SuccessRate)
        assert np.isclose(success_rate.fraction, manual.fraction, 0.01)

    @pytest.mark.parametrize("dataset_split", ["random", "date_split"])
    @pytest.mark.parametrize("scorer", ["rmsd", "posit"])
    def test_evaluator_performance(
        self,
        docking_data_model,
        refs,
        ligs,
        dataset_split,
        random_split,
        date_split,
        scorer,
        rmsd_scorer,
        posit_scorer,
        binary_evaluator,
    ):
        ev = Evaluator(
            dataset_split=random_split if dataset_split == "random" else date_split,
            scorer=rmsd_scorer if scorer == "rmsd" else posit_scorer,
            evaluator=binary_evaluator,
            n_bootstraps=1000,
        )

        import time

        start_time = time.perf_counter()
        results = ev.run(docking_data_model)
        total_time = time.perf_counter() - start_time

        assert total_time < 20

        print(f"Evaluator: {total_time:.3f} seconds")
        print(results)

    @pytest.mark.parametrize("dataset_split", ["random", "date_split"])
    @pytest.mark.parametrize("scorer", ["rmsd", "posit"])
    def test_serialization(
        self,
        docking_data_model,
        refs,
        ligs,
        dataset_split,
        random_split,
        date_split,
        scorer,
        rmsd_scorer,
        posit_scorer,
        binary_evaluator,
    ):
        ev = Evaluator(
            dataset_split=random_split if dataset_split == "random" else date_split,
            scorer=rmsd_scorer if scorer == "rmsd" else posit_scorer,
            evaluator=binary_evaluator,
            n_bootstraps=1000,
        )
        ev.to_json_file("test.json")
        ev2 = Evaluator.from_json_file("test.json")
        assert ev == ev2


def test_eval_on_local():
    data = DockingDataModel.deserialize(
        "/Users/alexpayne/Scientific_Projects/mers-drug-discovery/sars2-retrospective-analysis/ALL_combined_results.parquet"
    )
    evf = EvaluatorFactory(name="test")
    evf.reference_split_settings.use = True
    evf.reference_split_settings.date_split_settings.use = True
    evf.reference_split_settings.date_split_settings.reference_structure_date_column = (
        "Reference_Structure_Date"
    )
    evf.reference_split_settings.random_split_settings.use = True
    evf.scorer_settings.rmsd_scorer_settings.use = True
    evf.scorer_settings.posit_scorer_settings.use = True

    evf.reference_split_settings.update_reference_settings.use = True
    evf.reference_split_settings.update_reference_settings.use_logarithmic_scaling = (
        True
    )
    evs = evf.create_evaluators(data)

    results = Results.calculate_results(data, evs[0:2])

    df = Results.df_from_results(results)
    assert len(df) == 2


class TestSettings:

    def test_similarity_settings_roundtrip(self, tmpdir):
        from harbor.analysis.cross_docking import SimilaritySplitSettings

        default = SimilaritySplitSettings()
        print(default)
        fp = default.to_yaml_file(tmpdir / "settings.yaml")
        loaded = SimilaritySplitSettings.from_yaml_file(fp)

        assert default == loaded

    def test_evaluator_factory_roundtrip(self, tmpdir):
        evf = EvaluatorFactory(name="test")
        fp = evf.to_yaml_file(tmpdir)

        loaded = EvaluatorFactory.from_yaml_file(fp)
        fp2 = loaded.to_yaml_file()
        loaded_again = EvaluatorFactory.from_yaml_file(fp2)
        assert evf == loaded == loaded_again

        evs = loaded.create_evaluators()
        assert len(evs) == 2

    def test_reference_split_with_date_and_random(self, docking_data_model, tmpdir):
        evf = EvaluatorFactory(name="test")
        evf.reference_split_settings.use = True
        evf.reference_split_settings.date_split_settings.use = True
        evf.reference_split_settings.date_split_settings.reference_structure_date_column = (
            "Date"
        )
        evf.reference_split_settings.random_split_settings.use = True
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 4

    def test_pairwise_split_with_scaffold_settings(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_id_column = (
            "Query_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_id_column = (
            "Ref_Scaffold"
        )
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 10

    def test_scaffold_split_x_to_y(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_id_column = (
            "Query_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_id_column = (
            "Ref_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.scaffold_split_option = (
            ScaffoldSplitOptions.X_TO_Y
        )
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 40

    def test_scaffold_split_x_to_not_x(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_id_column = (
            "Query_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_id_column = (
            "Ref_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.scaffold_split_option = (
            ScaffoldSplitOptions.X_TO_NOT_X
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_min_count = (
            1
        )
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_min_count = 1
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 10

    def test_scaffold_split_not_x_to_x(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.use = True
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_id_column = (
            "Query_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_id_column = (
            "Ref_Scaffold"
        )
        evf.pairwise_split_settings.scaffold_split_settings.scaffold_split_option = (
            ScaffoldSplitOptions.NOT_X_TO_X
        )
        evf.pairwise_split_settings.scaffold_split_settings.reference_scaffold_min_count = (
            1
        )
        evf.pairwise_split_settings.scaffold_split_settings.query_scaffold_min_count = 1
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 10

    def test_similarity_split_with_logarithmic_scaling(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.similarity_split_settings.use = True
        evf.pairwise_split_settings.similarity_split_settings.include_similar = False
        evf.pairwise_split_settings.similarity_split_settings.similarity_groupby_dict = {
            "Aligned": True
        }
        evf.pairwise_split_settings.similarity_split_settings.update_reference_settings.use = (
            True
        )
        evf.pairwise_split_settings.similarity_split_settings.update_reference_settings.use_logarithmic_scaling = (
            True
        )
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 462

    def test_similarity_split_with_fixed_references(self, docking_data_model):
        evf = EvaluatorFactory(name="test")
        evf.pairwise_split_settings.use = True
        evf.pairwise_split_settings.similarity_split_settings.use = True
        evf.pairwise_split_settings.similarity_split_settings.include_similar = False
        evf.pairwise_split_settings.similarity_split_settings.similarity_groupby_dict = {
            "Aligned": True
        }
        evf.pairwise_split_settings.similarity_split_settings.n_reference_structures = [
            10
        ]
        evs = evf.create_evaluators(docking_data_model)
        assert len(evs) == 42
        results = Results.df_from_results(
            Results.calculate_results(evaluators=evs[-2:], data=docking_data_model)
        )
        results.to_csv("test.csv")
        assert len(results) == 2


class TestResults:
    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        return [
            Results(
                evaluator=Evaluator(
                    dataset_split=RandomSplit(
                        reference_structure_column="Reference_Structure",
                        n_reference_structures=5,
                    ),
                    scorer=RMSDScorer(),
                    evaluator=BinaryEvaluation(variable="RMSD", cutoff=2.0),
                    n_bootstraps=100,
                ),
                success_rate=SuccessRate(
                    total=100, fraction=0.5, replicates=[0.4, 0.5, 0.6]
                ),
            ),
            Results(
                evaluator=Evaluator(
                    dataset_split=RandomSplit(
                        reference_structure_column="Reference_Structure",
                        n_reference_structures=10,
                    ),
                    scorer=RMSDScorer(),
                    evaluator=BinaryEvaluation(variable="RMSD", cutoff=2.0),
                    n_bootstraps=100,
                ),
                success_rate=SuccessRate(
                    total=100, fraction=0.6, replicates=[0.5, 0.6, 0.7]
                ),
            ),
            Results(
                evaluator=Evaluator(
                    similarity_split=ScaffoldSplit(
                        query_scaffold_id_column="Query_Scaffold",
                        reference_scaffold_id_column="Ref_Scaffold",
                        split_option=ScaffoldSplitOptions.X_TO_X,
                        query_scaffold_id_subset=["Scaffold_1"],
                        reference_scaffold_id_subset=["Scaffold_1"],
                    ),
                    scorer=RMSDScorer(),
                    evaluator=BinaryEvaluation(variable="RMSD", cutoff=2.0),
                    n_bootstraps=100,
                ),
                success_rate=SuccessRate(
                    total=100, fraction=0.6, replicates=[0.5, 0.6, 0.7]
                ),
            ),
        ]

    def test_df_from_results(self, sample_results):
        """Test converting results to DataFrame."""
        df = Results.df_from_results(sample_results)
        print(df.Evaluator_Model.unique()[0])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert all(
            [
                col in df.columns
                for col in [
                    "Bootstraps",
                    "Score",
                    "Score_Choose_N",
                    "EvaluationMetric",
                    "EvaluationMetric_Cutoff",
                    "PoseSelection",
                    "PoseSelection_Choose_N",
                    "Reference_Split",
                    "N_Reference_Structures",
                    "Reference_Structure_Column",
                    "Min",
                    "Max",
                    "CI_Upper",
                    "CI_Lower",
                    "Total",
                    "Fraction",
                    "PairwiseSplit",
                    "Query_Scaffold_ID_Column",
                    "Reference_Scaffold_ID_Column",
                    "Scaffold_Split_Option",
                    "Query_Scaffold_ID_Subset",
                    "Reference_Scaffold_ID_Subset",
                    "Evaluator_Model",
                ]
            ]
        )

    def test_calculate_results(self, docking_data_model):
        """Test calculating results from evaluators."""
        evaluators = [
            Evaluator(
                dataset_split=RandomSplit(
                    reference_structure_column="Reference_Structure",
                    n_reference_structures=5,
                ),
                scorer=RMSDScorer(),
                evaluator=BinaryEvaluation(variable="RMSD", cutoff=2.0),
                n_bootstraps=100,
            )
        ]

        results = [
            results
            for results in Results.calculate_results(docking_data_model, evaluators)
        ]
        df = Results.df_from_results(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(evaluators)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Results)


def test_success_rate():
    fg = SuccessRate(total=100, fraction=0.5, replicates=[0.5, 0.6, 0.7])
    assert fg.get_records() == {
        "Min": 0.5,
        "Max": 0.7,
        "CI_Upper": 0.7,
        "CI_Lower": 0.5,
        "Total": 100,
        "Fraction": 0.5,
    }

    with pytest.raises(ValueError):
        SuccessRate(total=100, fraction=1.1, replicates=[0.5, 0.6, 0.7, 0.8])


class TestGetUniqueStructuresRandomizedByDate:
    @staticmethod
    def create_sample_dataframe():
        """Helper method to create a sample DataFrame for testing."""
        data = {
            "structure": ["A", "B", "C", "D", "E"],
            "date": [
                "2023-01-01 00:00:00",
                "2023-01-02 00:00:00",
                "2023-01-03 00:00:00",
                "2023-01-04 00:00:00",
                "2023-01-05 00:00:00",
            ],
        }
        return pd.DataFrame(data)

    def test_valid_input(self):
        """Test valid input with 3 structures returned."""
        df = self.create_sample_dataframe()
        result = get_unique_structures_randomized_by_date(
            df,
            structure_column="structure",
            date_column="date",
            n_days_to_randomize=2,
            n_structures_to_return=3,
        )
        assert len(result) == 1  # One bootstrap by default
        assert len(result[0]) == 3  # 3 structures returned
        assert all(structure in df["structure"].values for structure in result[0])

    def test_requesting_more_structures_than_available(self):
        """Test requesting more structures than available."""
        df = self.create_sample_dataframe()
        with pytest.raises(ValueError):
            get_unique_structures_randomized_by_date(
                df,
                structure_column="structure",
                date_column="date",
                n_structures_to_return=10,
                n_days_to_randomize=2,
            )

    def test_randomization_within_date_range(self):
        """Test randomization within a specific date range."""
        df = self.create_sample_dataframe()
        result = get_unique_structures_randomized_by_date(
            df,
            structure_column="structure",
            date_column="date",
            n_structures_to_return=2,
            n_days_to_randomize=1,
        )
        assert len(result) == 1
        assert len(result[0]) == 2
        selected_dates = df[df["structure"].isin(result[0])]["date"]
        selected_dates = pd.to_datetime(selected_dates)
        max_date = selected_dates.max()
        min_date = selected_dates.min()
        assert (max_date - min_date).days <= 1

    def test_bootstrapping(self):
        """Test bootstrapping functionality."""
        df = self.create_sample_dataframe()
        result = get_unique_structures_randomized_by_date(
            df,
            structure_column="structure",
            date_column="date",
            n_structures_to_return=2,
            n_days_to_randomize=2,
            bootstraps=3,
        )
        assert len(result) == 3  # 3 bootstraps
        for bootstrap in result:
            assert len(bootstrap) == 2
            assert all(structure in df["structure"].values for structure in bootstrap)

    def test_empty_dataframe(self):
        """Test behavior with an empty DataFrame."""
        empty_df = pd.DataFrame(columns=["structure", "date"])
        with pytest.raises(ValueError):
            get_unique_structures_randomized_by_date(
                empty_df,
                structure_column="structure",
                date_column="date",
                n_structures_to_return=1,
                n_days_to_randomize=2,
            )
