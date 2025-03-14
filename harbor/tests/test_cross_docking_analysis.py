import pytest

from harbor.analysis.cross_docking import (
    Evaluator,
    PoseSelector,
    DateSplit,
    SimilaritySplit,
    RandomSplit,
    Scorer,
    BinaryEvaluation,
    FractionGood,
    Settings,
)


def test_settings():
    settings = Settings()
    settings.to_yml_file("test.yml")

    import numpy as np

    new_settings = Settings(n_per_split=np.arange(1, 21))
    s2 = Settings.from_yml_file("test.yml")


def test_create_evaluators_from_settings():
    settings = Settings.from_yml_file("test.yml")
    settings.n_per_split = [1]
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
