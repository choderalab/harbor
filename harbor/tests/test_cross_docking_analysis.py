import pytest

from harbor.analysis.cross_docking import (
    Evaluator,
    PoseSelector,
    RandomSplit,
    Scorer,
    BinaryEvaluation,
    FractionGood,
)


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


def test_serialization():
    ev = Evaluator(
        pose_selector=PoseSelector(
            name="Default", variable="Pose_ID", number_to_return=1
        ),
        dataset_split=RandomSplit(variable="help", n_per_split=100),
        scorer=Scorer(
            name="RMSD", variable="RMSD", higher_is_better=False, number_to_return=1
        ),
        evaluator=BinaryEvaluation(variable="RMSD", cutoff=2),
        groupby=["Query_Ligand"],
    )
    ev.to_json_file("test.json")
    ev2 = Evaluator.from_json_file("test.json")
    assert ev == ev2
