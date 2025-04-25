import pytest
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import patch

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
    get_unique_structures_randomized_by_date,
)


def test_settings():
    settings = Settings()
    settings.to_yaml_file("test.yaml")

    import numpy as np

    new_settings = Settings(n_per_split=np.arange(1, 21))
    s2 = Settings.from_yaml_file("test.yaml")


def test_create_evaluators_from_settings():
    settings = Settings.from_yaml_file("test.yaml")
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
