import logging

from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum, auto
from typing_extensions import Self
import abc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta
import json
import yaml


class ModelBase(BaseModel):
    type_: str = Field(..., description="Type of model")

    @abc.abstractmethod
    def plot_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_records(self) -> dict:
        pass


class SettingsBase(BaseModel):
    def get_descriptions(self) -> dict:
        schema = self.model_json_schema()
        return {
            field: field_info.get("description", "")
            for field, field_info in schema["properties"].items()
        }

    def to_yaml(self):
        # Get the model's JSON schema
        return json.loads(self.model_dump_json())

    def to_yaml_file(self, file_path):
        # Convert to YAML
        output = self.to_yaml()
        descriptions = self.get_descriptions()

        # Write to file with descriptions as a block comment at the top
        with open(file_path, "w") as file:
            for key, value in output.items():
                if key in descriptions:
                    file.write(f"# {key}: {descriptions[key]}\n")

            # then write out full object
            yaml.dump(output, file, sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_str):
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, file_path):
        with open(file_path, "r") as file:
            return cls.from_yaml(file.read())


class SplitBase(ModelBase):
    """
    Base class for splitting the data (i.e. Random, Dataset, Scaffold, etc)
    """

    name: str = "SplitBase"
    type_: str = "SplitBase"
    n_splits: int = Field(1, description="number of splits to generate")
    n_per_split: int = Field(..., description="Number of values per split to generate")
    deterministic: bool = Field(
        False,
        description="Whether the split is deterministic, i.e. if True it should not be run in the bootstrapping loop.",
    )
    split_level: int = Field(
        0,
        description="Level of the split, 0 indexed. The first level is applied first, and so on.",
    )

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        pass

    @property
    def plot_name(self) -> str:
        return f"{self.name}_{self.n_per_split}"

    def get_records(self) -> dict:
        if self.split_level == 0:
            return self._get_records()
        else:
            return {
                f"{k}_{self.split_level}": v for k, v in self._get_records().items()
            }

    @abc.abstractmethod
    def _get_records(self) -> dict:
        pass


class ReferenceStructureSplitBase(SplitBase):
    """
    Base class for splitting the data based on some attributes of the reference structure
    """

    reference_structure_column: str = Field(
        ..., description="Name of the column to distinguish reference structures by"
    )

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        pass

    def _get_records(self) -> dict:
        return {
            "Split": self.name,
            "N_Per_Split": self.n_per_split,
            "Reference_Structure_Column": self.reference_structure_column,
        }


class RandomSplit(ReferenceStructureSplitBase):
    """
    Randomly split the structures into n_splits
    """

    name: str = "RandomSplit"
    type_: str = "RandomSplit"

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        from random import shuffle

        variable_list = df[self.reference_structure_column].unique()
        shuffle(variable_list)

        variable_splits = []
        dfs = []
        for i in range(self.n_splits):
            start = i * self.n_per_split
            end = i * self.n_per_split + self.n_per_split
            variable_splits.append(variable_list[start:end])
            dfs.append(
                df[df[self.reference_structure_column].isin(variable_list[start:end])]
            )
        return dfs


## write a function that takes a structure and returns a random structure within timedelta from it
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


class DateSplit(ReferenceStructureSplitBase):
    """
    Splits the data by date.
    """

    name: str = "DateSplit"
    type_: str = "DateSplit"
    date_dict: dict = Field(
        ...,
        description="Dictionary of dates to split the data by of the form dict[str, str] where the key is the structure name and the value is the date",
    )
    balanced: bool = Field(
        True,
        description="Whether to split the data uniformly in time (i.e. 1 split every N months) or balanced such that each split has the same number of structures",
    )
    initial_structure_error: int = Field(
        1,
        description="Initial error in the structure date. Error of 1 means no error (pick the first structure every time). Error of 20 means when you are picking up to the first 20 structures, randomize them.",
    )
    randomize_by_n_days: int = Field(
        0,
        description="Randomize the structures by n days. If 0 no randomization is done. If 1 or greater, for each structure, it can be randomly replaced by any other structure collected on that day or n-1 days from it's collection date.",
    )

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        # sort the structures by date
        dates = np.array(list(self.date_dict.values()))
        structures = np.array(list(self.date_dict.keys()))
        sort_idx = np.argsort(dates)
        structure_list = structures[sort_idx]
        variable_splits = []
        dfs = []
        for i in range(self.n_splits):
            start = i * self.n_per_split
            end = i * self.n_per_split + self.n_per_split

            if self.n_per_split < self.initial_structure_error:
                variable_split = np.random.choice(
                    structure_list[start : self.initial_structure_error],
                    self.n_per_split,
                    replace=False,
                )

            elif self.randomize_by_n_days > 0:
                # I think this is probably super slow!
                variable_split = structure_list[start:end]
                variable_split = [
                    get_random_structure(
                        structure,
                        self.date_dict,
                        timedelta=timedelta(days=self.randomize_by_n_days),
                    )
                    for structure in variable_split
                ]
            else:
                variable_split = structure_list[start:end]
            variable_splits.append(variable_split)
            dfs.append(df[df[self.reference_structure_column].isin(variable_split)])
        return dfs


class SimilaritySplit(SplitBase):
    """
    Splits the structures available to dock to by similarity to the query ligand
    """

    name: str = "SimilaritySplit"
    type_: str = "SimilaritySplit"
    similarity_column: str = Field(
        ...,
        description="Column name for the similarity between the query and reference ligands",
    )
    groupby: dict = Field(
        ...,
        description="Column name : value pairs to group the Tanimoto similarity data by.",
    )
    query_ligand_column: str = Field(
        ...,
        description="Column name for the query ligand ID in order to pick the top N structures to dock to",
    )
    threshold: float = Field(
        0.5,
        description="Threshold to use to determine if two structures are similar enough to be in the same split",
    )
    higher_is_more_similar: bool = Field(
        True, description="Higher values are more similar"
    )
    include_similar: bool = Field(
        True,
        description="If True, include structures that are more similar than the threshold. Otherwise, include structures that are less similar.",
    )
    deterministic: bool = True

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        if self.n_per_split != -1:
            self.deterministic = False
        return self

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:

        # first just get the necessary data
        for key, value in self.groupby.items():
            df = df[df[key] == value]

        # if include similar True and higher is MORE similar, or if similar False and higher is LESS similar
        if self.include_similar == self.higher_is_more_similar:
            df = df[df[self.similarity_column] >= self.threshold]

        # if include similar True and higher is LESS similar, or if similar False and higher is MORE similar
        elif self.include_similar != self.higher_is_more_similar:
            df = df[df[self.similarity_column] <= self.threshold]

        # finally, group by the query ligand column and randomly sample from the top N structures
        df = df.groupby(self.query_ligand_column).head(self.n_per_split)
        return [df]

    def _get_records(self) -> dict:
        return_dict = {
            "Split": self.name,
            "N_Per_Split": self.n_per_split,
            "Split_Variable": self.similarity_column,
            "Similarity_Threshold": self.threshold,
            "Include_Similar": self.include_similar,
            "Higher_Is_More_Similar": self.higher_is_more_similar,
        }
        return_dict.update({key: value for key, value in self.groupby.items()})
        return return_dict


class ScaffoldSplitOptions(Enum):
    """
    Options for how to split the structures by scaffold.
    If a datasets has scaffolds A-F,
    there are basically four comparisons that are interesting.
    The first three are easy to do in parallel in my setup,
    the last one requires that you set up separate evaluators for each combination of scaffolds.

    """

    X_TO_X = "x_to_x"  # , "Dock X to X for X in [all your scaffolds]")
    X_TO_NOT_X = "x_to_not_x"  # , "Dock X to NOT X for X in [all your scaffolds]")
    NOT_X_TO_X = "not_x_to_x"  # , "Dock NOT X to X for X in [all your scaffolds]")
    X_TO_Y = "x_to_y"  # ,"Dock X to Y for X, Y in zip([all your scaffolds], [all your scaffolds]",)
    X_TO_ALL = "x_to_all"  # Dock X to all data for X in [all your scaffolds]
    ALL_TO_X = "all_to_x"  # Dock all to X for X in [all your scaffolds]


class ScaffoldSplit(SplitBase):
    """
    Splits the structures available to dock to by whether they share a scaffold with the query ligand.
    """

    name: str = "ScaffoldSplit"
    type_: str = "ScaffoldSplit"
    query_scaffold_id_column: str = Field(
        ..., description="Column name for the query scaffold ID"
    )
    reference_scaffold_id_column: str = Field(
        ..., description="Column name for the reference scaffold ID"
    )
    query_scaffold_id_subset: Optional[list[int]] = Field(
        None,
        description="List of query scaffold IDs to consider. If None, consider all scaffolds.",
    )
    reference_scaffold_id_subset: Optional[list[int]] = Field(
        None,
        description="List of reference scaffold IDs to consider. If None, consider all scaffolds.",
    )
    split_option: ScaffoldSplitOptions = Field(
        ScaffoldSplitOptions.X_TO_Y,
        description="How to split the data by scaffold",
    )
    deterministic: bool = True

    @field_validator("split_option", mode="before")
    def convert_to_string(cls, v):
        if isinstance(v, Enum):
            return v.value
        return v

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        option = self.split_option

        if (
            option == ScaffoldSplitOptions.NOT_X_TO_X
            or option == ScaffoldSplitOptions.ALL_TO_X
        ):
            if (
                not self.reference_scaffold_id_subset
                or len(self.reference_scaffold_id_subset) != 1
            ):
                raise ValueError(
                    f"{option} requires exactly one item in reference_scaffold_id_subset"
                )

        elif (
            option == ScaffoldSplitOptions.X_TO_NOT_X
            or option == ScaffoldSplitOptions.X_TO_ALL
        ):
            if (
                not self.query_scaffold_id_subset
                or len(self.query_scaffold_id_subset) != 1
            ):
                raise ValueError(
                    f"{option} requires exactly one item in query_scaffold_id_subset"
                )

        elif option == ScaffoldSplitOptions.X_TO_Y:
            if (
                not self.query_scaffold_id_subset
                or not self.reference_scaffold_id_subset
                or len(self.query_scaffold_id_subset) != 1
                or len(self.reference_scaffold_id_subset) != 1
            ):
                raise ValueError(
                    f"{option} requires exactly one item in both query_ and reference_scaffold_id_subset"
                )

        # if both subsets are length 1 and are the same,
        # and the split option is not X_TO_X, X_TO_ALL, or ALL_TO_X, there won't be any data to analyze
        split_options_that_can_have_the_same_query_and_reference = [
            ScaffoldSplitOptions.X_TO_X,
            ScaffoldSplitOptions.X_TO_Y,
            ScaffoldSplitOptions.X_TO_ALL,
            ScaffoldSplitOptions.ALL_TO_X,
        ]
        if (
            not self.query_scaffold_id_subset is None
            and len(self.query_scaffold_id_subset) == 1
            and self.query_scaffold_id_subset == self.reference_scaffold_id_subset
            and not self.split_option
            in split_options_that_can_have_the_same_query_and_reference
        ):
            raise Warning(
                f"Query and reference scaffold IDs are the same ({self.query_scaffold_id_subset[0]}), "
                f"and there are only one of each, "
                f"but you haven't picked one of these split options: {split_options_that_can_have_the_same_query_and_reference}. "
                f"This means there's no data to analyze."
            )

        return self

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:

        dfs = []

        # first, filter by the query scaffold ID
        if self.query_scaffold_id_subset is not None:
            df = df[
                df[self.query_scaffold_id_column].isin(self.query_scaffold_id_subset)
            ]

        # then, filter by the reference scaffold ID
        if self.reference_scaffold_id_subset is not None:
            df = df[
                df[self.reference_scaffold_id_column].isin(
                    self.reference_scaffold_id_subset
                )
            ]

        split_option = self.split_option
        if split_option == ScaffoldSplitOptions.X_TO_X:
            df = df[
                df[self.query_scaffold_id_column]
                == df[self.reference_scaffold_id_column]
            ]
            dfs.append(df)

        elif split_option in [
            ScaffoldSplitOptions.NOT_X_TO_X,
            ScaffoldSplitOptions.X_TO_NOT_X,
        ]:
            df = df[
                df[self.query_scaffold_id_column]
                != df[self.reference_scaffold_id_column]
            ]
            dfs.append(df)

        elif split_option in [
            ScaffoldSplitOptions.X_TO_Y,
            ScaffoldSplitOptions.ALL_TO_X,
            ScaffoldSplitOptions.X_TO_ALL,
        ]:
            # we already did the necessary work up top!
            dfs.append(df)
        else:
            raise NotImplementedError(f"Split option {split_option} not implemented")

        return dfs

    def _get_records(self) -> dict:
        return_dict = {
            "Split": self.name,
            "N_Per_Split": self.n_per_split,
            "Query_Scaffold_ID_Column": self.query_scaffold_id_column,
            "Reference_Scaffold_ID_Column": self.reference_scaffold_id_column,
            "Split_Option": self.split_option,
            "Query_Scaffold_ID_Subset": self.query_scaffold_id_subset,
            "Reference_Scaffold_ID_Subset": self.reference_scaffold_id_subset,
        }
        return return_dict


# TODO: There might be a better way to do this.
DatasetSplitType = RandomSplit | DateSplit | SimilaritySplit | ScaffoldSplit
CoreSplit = RandomSplit | DateSplit
ChemicalSplit = SimilaritySplit | ScaffoldSplit


class SorterBase(ModelBase):
    type_: str = "SorterBase"
    name: str = Field(..., description="Name of sorting method")
    category: str = Field(
        ..., description="Category of sort (i.e. why is sorting necessary here"
    )
    variable: str = Field(..., description="Variable used to sort the data")
    higher_is_better: bool = Field(
        True, description="Higher values are better. Defaults True"
    )
    number_to_return: Optional[int] = Field(
        None, description="Number of values to return. Returns all values if None."
    )

    @field_validator("number_to_return", mode="before")
    def allow_number_to_return_to_be_none(cls, v):
        if v is None:
            return None
        return v

    def run(self, df, groupby: list[str]) -> pd.DataFrame:
        return (
            df.sort_values(self.variable, ascending=not self.higher_is_better)
            .groupby(groupby)
            .head(self.number_to_return)
        )

    @property
    def plot_name(self) -> str:
        return f"{self.name}_Choose_{'All' if not self.number_to_return else self.number_to_return}"

    def get_records(self) -> dict:
        return {
            self.category: self.name,
            f"{self.category}_Choose_N": (
                "All" if not self.number_to_return else self.number_to_return
            ),
        }


class StructureChoice(SorterBase):
    type_: str = "StructureChoice"
    category: str = "StructureChoice"


class Scorer(SorterBase):
    category: str = "Score"
    type_: str = "Scorer"


class POSITScorer(Scorer):
    name: str = "POSIT_Probability"
    variable: str = "docking-confidence-POSIT"
    higher_is_better: bool = True
    number_to_return: int = 1


class RMSDScorer(Scorer):
    name: str = "RMSD"
    variable: str = "RMSD"
    higher_is_better: bool = False
    number_to_return: int = 1


class PoseSelector(SorterBase):
    type_: str = "PoseSelector"
    category: str = "PoseSelection"
    groupby: list[str] = ["Query_Ligand", "Reference_Ligand"]
    higher_is_better: bool = False

    def run(self, df: pd.DataFrame, groupby=None) -> pd.DataFrame:
        return SorterBase.run(self, df, groupby=self.groupby)


class FractionGood(ModelBase):
    from pydantic import confloat

    name: str = "FractionGood"
    type_: str = "FractionGood"
    total: int = Field(..., description="Total number of items being evaluated")
    fraction: confloat(ge=0, le=1) = Field(
        ..., description='Fraction of "good" values returned'
    )
    replicates: list[float] = Field(
        [], description='List of "good" fractions for error bar analysis'
    )

    @property
    def min(self) -> float:
        return np.array(self.replicates).min()

    @property
    def max(self) -> float:
        return np.array(self.replicates).max()

    @property
    def ci_upper(self):
        n_reps = len(self.replicates)
        if n_reps == 1:
            # use beta function to get CIs
            from scipy.stats import beta

            n_successes = self.fraction * self.total
            n_failures = (1 - self.fraction) * self.total

            # this is the posterior probability of observing n_successes and n_failures
            ci_upper = beta(n_successes + 1, n_failures + 1).interval(0.95)[1]

        else:
            # otherwise used bootstrapped results
            self.replicates.sort()
            ci_upper = self.replicates[int(0.975 * n_reps)]
        return ci_upper

    @property
    def ci_lower(self):
        n_reps = len(self.replicates)
        if n_reps == 1:
            # use beta function to get CIs
            from scipy.stats import beta

            ci_lower = beta(
                self.fraction * self.total + 1, (1 - self.fraction) * self.total + 1
            ).interval(0.95)[0]

        else:
            # otherwise used bootstrapped results
            self.replicates.sort()
            ci_lower = self.replicates[int(0.025 * n_reps)]
        return ci_lower

    @classmethod
    def from_replicates(cls, reps: list["FractionGood"]) -> "FractionGood":
        all_fracs = np.array([rep.fraction for rep in reps])
        totals = np.array([rep.total for rep in reps])
        return FractionGood(
            total=totals.max(), fraction=all_fracs.mean(), replicates=list(all_fracs)
        )

    def get_records(self) -> dict:
        mydict = {
            "Min": self.min,
            "Max": self.max,
            "CI_Upper": self.ci_upper,
            "CI_Lower": self.ci_lower,
            "Total": self.total,
            "Fraction": self.fraction,
        }
        return mydict

    def plot_name(self) -> str:
        return "Fraction"


class BinaryEvaluation(ModelBase):
    name: str = "BinaryEvaluation"
    type_: str = "BinaryEvaluation"
    variable: str = Field(..., description="Variable used to evaluate the results")
    cutoff: float = Field(
        ..., description="Cutoff used to determine if a result is good"
    )
    below_cutoff_is_good: bool = Field(
        True,
        description="Whether values below or above the cutoff are good. Defaults to below.",
    )

    def run(self, df, groupby: list[str] = ()) -> FractionGood:
        total = len(df.groupby(groupby))
        if total == 0:
            return FractionGood(total=0, fraction=0)
        if self.below_cutoff_is_good:
            fraction = df[self.variable].apply(lambda x: x <= self.cutoff).sum() / total
        else:
            fraction = df[self.variable].apply(lambda x: x >= self.cutoff).sum() / total
        return FractionGood(total=total, fraction=fraction)

    def get_records(self) -> dict:
        return {
            "EvaluationMetric": self.variable,
            "EvaluationMetric_Cutoff": self.cutoff,
        }

    def plot_name(self) -> str:
        return "_".join([self.name, self.variable, self.cutoff])


def get_class_from_name(name: str):
    """
    Is this good? Is it safe? Is it smart? I don't know!
    :param name:
    :return:
    """
    match name:
        case "RandomSplit":
            return RandomSplit
        case "DateSplit":
            return DateSplit
        case "SimilaritySplit":
            return SimilaritySplit
        case "ScaffoldSplit":
            return ScaffoldSplit
        case "StructureChoice":
            return StructureChoice
        case "Scorer":
            return Scorer
        case "PoseSelector":
            return PoseSelector
        case "FractionGood":
            return FractionGood
        case "BinaryEvaluation":
            return BinaryEvaluation
        case "Evaluator":
            return Evaluator


class Evaluator(ModelBase):
    name: str = "Evaluator"
    type_: str = "Evaluator"
    pose_selector: PoseSelector = Field(
        PoseSelector(name="Default", variable="Pose_ID", number_to_return=1),
        description="How to choose which poses to keep",
    )
    dataset_split: DatasetSplitType = Field(..., description="Dataset split")
    extra_splits: Optional[list[DatasetSplitType]] = Field(
        None, description="Additional dataset splits to be run after the first one"
    )
    structure_choice: StructureChoice = Field(
        StructureChoice(name="Dock_to_All", variable="Tanimoto", higher_is_better=True),
        description="How to choose which structures to dock to",
    )
    scorer: Scorer = Field(..., description="How to score and rank resulting poses")
    evaluator: BinaryEvaluation = Field(
        ..., description="How to determine how good the results are"
    )
    n_bootstraps: int = Field(1, description="Number of bootstrap replicates to run")
    groupby: list[str] = Field(..., description="List of variables that group the data")

    def run(self, df: pd.DataFrame) -> FractionGood:
        df = self.pose_selector.run(df)
        results = []
        for i in range(self.n_bootstraps):
            split1 = self.dataset_split.run(df)[0]
            if self.extra_splits:
                for split in self.extra_splits:
                    if split is not None:
                        split1 = split.run(split1)[0]
            subset_df = self.structure_choice.run(split1, groupby=self.groupby)
            subset_df = self.scorer.run(subset_df, groupby=self.groupby)
            results.append(self.evaluator.run(subset_df, groupby=self.groupby))
            if self.dataset_split.deterministic:
                # no need to run more than once because nothing will change!
                break
        return FractionGood.from_replicates(results)

    @field_validator(
        "pose_selector",
        "dataset_split",
        "structure_choice",
        "scorer",
        "evaluator",
        mode="before",
    )
    def class_from_dict(cls, v):
        if isinstance(v, dict):
            return get_class_from_name(v["type_"])(**v)
        else:
            return v

    @model_validator(mode="after")
    def update_split_level(self) -> Self:
        self.dataset_split.split_level = 0
        if self.extra_splits:
            for level, split in enumerate(self.extra_splits, start=1):
                if split is not None:
                    split.split_level = level
        return self

    def to_json_file(self, file_path: str | Path) -> Path:
        import json

        with open(file_path, "w") as f:
            f.write(self.model_dump_json())
        return file_path

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "Evaluator":
        import json

        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @property
    def plot_name(self) -> str:
        variables = [
            model.plot_name
            for model in [self.dataset_split, self.structure_choice, self.scorer]
        ]
        variables += [f"{self.n_bootstraps}reps"]
        return "_".join(variables)

    def get_records(self) -> dict:
        mydict = {"Bootstraps": self.n_bootstraps}
        for container in [
            self.structure_choice,
            self.scorer,
            self.evaluator,
            self.dataset_split,
            self.pose_selector,
        ]:
            mydict.update(container.get_records())
        if self.extra_splits:
            for split in self.extra_splits:
                mydict.update(split.get_records())
        return mydict


class Results(BaseModel):
    evaluator: Evaluator
    fraction_good: FractionGood

    def get_records(self) -> dict:
        mydict = self.evaluator.get_records()
        mydict.update(self.fraction_good.get_records())
        return mydict

    @classmethod
    def calculate_result(cls, evaluator: Evaluator, df: pd.DataFrame) -> "Results":
        result = evaluator.run(df)
        return cls(evaluator=evaluator, fraction_good=result)

    @classmethod
    def calculate_results(
        cls, df: pd.DataFrame, evaluators: list["Evaluator"]
    ) -> list["Results"]:
        for ev in tqdm(evaluators):
            result = ev.run(df)
            yield cls(evaluator=ev, fraction_good=result)

    @classmethod
    def df_from_results(cls, results: list["Results"]) -> pd.DataFrame:
        return pd.DataFrame.from_records([result.get_records() for result in results])


class Settings(SettingsBase):
    # General Settings
    n_bootstraps: int = Field(1000, description="Number of bootstrapped samples to run")
    query_ligand_column: str = Field(
        "Query_Ligand", description="Name of the column containing the query ligand id"
    )
    reference_ligand_column: str = Field(
        "Reference_Ligand",
        description="Name of the column containing the reference ligand id",
    )
    reference_structure_column: str = Field(
        "Reference_Structure",
        description="Name of the column to distinguish reference structures by",
    )

    # Pose Selection
    pose_id_column: str = Field(
        "Pose_ID", description="Name of the column containing the pose id"
    )
    n_poses: list[int] = Field(None, description="Number of poses to select")

    # Dataset Splitting
    use_random_split: bool = True
    update_n_per_split: bool = Field(
        True, description="Update n_per_split based on data"
    )
    n_per_split: Optional[list[int]] = Field(
        None,
        description="A list of the number of reference structures that will be used in each split",
    )

    # Date Split
    use_date_split: bool = False
    reference_structure_date_column: str = Field(
        "Reference_Structure_Date",
        description="Name of the column where the date of the reference structure deposition is saved",
    )
    randomize_by_n_days: int = 1

    # Similarity splitting options
    use_similarity_split: bool = False
    similarity_column_name: Optional[str] = Field(
        None,
        description="Name of the column containing the similarity value between the query and reference ligands",
    )
    similarity_groupby: dict = {}
    similarity_range: list[float] = Field(
        [0, 1],
        description="Range of values the similarity score can take. Used to generate a standardized set of thresholds",
    )
    similarity_n_thresholds: int = Field(21, description="Number of thresholds to use")

    # Scaffold splitting options
    use_scaffold_split: bool = False
    scaffold_split_option: ScaffoldSplitOptions = Field(
        ScaffoldSplitOptions.X_TO_X,
        description=f"How to split the data by scaffold, one of {ScaffoldSplitOptions}",
    )
    query_scaffold_id_column: str = Field(
        "cluster_id", description="Name of the column containing the query scaffold id"
    )
    reference_scaffold_id_column: str = Field(
        "cluster_id_Reference",
        description="Name of the column containing the reference scaffold id",
    )
    query_scaffold_id_subset: Optional[list[int]] = Field(
        None, description="List of query scaffold IDs to consider"
    )
    reference_scaffold_id_subset: Optional[list[int]] = Field(
        None, description="List of reference scaffold IDs to consider"
    )
    query_scaffold_min_count: Optional[int] = Field(
        5,
        description="Minimum number of ligands in a query scaffold to consider it for docking",
    )
    reference_scaffold_min_count: Optional[int] = Field(
        1,
        description="Minimum number of ligands in a reference scaffold to consider it for docking",
    )

    # Combine Core and Chemical Splits
    combine_core_and_chemical_splits: bool = Field(
        False,
        description=f"Combine {CoreSplit} and {ChemicalSplit} into single splits",
    )

    # Scoring Options
    use_posit_scorer: bool = True
    posit_score_column_name: str = Field(
        "docking-confidence-POSIT",
        description="Name of the column containing the POSIT score",
    )
    posit_name: str = Field("POSIT_Probability", description="Name of the POSIT score")
    use_rmsd_scorer: bool = True
    rmsd_column_name: str = "RMSD"
    rmsd_name: str = Field("RMSD", description="Name of the RMSD score")

    # Evaluator Options
    rmsd_cutoff: float = Field(
        2.0, description="RMSD cutoff to label the resulting poses as successful"
    )

    class Config:
        validate_assignment = True

    @model_validator(mode="after")
    def check_valid_settings(
        self,
    ) -> Self:
        if self.use_similarity_split and self.similarity_column_name is None:
            raise ValueError(
                "Similarity column name must be provided if using similarity split"
            )
        if self.n_per_split is None and not self.update_n_per_split_from_data:
            raise ValueError(
                "n_per_split must be provided if update_n_per_split is False"
            )
        return self

    @field_validator("n_per_split", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, int):
            return [v]
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @property
    def similarity_thresholds(self) -> np.ndarray:
        """
        Generate similarity thresholds from the range and number of thresholds
        :return:
        """
        return np.linspace(
            self.similarity_range[0],
            self.similarity_range[1],
            self.similarity_n_thresholds,
        )

    def update_n_per_split_from_data(
        self, df: pd.DataFrame, initial_range: list = (1, 21), stride=20
    ) -> None:
        n_per_split = np.arange(*initial_range)
        n_per_split = np.concatenate(
            (
                n_per_split,
                np.arange(
                    25,
                    len(df[self.reference_structure_column].unique()) + stride,
                    stride,
                ),
            )
        )
        self.n_per_split = n_per_split

    def create_pose_selectors(self) -> list[PoseSelector]:
        if self.n_poses is None:
            self.n_poses = [1]
        return [
            PoseSelector(
                name="Default", variable=self.pose_id_column, number_to_return=n
            )
            for n in self.n_poses
        ]

    def create_dataset_splits(
        self, df: pd.DataFrame = None, combine_core_and_chemical=False
    ) -> list[DatasetSplitType]:
        if combine_core_and_chemical:
            self.combine_core_and_chemical_splits = True

        dataset_splits = []

        if self.use_random_split:
            dataset_splits.extend(
                [
                    RandomSplit(
                        reference_structure_column=self.reference_ligand_column,
                        n_splits=1,
                        n_per_split=n_per_split,
                    )
                    for n_per_split in self.n_per_split
                ]
            )
        if self.use_date_split:
            if df is None:
                raise ValueError("Must provide input dataframe to use date split")
            date_dict_list = (
                df.groupby(self.reference_structure_column)[
                    [
                        self.reference_structure_column,
                        self.reference_structure_date_column,
                    ]
                ]
                .head(1)
                .to_dict(orient="records")
            )

            simplified_date_dict = {
                date_dict[self.reference_structure_column]: date_dict[
                    self.reference_structure_date_column
                ]
                for date_dict in date_dict_list
            }
            dataset_splits.extend(
                [
                    DateSplit(
                        reference_structure_column=self.reference_structure_column,
                        n_per_split=n_per_split,
                        balanced=True,  # haven't implemented this otherwise
                        date_dict=simplified_date_dict,
                        randomize_by_n_days=self.randomize_by_n_days,
                    )
                    for n_per_split in self.n_per_split
                ]
            )

        if self.use_similarity_split:
            if self.combine_core_and_chemical_splits:
                n_per_splits_to_use = [-1]
            else:
                n_per_splits_to_use = self.n_per_split
            dataset_splits.extend(
                [
                    SimilaritySplit(
                        threshold=threshold,
                        similarity_column=self.similarity_column_name,
                        groupby=self.similarity_groupby,
                        n_per_split=n_per_split,
                        query_ligand_column=self.query_ligand_column,
                        higher_is_more_similar=True,
                        include_similar=False,
                    )
                    for threshold in self.similarity_thresholds
                    for n_per_split in n_per_splits_to_use
                ]
            )
        if self.use_scaffold_split:
            if self.combine_core_and_chemical_splits:
                n_per_splits_to_use = [-1]
            else:
                n_per_splits_to_use = self.n_per_split

            # subset can be a list, and we might want to make a list of subsets

            ref_subset_list = [self.reference_scaffold_id_subset]
            query_subset_list = [self.query_scaffold_id_subset]

            if self.scaffold_split_option in [
                ScaffoldSplitOptions.NOT_X_TO_X,
                ScaffoldSplitOptions.ALL_TO_X,
                ScaffoldSplitOptions.X_TO_Y,
            ]:
                # Get cluster sizes by counting unique ligands per cluster
                cluster_sizes = df.groupby(self.reference_scaffold_id_column)[
                    self.reference_ligand_column
                ].nunique()

                # Filter for clusters with more than 5 members
                ref_subset_list = [
                    [scaffold]
                    for scaffold in cluster_sizes[
                        cluster_sizes > self.reference_scaffold_min_count
                    ].index.tolist()
                ]

            if self.scaffold_split_option in [
                ScaffoldSplitOptions.X_TO_NOT_X,
                ScaffoldSplitOptions.X_TO_ALL,
                ScaffoldSplitOptions.X_TO_Y,
            ]:
                # Do the same thing but for the query
                cluster_sizes = df.groupby(self.query_scaffold_id_column)[
                    self.query_ligand_column
                ].nunique()

                query_subset_list = [
                    [scaffold]
                    for scaffold in cluster_sizes[
                        cluster_sizes > self.query_scaffold_min_count
                    ].index.tolist()
                ]
            if self.scaffold_split_option in [ScaffoldSplitOptions.X_TO_Y]:
                # let's make the subsets the union of both, removing duplicates
                # Convert inner lists to tuples to make them hashable
                set1 = set(tuple(x) for x in ref_subset_list)
                set2 = set(tuple(x) for x in query_subset_list)

                # Create union and convert back to lists
                union = [list(x) for x in set1.union(set2)]
                ref_subset_list = union
                query_subset_list = union

            dataset_splits.extend(
                [
                    ScaffoldSplit(
                        query_scaffold_id_column=self.query_scaffold_id_column,
                        reference_scaffold_id_column=self.reference_scaffold_id_column,
                        split_option=self.scaffold_split_option,
                        reference_scaffold_id_subset=ref_subset,
                        query_scaffold_id_subset=query_subset,
                        n_per_split=n_per_split,
                    )
                    for n_per_split in n_per_splits_to_use
                    for ref_subset in ref_subset_list
                    for query_subset in query_subset_list
                ]
            )
        return dataset_splits

    def combine_splits(
        self, splits: list[DatasetSplitType]
    ) -> (list[CoreSplit], list[ChemicalSplit]):
        """
        Combine multiple splits into one split
        :param splits:
        :return:
        """
        from collections import defaultdict
        from itertools import product

        core_splits = defaultdict(list)
        chemical_splits = defaultdict(list)
        for ds in splits:
            if isinstance(ds, CoreSplit):
                core_splits[ds.name].append(ds)
            elif isinstance(ds, ChemicalSplit):
                chemical_splits[ds.name].append(ds)

        dataset_splits = []
        extra_splits = []
        for split1_name, split1s in core_splits.items():
            for split2_name, split2s in chemical_splits.items():
                # this is combinatorial, so you better be careful!
                for s1, s2 in product(split1s, split2s):
                    dataset_splits.append(s1)
                    extra_splits.append([s2])
        return dataset_splits, extra_splits

    def create_scorers(self) -> list[Scorer]:
        scorers = []
        if self.use_posit_scorer:
            scorers.append(
                Scorer(
                    name=self.posit_name,
                    variable=self.posit_score_column_name,
                    higher_is_better=True,
                    number_to_return=1,
                )
            )
        if self.use_rmsd_scorer:
            scorers.append(
                Scorer(
                    name=self.rmsd_name,
                    variable=self.rmsd_column_name,
                    higher_is_better=False,
                    number_to_return=1,
                )
            )
        return scorers

    def create_success_metric(self) -> BinaryEvaluation:
        return BinaryEvaluation(variable=self.rmsd_column_name, cutoff=self.rmsd_cutoff)

    def create_evaluators(
        self,
        df: pd.DataFrame = None,
        logger: logging.Logger = None,
    ) -> list[Evaluator]:
        if logger is None:
            logger = logging.getLogger(__name__)

        logger.info("Creating evaluators")
        if self.update_n_per_split:
            if df is None:
                raise ValueError("Must provide input dataframe to update n_per_split")
            else:
                self.update_n_per_split_from_data(df)

        logger.info("Creating pose selectors")
        pose_selectors = self.create_pose_selectors()

        logger.info("Creating dataset splits")
        dataset_splits = self.create_dataset_splits(df)
        if self.combine_core_and_chemical_splits:
            dataset_splits, extra_splits = self.combine_splits(dataset_splits)
        else:
            extra_splits = None

        logger.info("Creating scorers")
        scorers = self.create_scorers()
        rmsd_evaluator = self.create_success_metric()

        logger.info("Creating evaluators")
        evaluators = []
        for pose_selector in pose_selectors:
            for i, dataset_split in enumerate(dataset_splits):
                for scorer in scorers:
                    evaluator = Evaluator(
                        pose_selector=pose_selector,
                        dataset_split=dataset_split,
                        extra_splits=extra_splits[i] if extra_splits else None,
                        scorer=scorer,
                        evaluator=rmsd_evaluator,
                        groupby=[self.query_ligand_column],
                        n_bootstraps=self.n_bootstraps,
                    )
                    evaluators.append(evaluator)

        return evaluators
