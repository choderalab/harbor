from pydantic import BaseModel, Field, field_validator
import abc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional


class ModelBase(BaseModel):
    type_: str = Field(..., description="Type of model")

    @abc.abstractmethod
    def plot_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_records(self) -> dict:
        pass


class SplitBase(ModelBase):
    """
    Base class for splitting the data (i.e. Random, Dataset, Scaffold, etc)
    """

    name: str = "SplitBase"
    type_: str = "SplitBase"
    variable: str = Field(description="Name of variable used to split the data")
    n_splits: int = Field(1, description="number of splits to generate")
    n_per_split: int = Field(..., description="Number of values per split to generate")
    deterministic: bool = Field(
        False,
        description="Whether the split is deterministic, i.e. if True it should not be run in the bootstrapping loop.",
    )

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        pass

    @property
    def plot_name(self) -> str:
        return f"{self.name}_{self.n_per_split}"

    def get_records(self) -> dict:
        return {
            "Split": self.name,
            "N_Per_Split": self.n_per_split,
            "Split_Variable": self.variable,
        }


class RandomSplit(SplitBase):
    """
    Randomly split the structures into n_splits
    """

    name: str = "RandomSplit"
    type_: str = "RandomSplit"

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        from random import shuffle

        variable_list = df[self.variable].unique()
        shuffle(variable_list)

        variable_splits = []
        dfs = []
        for i in range(self.n_splits):
            start = i * self.n_per_split
            end = i * self.n_per_split + self.n_per_split
            variable_splits.append(variable_list[start:end])
            dfs.append(df[df[self.variable].isin(variable_list[start:end])])
        return dfs


from datetime import datetime, timedelta


## write a function that takes a structure and returns a random structure within timedelta from it
def get_random_structure(structure, date_dict, timedelta=timedelta(days=365)):
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


class DateSplit(SplitBase):
    """
    Splits the data by date
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
            dfs.append(df[df[self.variable].isin(variable_split)])
        return dfs


class SimilaritySplit(SplitBase):
    """
    Splits the structures available to dock to by similarity to the query ligand
    "Variable" is the column name for the similarity between the query and reference ligands
    """

    name: str = "SimilaritySplit"
    type_: str = "SimilaritySplit"
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

    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        dfs = []
        # this is a bit of a confusing logic gate but if you sort it out it makes sense
        if self.include_similar == self.higher_is_more_similar:
            dfs.append(df[df[self.variable] >= self.threshold])
        elif self.include_similar != self.higher_is_more_similar:
            dfs.append(df[df[self.variable] <= self.threshold])
        return dfs

    def get_records(self) -> dict:
        return {
            "Split": self.name,
            "N_Per_Split": self.n_per_split,
            "Split_Variable": self.variable,
            "Similarity_Threshold": self.threshold,
            "Include_Similar": self.include_similar,
            "Higher_Is_More_Similar": self.higher_is_more_similar,
        }


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

            ci_upper = beta(
                self.fraction * self.total + 1, (1 - self.fraction) * self.total + 1
            ).interval(0.95)[1]

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
    dataset_split: SplitBase = Field(..., description="Dataset split")
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

    def to_json_file(self, file_path: str | Path) -> Path:
        import json

        with open(file_path, "w") as f:
            json.dump(self.dict(), f)
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


class Settings(BaseModel):
    date_dict_path: Optional[str] = Field(None, description="Path to the date dict")
    n_bootstraps: int = 1000
    rmsd_cutoff: float = 2.0
    n_per_split: list[int] = list([1] + list(range(5, 206, 5)))
    n_structures: list[int] = [1, 2, 5, 10]
    query_ligand_column: str = "Query_Ligand"
    reference_ligand_column: str = "Reference_Ligand"
    reference_structure_column: str = "Reference_Structure"
    reference_structure_date_column: str = "Reference_Structure_Date"
    pose_id_column: str = "Pose_ID"
    n_poses: list[int] = [1]
    use_date_split: bool = False
    use_random_split: bool = True
    randomize_by_n_days: int = 0
    use_posit_scorer: bool = True
    posit_score_column_name: str = "docking-confidence-POSIT"
    posit_name: str = "POSIT_Probability"
    use_rmsd_scorer: bool = True
    rmsd_column_name: str = "RMSD"
    rmsd_name: str = "RMSD"

    @field_validator("date_dict_path", mode="before")
    def check_date_dict_path(cls, v):
        if v is None:
            return None
        if not Path(v).exists():
            raise ValueError(f"Path {v} does not exist")
        return v

    @field_validator("n_per_split", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, int):
            return [v]
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v

    @classmethod
    def from_yml_file(cls, file_path: Path) -> "Settings":
        import yaml

        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yml_file(self, file_path: Path) -> Path:
        import yaml

        with open(file_path, "w") as f:
            yaml.safe_dump(self.dict(), f)
        return file_path

    class Config:
        validate_assignment = True
