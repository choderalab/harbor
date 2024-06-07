from pydantic import BaseModel, Field
import abc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


class ModelBase(BaseModel):

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
    variable: str = Field(description="Name of variable used to split the data")
    n_splits: int = Field(1, description="number of splits to generate")
    n_per_split: int = Field(..., description="Number of values per split to generate")

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        pass

    @property
    def plot_name(self) -> str:
        return f"{self.name}_{self.n_per_split}"

    def get_records(self) -> dict:
        return {"Split": self.name, "N_Per_Split": self.n_per_split}


class RandomSplit(SplitBase):
    """
    Randomly split the structures into n_splits
    """

    name: str = "RandomSplit"

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


class DateSplit(SplitBase):
    """
    Splits the data by date
    """

    name: str = "DateSplit"
    date_dict: dict = Field(
        ...,
        description="Dictionary of dates to split the data by of the form dict[str, str] where the key is the structure name and the value is the date",
    )
    balanced: bool = Field(
        True,
        description="Whether to split the data uniformly in time (i.e. 1 split every N months) or balanced such that each split has the same number of structures",
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
            variable_splits.append(structure_list[start:end])
            dfs.append(df[df[self.variable].isin(structure_list[start:end])])
        return dfs


class SorterBase(ModelBase):
    name: str = Field(..., description="Name of sorting method")
    category: str = Field(
        ..., description="Category of sort (i.e. why is sorting necessary here"
    )
    variable: str = Field(..., description="Variable used to sort the data")
    higher_is_better: bool = Field(
        True, description="Higher values are better. Defaults True"
    )
    number_to_return: int = Field(
        None, description="Number of values to return. Returns all values if None."
    )

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
    category: str = "StructureChoice"


class Scorer(SorterBase):
    category: str = "Score"


class PoseSelector(SorterBase):
    category: str = "PoseSelection"
    groupby: list[str] = ["Query_Ligand", "Reference_Ligand"]
    higher_is_better: bool = False

    def run(self, df: pd.DataFrame, groupby=None) -> pd.DataFrame:
        return SorterBase.run(self, df, groupby=self.groupby)


class FractionGood(ModelBase):
    from pydantic import confloat

    name: str = "FractionGood"
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
            total=totals.mean(), fraction=all_fracs.mean(), replicates=list(all_fracs)
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


class Evaluator(ModelBase):
    name: str = "Evaluator"
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
        return FractionGood.from_replicates(results)

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
    date_dict_path: str
    n_bootstraps: int = 1000
    rmsd_cutoff: float = 2.0
    n_per_split: list[int] = list([1] + list(range(5, 206, 5)))
    n_structures: list[int] = [1, 2, 5, 10]
    query_ligand_column: str = "Query_Ligand"
    reference_ligand_column: str = "Reference_Ligand"
    reference_structure_column: str = "Reference_Structure"
    pose_id_column: str = "Pose_ID"
    n_poses: list[int] = [1, 2, 5, 10, 20, 50]

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
