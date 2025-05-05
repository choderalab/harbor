import itertools
import logging
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
from typing_extensions import Self
import abc
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional
from enum import Flag, auto
import json
import yaml
from enum import Enum, StrEnum
from operator import eq, gt, lt, ge, le, ne
from pydantic import confloat


class Operator(StrEnum):
    EQ = "eq"
    GT = "gt"
    LT = "lt"
    GE = "ge"
    LE = "le"
    NE = "ne"
    IN = "in"

    def to_callable(self) -> callable:
        def isin(x, value):
            if isinstance(value, (list, tuple, set)):
                return x in value
            return False

        return {
            self.EQ: eq,
            self.GT: gt,
            self.LT: lt,
            self.GE: ge,
            self.LE: le,
            self.NE: ne,
            self.IN: isin,
        }[self]


class ColumnFilter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    column: str = Field(..., description="Column to filter on")
    value: str | int | float | list
    operator: Operator = Operator.EQ

    @model_validator(mode="after")
    def match_operator_with_value(self):
        if self.operator == Operator.IN and not isinstance(self.value, (list, tuple)):
            raise ValueError("Operator 'in' requires value to be a list or tuple.")
        if self.operator != Operator.IN and isinstance(self.value, (list, tuple)):
            raise ValueError("Operator 'in' is only valid for list or tuple values.")
        return self

    def filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.column not in dataframe.columns:
            raise ValueError(f"Column '{self.column}' not found in DataFrame.")
        df = dataframe[
            dataframe[self.column].apply(
                lambda x: self.operator.to_callable()(x, self.value)
            )
        ]
        return df


class ColumnSortFilter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    key_columns: list[str] = Field(..., description="Columns to get unique data from")
    sort_column: str = Field(..., description="Columns to sort by")
    ascending: bool = Field(
        True, description="Sort in ascending order if True, descending if False"
    )
    number_to_return: int = Field(1, description="Number of rows to return")

    def filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        if self.sort_column in self.key_columns:
            # If the sort column is also a key column, we need to remove it from the key columns
            self.key_columns.remove(self.sort_column)

        if self.sort_column not in dataframe.columns:
            raise ValueError(f"Column '{self.sort_column}' not found in DataFrame.")
        df = (
            dataframe.sort_values(self.sort_column, ascending=self.ascending)
            .groupby([key for key in self.key_columns])
            .head(self.number_to_return)
        )
        return df


def merge_on_common_columns(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """Merge two DataFrames on all their common columns."""
    common_cols = list(set(df1.columns) & set(df2.columns))
    if not common_cols:
        raise ValueError("No common columns found between DataFrames")
    return pd.merge(df1, df2, on=common_cols, how="inner")


class DataFrameType(StrEnum):
    """Enum for DataFrame types."""

    REFERENCE = "ReferenceData"
    QUERY = "QueryData"
    PAIRED = "PairedData"
    POSE = "PoseData"
    CHEMICAL_SIMILARITY = "ChemicalSimilarityData"
    COMBINED = "CombinedData"

    def __or__(self, other):
        if not isinstance(other, DataFrameType):
            return NotImplemented
        return (self, other)


class DataFrameModelBase(BaseModel):
    name: str = Field(..., description="Unique name refering to this dataframe's data")
    type: str = Field(..., description="Data frame type. Used for grouping.")
    dataframe: pd.DataFrame = Field(
        ..., description="DataFrame containing model data", exclude=True
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __eq__(self, other):
        if not isinstance(other, DataFrameModelBase):
            return False
        return self.dataframe.equals(other.dataframe)

    @field_validator("type")
    def check_matches_dataframe_types(cls, v):
        if not DataFrameType(v):
            raise ValueError
        return v

    def serialize(self, file_path: str | Path) -> Path:
        # first write schema to json
        file_path = Path(file_path)
        with open(file_path.with_suffix(".json"), "w") as f:
            json.dump(self.model_dump(), f)

        # then write dataframe to parquet
        self.dataframe.to_parquet(file_path.with_suffix(".parquet"))

        return file_path

    @classmethod
    def deserialize(cls, file_path: str | Path) -> "DataFrameModelBase":
        # load json schema
        file_path = Path(file_path)

        with open(file_path.with_suffix(".json"), "r") as f:
            model_schema = json.load(f)

        df = pd.read_parquet(file_path.with_suffix(".parquet"))
        return cls(dataframe=df, **model_schema)


class DataFrameModel(DataFrameModelBase):
    key_columns: list[str] = Field(
        ..., description="Columns that specify the unique keys"
    )
    other_columns: list[str] = Field(
        [], description="The other columns expected in the data"
    )

    def __eq__(self, other):
        if not isinstance(other, DataFrameModel):
            return False
        return (
            self.dataframe.equals(other.dataframe)
            and self.key_columns == other.key_columns
        )

    @field_validator("key_columns", "other_columns")
    def check_columns_are_unique(cls, v):
        if len(set(v)) < len(v):
            raise ValueError
        return v

    @model_validator(mode="after")
    def check_columns_in_dataframe(self):
        """Check if all expected columns are in DataFrame and ensure key_columns uniqueness."""
        expected_cols = self.key_columns + self.other_columns
        for col in expected_cols:
            if col not in self.dataframe.columns:
                raise ValueError(
                    f"Expected Column '{col}' specified is not present in the DataFrame columns {self.dataframe.columns}."
                )

        additional_cols = [
            col for col in self.dataframe.columns if col not in expected_cols
        ]
        self.other_columns.extend(additional_cols)
        return self


class DockingDataModel(DataFrameModelBase):
    dataframe: pd.DataFrame = Field(
        ..., description="DataFrame containing model data", exclude=True
    )
    data_types_dict: dict[str, str] = Field(
        ..., description="Dictionary mapping unique names to their type."
    )
    key_columns_dict: dict[str, list[str]] = Field(
        ...,
        description="Dictionary mapping a unique name to a list of keys needed to query the dataset",
    )
    other_columns_dict: dict[str, list[str]] = Field(
        ...,
        description="Dictionary mapping a unique name to a list of columns corresponding to each internal dataframe",
    )

    def __eq__(self, other):
        if not isinstance(other, DockingDataModel):
            return False
        return (
            self.dataframe.equals(other.dataframe)
            and set(self.get_key_columns()) == set(other.get_key_columns())
            and set(self.get_other_columns()) == set(other.get_other_columns())
            and set(self.get_dataframe_names()) == set(other.get_dataframe_names())
        )

    def get_ref_data_name(self) -> str:
        return [
            k for k, v in self.data_types_dict.items() if v == DataFrameType.REFERENCE
        ][0]

    def get_ref_column(self) -> str:
        return self.key_columns_dict[self.get_ref_data_name()][0]

    def get_lig_data_name(self) -> str:
        return [k for k, v in self.data_types_dict.items() if v == DataFrameType.QUERY][
            0
        ]

    def get_lig_column(self) -> str:
        return self.key_columns_dict[self.get_lig_data_name()][0]

    def get_unique_refs(self) -> list:
        return list(self.dataframe[self.get_ref_column()].unique())

    def get_unique_ligs(self) -> list:
        return list(self.dataframe[self.get_lig_column()].unique())

    def get_dataframe_names(self) -> list:
        return [ky for ky in self.key_columns_dict.keys()]

    def get_key_columns(self) -> list:
        return list(
            set([col for cols in self.key_columns_dict.values() for col in cols])
        )

    def get_other_columns(self) -> list:
        return list(
            set([col for cols in self.other_columns_dict.values() for col in cols])
        )

    def get_pose_data_columns(self) -> list:
        pose_data_key = [
            k for k, v in self.data_types_dict.items() if v == DataFrameType.POSE
        ][0]
        return list(self.key_columns_dict[pose_data_key])

    def get_total_poses(self) -> int:
        return len(self.dataframe.groupby(self.get_pose_data_columns()))

    def get_lig_dataframe(self) -> pd.DataFrame:
        return self.dataframe.groupby(self.get_lig_column()).head(1)

    def get_ref_dataframe(self) -> pd.DataFrame:
        return self.dataframe.groupby(self.get_ref_column()).head(1)

    @classmethod
    def from_models(cls, data_models) -> "DockingDataModel":
        from functools import reduce

        names = [model.name for model in data_models]
        if len(set(names)) < len(names):
            raise ValueError(f"DataFrameModels should have unique names: {names}")

        df = reduce(merge_on_common_columns, [model.dataframe for model in data_models])

        return DockingDataModel(
            name="DockingDataModel",
            type=DataFrameType.COMBINED,
            dataframe=df,
            data_types_dict={model.name: model.type for model in data_models},
            key_columns_dict={model.name: model.key_columns for model in data_models},
            other_columns_dict={
                model.name: model.other_columns for model in data_models
            },
        )

    @model_validator(mode="after")
    def check_columns_in_dataframe(self):
        """Check if all expected columns are in DataFrame and ensure key_columns uniqueness."""
        expected_cols = self.get_key_columns() + self.get_other_columns()
        for col in expected_cols:
            if col not in self.dataframe.columns:
                raise ValueError(
                    f"Expected Column '{col}' specified is not present in the DataFrame columns {self.dataframe.columns}."
                )
        return self

    def apply_filters(self, filters: list[ColumnFilter | ColumnSortFilter]):
        """
        Apply filters in place to self.dataframe
        :param filters:
        :return:
        """
        for filter_ in filters:
            self.dataframe = filter_.filter(self.dataframe)


class ModelBase(BaseModel):
    type_: str = Field(..., description="Type of model")

    @abc.abstractmethod
    def plot_name(self) -> str:
        pass

    def get_records(self) -> dict:
        return {}


class EmptyModel(ModelBase):
    type_: str = Field("EmptyModel", description="Empty model")

    def plot_name(self) -> str:
        return ""


class SplitBase(ModelBase):
    """
    Base class for splitting the data (i.e. Random, Dataset, Scaffold, etc)
    """

    name: str = "SplitBase"
    type_: str = "SplitBase"
    deterministic: bool = Field(
        False,
        description="Whether the split is deterministic, i.e. if True it should not be run in the bootstrapping loop.",
    )
    split_level: int = Field(
        0,
        description="Level of the split, 0 indexed. The first level is applied first, and so on.",
    )

    @abc.abstractmethod
    def run(self, data: DockingDataModel) -> [DockingDataModel]:
        pass

    @property
    def plot_name(self) -> str:
        return f"{self.name}"


class ReferenceStructureSplitBase(SplitBase):
    """
    Base class for splitting the data based on some attributes of the reference structure
    """

    reference_structure_column: str = Field(
        ..., description="Name of the column to distinguish reference structures by"
    )
    n_reference_structures: Optional[int] = Field(
        None, description="Number of values per split to generate"
    )

    def get_records(self) -> dict:
        return {
            "Reference_Split": self.name,
            "N_Reference_Structures": self.n_reference_structures,
            "Reference_Structure_Column": self.reference_structure_column,
        }


def get_unique_structures_randomized_by_date(
    df: pd.DataFrame,
    structure_column: str,
    date_column: str,
    n_structures_to_return: int,
    n_days_to_randomize: int,
    date_format="%Y-%m-%d %H:%M:%S",
    bootstraps: int = 1,
) -> list[set]:
    """
    Get a set of structures randomized by date from a dataframe.

    Args:
        df: DataFrame containing structure and date information
        structure_column: Name of the column containing structure identifiers
        date_column: Name of the column containing dates
        n_structures_to_return: Number of structures to return
        n_days_to_randomize: Number of days to randomize the selection
        date_format: Format of the dates in date_column

    Returns:
        Set of selected structure identifiers
    """
    # Get unique structures
    unique_structures = df[structure_column].unique()

    if len(unique_structures) < n_structures_to_return:
        raise ValueError(
            f"Number of Unique Structures ({len(unique_structures)}) < N Structures to Return ({n_structures_to_return})."
            f"Returning all unique structures."
        )

    # Create working dataframe with unique structures and their dates
    working_df = df[[structure_column, date_column]].drop_duplicates()
    working_df["date"] = pd.to_datetime(working_df[date_column], format=date_format)
    working_df.sort_values(by="date", inplace=True)

    # Get the date of the nth structure
    last_date = working_df.iloc[n_structures_to_return - 1]["date"]
    last_date_with_buffer = last_date + pd.Timedelta(days=n_days_to_randomize)

    # Get all structures within the date range
    candidates = working_df[working_df["date"] <= last_date_with_buffer][
        structure_column
    ].tolist()

    candidate_list = []
    for i in range(bootstraps):
        # Get a random sample of the candidates
        if len(candidates) > n_structures_to_return:
            candidate_sample = np.random.choice(
                candidates, size=n_structures_to_return, replace=False
            )
            candidate_list.append(set(candidate_sample))

        elif len(candidates) < n_structures_to_return:
            raise RuntimeError(
                f"{len(candidates)} candidates < {n_structures_to_return} structures to return."
            )
    return candidate_list


def generate_random_samples(
    values: list, n_values: int, n_samples: int
) -> list[np.ndarray]:
    """
    Generate multiple random samples from a list of values.

    Args:
        values: List of values to sample from
        n_values: Number of values to sample each time
        n_samples: Number of samples to generate

    Returns:
        List of arrays containing the sampled values
    """
    if n_values > len(values):
        raise ValueError(
            f"Cannot sample {n_values} values from a list of {len(values)} values."
        )
    return [
        np.random.choice(list(values), size=n_values, replace=False)
        for _ in range(n_samples)
    ]


class RandomSplit(ReferenceStructureSplitBase):
    """
    Randomly split the structures into n_splits
    """

    name: str = "RandomSplit"
    type_: str = "RandomSplit"

    def run(self, data: DockingDataModel, bootstraps=1) -> [DockingDataModel]:
        unique_refs = data.dataframe[self.reference_structure_column].unique()
        if self.n_reference_structures is None or self.n_reference_structures == len(
            unique_refs
        ):
            # then we're returning everything, so no differences
            return [data]
        else:
            random_ref_samples = generate_random_samples(
                unique_refs,
                n_values=self.n_reference_structures,
                n_samples=bootstraps,
            )
        return [
            DockingDataModel(
                dataframe=data.dataframe[
                    data.dataframe[self.reference_structure_column].isin(sample)
                ],
                **data.model_dump(),
            )
            for sample in random_ref_samples
        ]


class DateSplit(ReferenceStructureSplitBase):
    """
    Splits the data by date.
    """

    name: str = "DateSplit"
    type_: str = "DateSplit"
    date_column: str = Field(
        ...,
        description="Column corresponding to date deposition",
    )
    randomize_by_n_days: int = Field(
        0,
        description="Randomize the structures by n days. If 0 no randomization is done. If 1 or greater, for each structure, it can be randomly replaced by any other structure collected on that day or n-1 days from it's collection date.",
    )

    def get_records(self) -> dict:
        records = super().get_records()
        records.update(
            {
                "Randomize_by_N_Days": self.randomize_by_n_days,
                "Date_Column": self.date_column,
            }
        )
        return records

    def run(self, data: DockingDataModel, bootstraps=1) -> [DockingDataModel]:
        unique_refs = data.dataframe[self.reference_structure_column].unique()
        if self.n_reference_structures is None or self.n_reference_structures == len(
            unique_refs
        ):
            # then we're returning everything, so no differences
            return [data]

        ref_lists = get_unique_structures_randomized_by_date(
            data.dataframe,
            self.reference_structure_column,
            self.date_column,
            self.n_reference_structures,
            self.randomize_by_n_days,
            bootstraps=bootstraps,
        )
        return [
            DockingDataModel(
                dataframe=data.dataframe[
                    data.dataframe[self.reference_structure_column].isin(ref_list)
                ],
                **data.model_dump(),
            )
            for ref_list in ref_lists
        ]


class PairwiseSplitBase(SplitBase):
    name: str = "PairwiseSplitBase"
    type_: str = "PairwiseSplitBase"

    def get_records(self) -> dict:
        records = super().get_records()
        records.update({"PairwiseSplit": self.name})
        return records


class SimilaritySplit(PairwiseSplitBase):
    """
    Splits the structures available to dock to by similarity to the query ligand
    """

    name: str = "SimilaritySplit"
    type_: str = "SimilaritySplit"
    n_reference_structures: Optional[int] = Field(
        None, description="Number of values per split to generate"
    )
    similarity_column: str = Field(
        ...,
        description="Column name for the similarity between the query and reference ligands",
    )
    groupby: dict = Field(
        {},
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
        if self.n_reference_structures != -1:
            self.deterministic = False
        return self

    def run(self, data: DockingDataModel, bootstraps=1) -> [pd.DataFrame]:
        df = data.dataframe

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
        return [
            DockingDataModel(
                dataframe=(
                    df.groupby(self.query_ligand_column)
                    .apply(
                        lambda x: (
                            x
                            if len(x) <= self.n_reference_structures
                            else x.sample(n=self.n_reference_structures)
                        )
                    )
                    .reset_index(drop=True)
                ),
                **data.model_dump(),
            )
            for _ in range(bootstraps)
        ]

    def get_records(self) -> dict:
        records = super().get_records()
        records.update(
            {
                "N_Reference_Structures": self.n_reference_structures,
                "Similarity_Column": self.similarity_column,
                "Similarity_Threshold": self.threshold,
                "Include_Similar": self.include_similar,
                "Higher_Is_More_Similar": self.higher_is_more_similar,
            }
        )
        records.update({key: value for key, value in self.groupby.items()})
        return records


class ScaffoldSplitFlags(Flag):
    NONE = 0
    REQUIRES_QUERY_SUBSET = auto()
    REQUIRES_REFERENCE_SUBSET = auto()

    # one of the two must be passed
    REQUIRES_EITHER_SUBSET = auto()

    # both must be passed
    REQUIRES_BOTH = REQUIRES_QUERY_SUBSET | REQUIRES_REFERENCE_SUBSET

    # might not necessarily require them, but if they are passed, there should only be one of them
    REQUIRES_SINGLE_QUERY_SUBSET_IF_PASSED = auto()
    REQUIRES_SINGLE_QUERY_SUBSET = (
        REQUIRES_SINGLE_QUERY_SUBSET_IF_PASSED | REQUIRES_QUERY_SUBSET
    )

    REQUIRES_SINGLE_REFERENCE_SUBSET_IF_PASSED = auto()
    REQUIRES_SINGLE_REFERENCE_SUBSET = (
        REQUIRES_SINGLE_REFERENCE_SUBSET_IF_PASSED | REQUIRES_REFERENCE_SUBSET
    )

    REQUIRES_SINGLE_SUBSETS_IF_PASSED = (
        REQUIRES_SINGLE_QUERY_SUBSET_IF_PASSED
        | REQUIRES_SINGLE_REFERENCE_SUBSET_IF_PASSED
    )
    REQUIRES_SINGLE_SUBSETS = REQUIRES_BOTH | REQUIRES_SINGLE_SUBSETS_IF_PASSED

    ALLOW_OVERLAPPING_QUERY_AND_REFERENCE = auto()


class ScaffoldSplitOptions(StrEnum):
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

    @property
    def flags(self) -> ScaffoldSplitFlags:
        return {
            self.X_TO_X: (
                ScaffoldSplitFlags.REQUIRES_EITHER_SUBSET
                | ScaffoldSplitFlags.REQUIRES_SINGLE_SUBSETS_IF_PASSED
                | ScaffoldSplitFlags.ALLOW_OVERLAPPING_QUERY_AND_REFERENCE
            ),
            self.X_TO_NOT_X: ScaffoldSplitFlags.REQUIRES_SINGLE_QUERY_SUBSET,
            self.NOT_X_TO_X: ScaffoldSplitFlags.REQUIRES_SINGLE_REFERENCE_SUBSET,
            self.X_TO_Y: ScaffoldSplitFlags.REQUIRES_SINGLE_SUBSETS,
            self.X_TO_ALL: ScaffoldSplitFlags.REQUIRES_SINGLE_QUERY_SUBSET
            | ScaffoldSplitFlags.ALLOW_OVERLAPPING_QUERY_AND_REFERENCE,
            self.ALL_TO_X: ScaffoldSplitFlags.REQUIRES_SINGLE_REFERENCE_SUBSET
            | ScaffoldSplitFlags.ALLOW_OVERLAPPING_QUERY_AND_REFERENCE,
        }[self]


class ScaffoldSplit(PairwiseSplitBase):
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
    query_scaffold_id_subset: Optional[list[int | str]] = Field(
        None,
        description="List of query scaffold IDs to consider. If None, consider all scaffolds.",
    )
    reference_scaffold_id_subset: Optional[list[int | str]] = Field(
        None,
        description="List of reference scaffold IDs to consider. If None, consider all scaffolds.",
    )
    split_option: ScaffoldSplitOptions = Field(
        ...,
        description="How to split the data by scaffold",
    )
    deterministic: bool = Field(True, description="Deterministic split")

    @field_validator("split_option", mode="before")
    def convert_to_string(cls, v):
        if isinstance(v, Enum):
            return v.value
        return v

    def get_records(self) -> dict:
        records = super().get_records()
        records.update(
            {
                "Query_Scaffold_ID_Column": self.query_scaffold_id_column,
                "Reference_Scaffold_ID_Column": self.reference_scaffold_id_column,
                "Scaffold_Split_Option": self.split_option,
                "Query_Scaffold_ID_Subset": self.query_scaffold_id_subset,
                "Reference_Scaffold_ID_Subset": self.reference_scaffold_id_subset,
            }
        )
        return records

    @model_validator(mode="after")
    def validate_model(self) -> Self:
        option = self.split_option
        flags = self.split_option.flags

        # Check reference subset requirements
        if (
            ScaffoldSplitFlags.REQUIRES_REFERENCE_SUBSET in flags
            and not self.reference_scaffold_id_subset
        ):
            raise ValueError(
                f"{option} requires at least one item in reference_scaffold_id_subset"
            )
        if (
            ScaffoldSplitFlags.REQUIRES_SINGLE_REFERENCE_SUBSET_IF_PASSED in flags
            and self.reference_scaffold_id_subset
            and len(self.reference_scaffold_id_subset) != 1
        ):
            raise ValueError(
                f"{option} requires exactly one item in reference_scaffold_id_subset"
            )

        # Check query subset requirements
        if (
            ScaffoldSplitFlags.REQUIRES_QUERY_SUBSET in flags
            and not self.query_scaffold_id_subset
        ):
            raise ValueError(
                f"{option} requires at least one item in query_scaffold_id_subset"
            )
        if (
            flags & ScaffoldSplitFlags.REQUIRES_SINGLE_QUERY_SUBSET_IF_PASSED
            and self.query_scaffold_id_subset
            and len(self.query_scaffold_id_subset) != 1
        ):
            raise ValueError(
                f"{option} requires exactly one item in query_scaffold_id_subset"
            )

        # Check either subset requirement
        if ScaffoldSplitFlags.REQUIRES_EITHER_SUBSET in flags and not (
            self.query_scaffold_id_subset or self.reference_scaffold_id_subset
        ):
            raise ValueError(
                f"{option} requires at least one of query_scaffold_id_subset or reference_scaffold_id_subset"
            )

        # Check both subsets requirement
        if ScaffoldSplitFlags.REQUIRES_BOTH in flags and not (
            self.query_scaffold_id_subset and self.reference_scaffold_id_subset
        ):
            raise ValueError(
                f"{option} requires both query_scaffold_id_subset and reference_scaffold_id_subset"
            )

        # Check for overlapping scaffolds when not allowed
        if (
            self.query_scaffold_id_subset
            and self.reference_scaffold_id_subset
            and len(
                set(self.query_scaffold_id_subset).intersection(
                    self.reference_scaffold_id_subset
                )
            )
            > 0
            and not (ScaffoldSplitFlags.ALLOW_OVERLAPPING_QUERY_AND_REFERENCE in flags)
        ):
            raise ValueError(
                f"Query and reference scaffold IDs are the same ({self.query_scaffold_id_subset[0]}), "
                f"but {option} does not allow overlapping scaffolds."
            )

        return self

    def run(self, data: DockingDataModel) -> [DockingDataModel]:
        """Split data based on scaffold relationships."""
        df = data.dataframe
        # set scaffold subsets if not provided
        if self.reference_scaffold_id_subset is None:
            self.reference_scaffold_id_subset = (
                df[self.reference_scaffold_id_column].unique().tolist()
            )
        if self.query_scaffold_id_subset is None:
            self.query_scaffold_id_subset = (
                df[self.query_scaffold_id_column].unique().tolist()
            )

        # Filter by scaffold subsets first
        mask = df[self.query_scaffold_id_column].isin(self.query_scaffold_id_subset)
        mask &= df[self.reference_scaffold_id_column].isin(
            self.reference_scaffold_id_subset
        )
        df = df[mask]

        # Handle different split options
        if self.split_option == ScaffoldSplitOptions.X_TO_X:
            df = df[
                df[self.query_scaffold_id_column]
                == df[self.reference_scaffold_id_column]
            ]
        elif self.split_option in (
            ScaffoldSplitOptions.NOT_X_TO_X,
            ScaffoldSplitOptions.X_TO_NOT_X,
        ):
            df = df[
                df[self.query_scaffold_id_column]
                != df[self.reference_scaffold_id_column]
            ]
        return [DockingDataModel(dataframe=df, **data.model_dump())]


# TODO: There might be a better way to do this.
ReferenceSplitType = RandomSplit | DateSplit
SimilaritySplitType = SimilaritySplit | ScaffoldSplit


class SorterBase(ModelBase):
    type_: str = "SorterBase"
    name: str = Field(..., description="Name of sorting method")
    category: str = Field(
        ..., description="Category of sort (i.e. why is sorting necessary here"
    )
    variable: str = Field(..., description="Variable used to sort the data")
    ascending: bool = Field(True, description="Higher values are better. Defaults True")
    number_to_return: Optional[int] = Field(
        None, description="Number of values to return. Returns all values if None."
    )

    @field_validator("number_to_return", mode="before")
    def allow_number_to_return_to_be_none(cls, v):
        if v is None:
            return None
        return v

    @abc.abstractmethod
    def run(self, data: DockingDataModel) -> DockingDataModel:
        pass

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


class PoseSelector(SorterBase):
    type_: str = "PoseSelector"
    category: str = "PoseSelection"

    def run(self, data: DockingDataModel) -> DockingDataModel:
        key_columns = data.get_pose_data_columns()
        sf = ColumnSortFilter(
            sort_column=self.variable,
            key_columns=key_columns,
            ascending=self.ascending,
            number_to_return=self.number_to_return,
        )
        data.apply_filters([sf])
        return data


class Scorer(SorterBase):
    category: str = "Score"
    type_: str = "Scorer"

    def run(self, data: DockingDataModel) -> DockingDataModel:
        key_columns = [data.get_lig_column()]

        sf = ColumnSortFilter(
            sort_column=self.variable,
            key_columns=key_columns,
            ascending=self.ascending,
            number_to_return=self.number_to_return,
        )
        data.apply_filters([sf])
        return data


class POSITScorer(Scorer):
    type_: str = "POSITScorer"
    name: str = "POSIT_Probability"
    variable: str = "docking-confidence-POSIT"
    ascending: bool = False
    number_to_return: int = 1


class RMSDScorer(Scorer):
    type_: str = "RMSDScorer"
    name: str = "RMSD"
    variable: str = "RMSD"
    ascending: bool = True
    number_to_return: int = 1


class SuccessRate(ModelBase):
    name: str = "SuccessRate"
    type_: str = "SuccessRate"
    total: int = Field(..., description="Total number of items being evaluated")
    fraction: confloat(ge=0, le=1) = Field(..., description="Fraction of successes")
    replicates: list[float] = Field(
        [], description="Replicates used for error bar analysis"
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
    def from_replicates(cls, reps: list["SuccessRate"]) -> "SuccessRate":
        all_fracs = np.array([rep.fraction for rep in reps])
        totals = np.array([rep.total for rep in reps])
        return SuccessRate(
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

    def run(self, data: DockingDataModel) -> SuccessRate:
        df = data.dataframe
        total_by_ligand = len(df.groupby(data.get_lig_column()))
        if total_by_ligand < len(df):
            # check which columns it can be
            raise ValueError(
                f"There are more rows in your dataframe ({len(df)}) than can be selected by {data.get_lig_column()}, ({total_by_ligand})"
            )
        if total_by_ligand == 0:
            return SuccessRate(total=0, fraction=0)
        if self.below_cutoff_is_good:
            fraction = (
                df[self.variable].apply(lambda x: x <= self.cutoff).sum()
                / total_by_ligand
            )
        else:
            fraction = (
                df[self.variable].apply(lambda x: x >= self.cutoff).sum()
                / total_by_ligand
            )
        return SuccessRate(total=total_by_ligand, fraction=fraction)

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
        case "Scorer":
            return Scorer
        case "RMSDScorer":
            return RMSDScorer
        case "POSITScorer":
            return POSITScorer
        case "PoseSelector":
            return PoseSelector
        case "FractionGood":
            return SuccessRate
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
    dataset_split: Optional[ReferenceSplitType] = Field(
        None, description="Dataset split"
    )
    similarity_split: Optional[SimilaritySplitType] = Field(
        None, description="Additional dataset splits to be run after the first one"
    )
    scorer: Scorer = Field(..., description="How to score and rank resulting poses")
    evaluator: BinaryEvaluation = Field(
        ..., description="How to determine how good the results are"
    )
    n_bootstraps: int = Field(1, description="Number of bootstrap replicates to run")

    def run_pose_selector(self, data: DockingDataModel) -> DockingDataModel:
        return self.pose_selector.run(data)

    def run_dataset_split(self, data: DockingDataModel) -> [DockingDataModel]:
        if self.dataset_split is not None:
            return self.dataset_split.run(data, bootstraps=self.n_bootstraps)
        else:
            return [data]

    def run_similarity_split(
        self, data_splits: [DockingDataModel]
    ) -> [DockingDataModel]:
        if self.similarity_split is not None:
            data_splits: list[DockingDataModel] = [
                split
                for data_ in data_splits
                for split in self.similarity_split.run(data_)
            ]
        return data_splits

    def run_scorer(self, data_splits: [DockingDataModel]) -> [DockingDataModel]:
        return [self.scorer.run(data_) for data_ in data_splits]

    def calculate_results(self, data_splits: [DockingDataModel]) -> SuccessRate:
        results = [self.evaluator.run(data_) for data_ in data_splits]
        return SuccessRate.from_replicates(results)

    def run(self, data: DockingDataModel) -> SuccessRate:
        pose_selected = self.run_pose_selector(data)
        dataset_split_data = self.run_dataset_split(pose_selected)
        similarity_split_data = self.run_similarity_split(dataset_split_data)
        scored_data = self.run_scorer(similarity_split_data)
        return self.calculate_results(scored_data)

    @field_validator(
        "pose_selector",
        "dataset_split",
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
        with open(file_path, "w") as f:
            f.write(self.model_dump_json())
        return file_path

    @classmethod
    def from_json_file(cls, file_path: str | Path) -> "Evaluator":
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)

    @property
    def plot_name(self) -> str:
        variables = [
            model.plot_name
            for model in [self.dataset_split, self.scorer]
            if model is not None
        ]
        variables += [f"{self.n_bootstraps}reps"]
        return "_".join(variables)

    def get_records(self) -> dict:
        mydict = {"Bootstraps": self.n_bootstraps}
        for container in [
            self.scorer,
            self.evaluator,
            self.pose_selector,
        ]:
            if container is not None:
                mydict.update(container.get_records())
        if self.dataset_split:
            mydict.update(self.dataset_split.get_records())
        if self.similarity_split:
            mydict.update(self.similarity_split.get_records())
        return mydict


class Results(BaseModel):
    evaluator: Evaluator
    success_rate: SuccessRate = Field(
        ..., description="Resulting success rate, with some information about the data"
    )

    def get_records(self) -> dict:
        mydict = self.evaluator.get_records()
        mydict.update(self.success_rate.get_records())
        return mydict

    @classmethod
    def calculate_result(
        cls, evaluator: Evaluator, data: DockingDataModel
    ) -> "Results":
        result = evaluator.run(data)
        return cls(evaluator=evaluator, success_rate=result)

    @classmethod
    def calculate_results(
        cls, data: DockingDataModel, evaluators: list[Evaluator]
    ) -> list["Results"]:
        for ev in tqdm(evaluators):
            result = ev.run(data)
            yield cls(evaluator=ev, success_rate=result)

    @classmethod
    def df_from_results(cls, results: list["Results"]) -> pd.DataFrame:
        return pd.DataFrame.from_records([result.get_records() for result in results])


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

    def to_yaml_file(self, file_path) -> Path:
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
        return Path(file_path)

    @classmethod
    def from_yaml(cls, yaml_str):
        data = yaml.safe_load(yaml_str)
        return cls(**data)

    @classmethod
    def from_yaml_file(cls, file_path):
        with open(file_path, "r") as file:
            return cls.from_yaml(file.read())


class EvaluatorSettingsBase(SettingsBase):
    """Base class for all evaluator settings with automatic exclude handling"""

    use: bool = Field(
        False, description="Whether this class of settings should be used"
    )


class PoseSelectionSettings(EvaluatorSettingsBase):
    """
    Settings flags used to generate
    """

    use: bool = True
    pose_id_column: str = Field(
        "Pose_ID", description="Name of the column containing the pose id"
    )
    n_poses: list[int] = Field([1], description="Number of poses to select")


class RandomSplitSettings(EvaluatorSettingsBase):
    pass


class DateSplitSettings(EvaluatorSettingsBase):
    """Settings for date-based splitting"""

    reference_structure_date_column: str = Field(
        "Reference_Structure_Date",
        description="Column containing reference structure deposition date",
    )
    randomize_by_n_days: int = Field(1, description="Days to randomize by")


class UpdateReferenceSettings(EvaluatorSettingsBase):
    use_logarithmic_scaling: bool = False
    log_base: int = 10


class CompositSettingsBase(EvaluatorSettingsBase):

    @abc.abstractmethod
    def get_component_settings(self) -> list[EvaluatorSettingsBase]:
        pass

    @model_validator(mode="after")
    def validate_component_settings(self):
        if self.use and not any(
            [component.use for component in self.get_component_settings()]
        ):
            raise ValueError(
                f"At least one of {self.get_component_settings()} must be set to use=True"
            )
        return self


class ReferenceSplitSettings(CompositSettingsBase):
    random_split_settings: RandomSplitSettings = RandomSplitSettings()
    date_split_settings: DateSplitSettings = DateSplitSettings()

    n_reference_structures: Optional[list[None | int]] = Field(
        [None], description="List of number of structures to try"
    )
    update_reference_settings: UpdateReferenceSettings = UpdateReferenceSettings()

    def get_component_settings(self) -> list[EvaluatorSettingsBase]:
        return [self.random_split_settings, self.date_split_settings]

    @field_validator("n_reference_structures", mode="before")
    def convert_to_list(cls, v):
        if isinstance(v, int):
            return [v]
        elif isinstance(v, np.ndarray):
            return v.tolist()
        return v


class ScaffoldSplitSettings(EvaluatorSettingsBase):
    """Settings for scaffold-based splitting"""

    scaffold_split_option: ScaffoldSplitOptions = Field(
        ScaffoldSplitOptions.X_TO_X, description="How to split data by scaffold"
    )
    query_scaffold_id_column: str = Field(
        "cluster_id", description="Column containing query scaffold ID"
    )
    reference_scaffold_id_column: str = Field(
        "cluster_id_Reference", description="Column containing reference scaffold ID"
    )
    query_scaffold_id_subset: Optional[list[int]] = Field(
        None, description="List of query scaffold IDs to consider"
    )
    reference_scaffold_id_subset: Optional[list[int]] = Field(
        None, description="List of reference scaffold IDs to consider"
    )
    query_scaffold_min_count: Optional[int] = Field(
        5, description="Minimum ligands in query scaffold"
    )
    reference_scaffold_min_count: Optional[int] = Field(
        5, description="Minimum ligands in reference scaffold"
    )


class SimilaritySplitSettings(EvaluatorSettingsBase):
    similarity_column_name: Optional[str] = Field("Tanimoto")
    similarity_range: list[float] = Field([0, 1])
    similarity_n_thresholds: int = Field(21)
    similarity_groupby_dict: dict = {}
    higher_is_more_similar: bool = True
    include_similar: bool = True
    n_reference_structures: Optional[list[None | int]] = Field(
        [None], description="List of number of structures to try"
    )
    update_reference_settings: UpdateReferenceSettings = UpdateReferenceSettings()

    def get_similarity_thresholds(self) -> np.ndarray:
        """
        Generate similarity thresholds from the range and number of thresholds
        :return:
        """
        return np.linspace(
            self.similarity_range[0],
            self.similarity_range[1],
            self.similarity_n_thresholds,
        )


class PairwiseSplitSettings(CompositSettingsBase):
    similarity_split_settings: SimilaritySplitSettings = Field(
        SimilaritySplitSettings()
    )
    scaffold_split_settings: ScaffoldSplitSettings = Field(ScaffoldSplitSettings())

    def get_component_settings(self) -> list[EvaluatorSettingsBase]:
        return [self.similarity_split_settings, self.scaffold_split_settings]


class POSITScorerSettings(EvaluatorSettingsBase):
    """Settings for scoring methods"""

    use: bool = True
    posit_score_column_name: str = Field(
        "docking-confidence-POSIT",
        description="Name of the column containing the POSIT score",
    )
    posit_name: str = Field("POSIT_Probability", description="Name of the POSIT score")


class RMSDScorerSettings(EvaluatorSettingsBase):
    use: bool = True
    rmsd_column_name: str = "RMSD"
    rmsd_name: str = Field("RMSD", description="Name of the RMSD score")


class ScorerSettings(CompositSettingsBase):
    use: bool = True
    rmsd_scorer_settings: RMSDScorerSettings = RMSDScorerSettings()
    posit_scorer_settings: POSITScorerSettings = POSITScorerSettings()

    def get_component_settings(self) -> list[EvaluatorSettingsBase]:
        return [self.rmsd_scorer_settings, self.posit_scorer_settings]


class SuccessRateSettings(EvaluatorSettingsBase):
    use: bool = True
    success_rate_column: str = "RMSD"
    rmsd_cutoff: float = Field(
        2.0, description="RMSD cutoff to label the resulting poses as successful"
    )


def generate_logarithmic_scale(n_max: int, base: int = 10) -> list[int]:
    """
    Generate a logarithmic scale with nice number spacing up to n_max.

    Args:
        n_max: Maximum value in the sequence
        base: Logarithm base (default=10)

    Returns:
        List of integers representing the scale

    Example:
        >>> generate_logarithmic_scale(300)
        [1, 2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300]
    """
    scale = []
    for exp in range(int(np.log(n_max) / np.log(base)) + 1):
        power = base**exp
        if power > n_max:
            break

        # Add standard power values
        if power <= n_max:
            scale.append(int(power))

        # Add intermediate values
        if power * 2 <= n_max:
            scale.append(int(power * 2))
        if power * 5 <= n_max:
            scale.append(int(power * 5))

        # Add quarter values for larger numbers
        if power >= 100:
            if power * 1.5 <= n_max:
                scale.append(int(power * 1.5))
            if power * 2.5 <= n_max:
                scale.append(int(power * 2.5))

    # Add the maximum value if it's not already included
    if n_max not in scale:
        scale.append(n_max)

    return sorted(list(set(scale)))


class EvaluatorFactory(SettingsBase):
    name: str = Field(..., help="Name of this collection of settings")
    pose_selection_settings: PoseSelectionSettings = Field(PoseSelectionSettings())
    reference_split_settings: ReferenceSplitSettings = Field(ReferenceSplitSettings())
    pairwise_split_settings: PairwiseSplitSettings = Field(PairwiseSplitSettings())
    scorer_settings: ScorerSettings = Field(ScorerSettings())
    success_rate_evaluator_settings: SuccessRateSettings = Field(SuccessRateSettings())

    class Config:
        validate_assignment = True

    combine_reference_and_similarity_splits: bool = Field(
        True,
        description="If both reference  and pairwise splits are set to use=True, evaluate them at the same time. ",
    )
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

    def to_yaml_file(self, directory: Path = Path("./")) -> Path:
        # Convert to YAML
        output = self.to_yaml()
        descriptions = self.get_descriptions()

        # Write to file with descriptions as a block comment at the top
        file_path = directory / f"{self.name}.yaml"
        with open(file_path, "w") as file:
            for key, value in output.items():
                if key in descriptions:
                    file.write(f"# {key}: {descriptions[key]}\n")

            # then write out full object
            yaml.dump(output, file, sort_keys=False)
        return file_path

    def create_pose_selectors(self) -> list[PoseSelector]:
        return [
            PoseSelector(
                name="Default",
                variable=self.pose_selection_settings.pose_id_column,
                number_to_return=n,
            )
            for n in self.pose_selection_settings.n_poses
        ]

    def create_reference_splits(
        self, data: DockingDataModel = None
    ) -> list[ReferenceSplitType]:

        reference_splits = []
        if self.reference_split_settings.update_reference_settings.use:
            if (
                self.reference_split_settings.update_reference_settings.use_logarithmic_scaling
            ):
                number_of_refs = len(data.get_unique_refs())
                self.reference_split_settings.n_reference_structures = generate_logarithmic_scale(
                    n_max=number_of_refs,
                    base=self.reference_split_settings.update_reference_settings.log_base,
                )
            else:
                raise NotImplementedError

        if self.reference_split_settings.random_split_settings.use:
            reference_splits.extend(
                [
                    RandomSplit(
                        reference_structure_column=self.reference_structure_column,
                        n_reference_structures=i,
                    )
                    for i in self.reference_split_settings.n_reference_structures
                ]
            )
        if self.reference_split_settings.date_split_settings.use:
            date_settings = self.reference_split_settings.date_split_settings
            if data is None:
                raise ValueError("Must provide input dataframe to use date split")
            reference_splits.extend(
                [
                    DateSplit(
                        reference_structure_column=self.reference_structure_column,
                        date_column=date_settings.reference_structure_date_column,
                        n_reference_structures=i,
                        randomize_by_n_days=date_settings.randomize_by_n_days,
                    )
                    for i in self.reference_split_settings.n_reference_structures
                ]
            )
        return reference_splits

    def create_pairwise_split(
        self, data: DockingDataModel = None
    ) -> list[SimilaritySplitType]:
        """Create pairwise splits (scaffold or similarity) based on settings"""
        settings = self.pairwise_split_settings
        splits = []

        # Handle scaffold splits
        if settings.scaffold_split_settings.use:
            scaffold_settings = settings.scaffold_split_settings

            # Filter scaffolds by minimum count if data is provided
            if data and (
                scaffold_settings.query_scaffold_min_count
                or scaffold_settings.reference_scaffold_min_count
            ):
                lig_data = data.get_lig_dataframe()
                ref_data = data.get_ref_dataframe()
                query_counts = lig_data[
                    scaffold_settings.query_scaffold_id_column
                ].value_counts()
                ref_counts = ref_data[
                    scaffold_settings.reference_scaffold_id_column
                ].value_counts()

                if scaffold_settings.query_scaffold_min_count:
                    valid_query_scaffolds = query_counts[
                        query_counts >= scaffold_settings.query_scaffold_min_count
                    ].index.tolist()
                    scaffold_settings.query_scaffold_id_subset = valid_query_scaffolds

                if scaffold_settings.reference_scaffold_min_count:
                    valid_ref_scaffolds = ref_counts[
                        ref_counts >= scaffold_settings.reference_scaffold_min_count
                    ].index.tolist()
                    scaffold_settings.reference_scaffold_id_subset = valid_ref_scaffolds

            # Handle different split options
            split_option = scaffold_settings.scaffold_split_option
            flags = split_option.flags

            # Initialize lists of query and reference scaffolds
            query_scaffolds = scaffold_settings.query_scaffold_id_subset or []
            ref_scaffolds = scaffold_settings.reference_scaffold_id_subset or []

            if split_option == ScaffoldSplitOptions.X_TO_X:
                # Use the same scaffolds for both query and reference
                common_scaffolds = list(
                    set(query_scaffolds).intersection(ref_scaffolds)
                )
                splits.extend(
                    [
                        ScaffoldSplit(
                            query_scaffold_id_column=scaffold_settings.query_scaffold_id_column,
                            reference_scaffold_id_column=scaffold_settings.reference_scaffold_id_column,
                            query_scaffold_id_subset=[scaffold],
                            reference_scaffold_id_subset=[scaffold],
                            split_option=split_option,
                        )
                        for scaffold in common_scaffolds
                    ]
                )

            elif split_option == ScaffoldSplitOptions.X_TO_NOT_X:
                # Each scaffold against all other scaffolds
                for scaffold in query_scaffolds:
                    splits.append(
                        ScaffoldSplit(
                            query_scaffold_id_column=scaffold_settings.query_scaffold_id_column,
                            reference_scaffold_id_column=scaffold_settings.reference_scaffold_id_column,
                            query_scaffold_id_subset=[scaffold],
                            reference_scaffold_id_subset=[
                                s for s in ref_scaffolds if s != scaffold
                            ],
                            split_option=split_option,
                        )
                    )

            elif split_option == ScaffoldSplitOptions.X_TO_Y:
                # All possible pairs of different scaffolds
                splits.extend(
                    [
                        ScaffoldSplit(
                            query_scaffold_id_column=scaffold_settings.query_scaffold_id_column,
                            reference_scaffold_id_column=scaffold_settings.reference_scaffold_id_column,
                            query_scaffold_id_subset=[q],
                            reference_scaffold_id_subset=[r],
                            split_option=split_option,
                        )
                        for q, r in itertools.product(query_scaffolds, ref_scaffolds)
                        if q != r
                    ]
                )

        # Handle similarity splits (if implemented)
        if settings.similarity_split_settings.use:
            sim_settings = settings.similarity_split_settings

            # updated n_refs if passed
            if sim_settings.update_reference_settings.use:
                if sim_settings.update_reference_settings.use_logarithmic_scaling:
                    number_of_refs = len(data.get_unique_refs())
                    sim_settings.n_reference_structures = generate_logarithmic_scale(
                        n_max=number_of_refs,
                        base=sim_settings.update_reference_settings.log_base,
                    )
                else:
                    raise NotImplementedError

            # Add similarity split implementation here
            splits.extend(
                [
                    SimilaritySplit(
                        n_reference_structures=refs,
                        threshold=threshold,
                        similarity_column=sim_settings.similarity_column_name,
                        groupby=sim_settings.similarity_groupby_dict,
                        higher_is_more_similar=sim_settings.higher_is_more_similar,
                        include_similar=sim_settings.include_similar,
                        query_ligand_column=self.query_ligand_column,
                    )
                    for refs in settings.similarity_split_settings.n_reference_structures
                    for threshold in settings.similarity_split_settings.get_similarity_thresholds()
                ]
            )
        return splits

    def create_scorers(self) -> list[Scorer]:
        """Create scorers based on settings"""
        scorers = []
        settings = self.scorer_settings

        if settings.rmsd_scorer_settings.use:
            rmsd_settings = settings.rmsd_scorer_settings
            scorers.append(
                RMSDScorer(
                    name=rmsd_settings.rmsd_name,
                    variable=rmsd_settings.rmsd_column_name,
                )
            )

        if settings.posit_scorer_settings.use:
            posit_settings = settings.posit_scorer_settings
            scorers.append(
                POSITScorer(
                    name=posit_settings.posit_name,
                    variable=posit_settings.posit_score_column_name,
                )
            )

        return scorers

    def create_success_rate_evaluator(self) -> [BinaryEvaluation]:
        return [
            BinaryEvaluation(
                variable=self.success_rate_evaluator_settings.success_rate_column,
                cutoff=self.success_rate_evaluator_settings.rmsd_cutoff,
            )
        ]

    def create_evaluators(self, data: DockingDataModel = None) -> list[Evaluator]:
        """Create all evaluator combinations based on settings"""
        pose_selectors = self.create_pose_selectors()
        reference_splits = (
            self.create_reference_splits(data)
            if self.reference_split_settings.use
            else None
        )
        similarity_splits = (
            self.create_pairwise_split(data)
            if self.pairwise_split_settings.use
            else None
        )
        scorers = self.create_scorers()
        success_rate_evaluators = self.create_success_rate_evaluator()

        evaluators = []
        for pose_selector in pose_selectors:
            for scorer in scorers:
                for success_rate_evaluator in success_rate_evaluators:
                    # Create basic evaluator
                    evaluator = Evaluator(
                        pose_selector=pose_selector,
                        scorer=scorer,
                        evaluator=success_rate_evaluator,
                        n_bootstraps=self.n_bootstraps,
                    )
                    if reference_splits is not None or similarity_splits is not None:

                        if (
                            reference_splits is not None
                            and similarity_splits is not None
                        ) and self.combine_reference_and_similarity_splits:
                            for sim_split, ref_split in itertools.product(
                                similarity_splits, reference_splits
                            ):
                                ev = evaluator.copy()
                                ev.dataset_split = ref_split
                                ev.similarity_split = sim_split
                                evaluators.append(ev)
                        if reference_splits is not None:
                            for ref_split in reference_splits:
                                ev = evaluator.copy()
                                ev.dataset_split = ref_split
                                evaluators.append(ev)
                        if similarity_splits is not None:
                            for sim_split in similarity_splits:
                                ev = evaluator.copy()
                                ev.similarity_split = sim_split
                                evaluators.append(ev)
                    else:
                        evaluators.append(evaluator)

        return evaluators
