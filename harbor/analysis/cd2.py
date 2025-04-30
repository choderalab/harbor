import logging
import warnings

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
from pydantic import BaseModel, Field, model_validator, field_validator, ConfigDict
import pandas as pd
from pyarrow import parquet as pq
import pyarrow as pa
from pathlib import Path
from enum import Enum, StrEnum
from operator import eq, gt, lt, ge, le, ne
from pydantic import confloat


class ColumnType(StrEnum):
    """Enum for column types."""

    KEY = "key"
    VALUE = "value"
    PARAM = "param"
    INFO = "info"

    def __or__(self, other):
        if not isinstance(other, ColumnType):
            return NotImplemented
        return (self, other)


class ColumnName(BaseModel):
    name: str = Field(..., description="Column name")
    column_type: ColumnType
    required: bool = Field(True, description="Is this column required?")

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"{self.column_type.value}:{self.name}"


class ValueColumn(ColumnName):
    """Class for value columns."""

    column_type: ColumnType = Field(ColumnType.VALUE, description="Column type")


class KeyColumn(ColumnName):
    """Class for key columns."""

    column_type: ColumnType = Field(ColumnType.KEY, description="Column type")


class ParamColumn(ColumnName):
    """Class for parameter columns."""

    column_type: ColumnType = Field(ColumnType.PARAM, description="Column type")


class InfoColumn(ColumnName):
    """Class for info columns."""

    column_type: ColumnType = Field(ColumnType.INFO, description="Column type")


ColumnNames = InfoColumn | KeyColumn | ValueColumn | ParamColumn


REFERENCE_COLUMN = KeyColumn(name="Reference_Structure")
QUERY_COLUMN = KeyColumn(name="Query_Ligand")
POSE_ID_COLUMN = KeyColumn(name="Pose_ID")


class DataFrameType(StrEnum):
    """Enum for DataFrame types."""

    REFERENCE = "ReferenceData"
    QUERY = "QueryData"
    POSE = "PoseData"
    CHEMICAL_SIMILARITY = "ChemicalSimilarityData"

    def __or__(self, other):
        if not isinstance(other, DataFrameType):
            return NotImplemented
        return (self, other)


class DataFrameModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: InfoColumn = Field(..., description="Name of the dataframe model")

    dataframe: pd.DataFrame = Field(
        ...,
        description="DataFrame containing model data",
    )
    other_columns: list[ColumnName] = Field(
        [],
        description="Other columns",
    )

    def __eq__(self, other):
        if not isinstance(other, DataFrameModel):
            return False
        return (
            self.dataframe.equals(other.dataframe)
            and self.to_parquet_metadata() == other.to_parquet_metadata()
        )

    def get_columns(
        self, column_type: ColumnType | None = None, as_str: bool = False
    ) -> list[ColumnNames | str]:
        """Get all column names from model fields."""
        columns = self.columns

        # Filter by column type if specified
        if column_type:
            columns = [col for col in columns if col.column_type in column_type]

        return [str(col) for col in columns] if as_str else columns

    def to_parquet_metadata(self) -> dict:
        """Convert model fields to parquet metadata."""
        metadata = {}
        for field_name, field_value in self:
            if field_name == "dataframe":
                continue
            elif isinstance(field_value, ColumnName):
                metadata[field_name] = field_value.name
            elif isinstance(field_value, list):
                for col in field_value:
                    if isinstance(col, ColumnName):
                        metadata[col.name] = col.column_type.value
            else:
                metadata[field_name] = field_value
        return metadata

    @model_validator(mode="after")
    def check_columns_in_dataframe(self):
        non_info_types = tuple(
            col_type for col_type in ColumnType if col_type != ColumnType.INFO
        )
        in_df_columns = self.get_columns(non_info_types, as_str=True)
        for column in in_df_columns:
            if column not in self.dataframe.columns:
                raise ValueError(
                    f"Column '{column}' specified in metadata is not present in the DataFrame."
                )
        df_columns = set(self.dataframe.columns)
        for column in df_columns:
            if column not in self.get_columns(as_str=True):
                raise ValueError(
                    f"Column '{column}' in DataFrame is not specified in metadata."
                )

    @classmethod
    def from_parquet_metadata(cls, metadata: dict) -> dict:
        """Convert parquet metadata back to model fields."""
        parsed_metadata = {
            k.decode("utf-8"): v.decode("utf-8") for k, v in metadata.items()
        }

        result = {}
        # Handle model fields with ColumnName types
        for field_name, field in cls.model_fields.items():
            if field_name == "dataframe":
                continue
            elif field.annotation in [KeyColumn, ValueColumn, ParamColumn, InfoColumn]:
                if field_name in parsed_metadata:
                    result[field_name] = field.annotation(
                        name=parsed_metadata[field_name]
                    )

        # Handle other columns that were stored as column_name: column_type pairs
        other_columns = []
        for name, type_value in parsed_metadata.items():
            if type_value in ColumnType.__members__.values():
                column_type = ColumnType(type_value)
                other_columns.append(ColumnName(name=name, column_type=column_type))
            else:
                # if we haven't already parsed it
                if name not in result:
                    result[name] = parsed_metadata[name]

        if other_columns:
            result["other_columns"] = other_columns
        return result

    def to_parquet(self, path: str | Path):
        table = pa.Table.from_pandas(self.dataframe)
        table = table.replace_schema_metadata(self.to_parquet_metadata())
        pq.write_table(table, path)

    def to_csv(self, path: str):
        self.dataframe.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "DataFrameModel":
        return cls(dataframe=pd.read_csv(path), **kwargs)

    @classmethod
    def from_parquet(cls, path: str) -> "DataFrameModel":
        table = pq.read_table(path)
        metadata = cls.from_parquet_metadata(table.schema.metadata)
        return cls(dataframe=table.to_pandas(), **metadata)


class ReferenceData(DataFrameModel):
    type: DataFrameType = Field(DataFrameType.REFERENCE, description="Type of data")
    name: InfoColumn = Field(
        InfoColumn(name="ReferenceData"), description="Type of data"
    )
    reference_column: KeyColumn = Field(
        REFERENCE_COLUMN, description="Reference structure column"
    )

    @property
    def columns(self):
        return [self.name, self.reference_column] + self.other_columns


class QueryData(DataFrameModel):
    type: DataFrameType = Field(DataFrameType.QUERY, description="Type of data")
    name: InfoColumn = Field(InfoColumn(name="QueryData"), description="Type of data")
    query_column: KeyColumn = Field(QUERY_COLUMN, description="Query structure column")

    @property
    def columns(self):
        return [self.name, self.query_column] + self.other_columns


class PairwiseData(ReferenceData, QueryData):
    name: InfoColumn = Field(
        InfoColumn(name="PairwiseData"), description="Type of data"
    )
    pass

    @property
    def columns(self):
        return [
            self.name,
            self.reference_column,
            self.query_column,
        ] + self.other_columns


class PoseData(PairwiseData):
    type: DataFrameType = Field(DataFrameType.POSE, description="Type of data")
    name: InfoColumn = Field(InfoColumn(name="PoseData"), description="Type of data")
    pose_id_column: KeyColumn = Field(POSE_ID_COLUMN, description="Pose ID column")
    rmsd_column: ValueColumn = Field(
        ValueColumn(name="RMSD"), description="RMSD column"
    )

    @property
    def columns(self):
        return [
            self.name,
            self.reference_column,
            self.query_column,
            self.pose_id_column,
            self.rmsd_column,
        ] + self.other_columns


class ChemicalSimilarityData(PairwiseData):
    type: DataFrameType = Field(
        DataFrameType.CHEMICAL_SIMILARITY, description="Type of data"
    )
    name: InfoColumn = Field(..., description="Type of data")
    tanimoto_column: ValueColumn = Field(
        ValueColumn(name="Tanimoto"), description="Tanimoto similarity column"
    )

    @property
    def columns(self):
        return [
            self.name,
            self.reference_column,
            self.query_column,
            self.tanimoto_column,
        ] + self.other_columns


def get_common_columns(
    dataframes: list[DataFrameModel], column_type: ColumnType = None
) -> list[str]:
    """Get common columns across multiple DataFrames."""
    common_columns = set(dataframes[0].get_columns(column_type, as_str=True))
    for df in dataframes[1:]:
        common_columns.intersection_update(
            set(df.get_columns(column_type, as_str=True))
        )
    return list(common_columns)


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
    data_type: DataFrameType = Field(None, description="Type of data")
    data_name: ColumnName = Field(
        None, description="InfoColumn name given to a dataframe"
    )
    column: ColumnName = Field(..., description="Column to filter on")
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
        if self.column.name not in dataframe.columns:
            raise ValueError(f"Column '{self.column.name}' not found in DataFrame.")
        df = dataframe[
            dataframe[self.column.name].apply(
                lambda x: self.operator.to_callable()(x, self.value)
            )
        ]
        return df


class ColumnSortFilter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    data_type: DataFrameType = Field(None, description="Type of data")
    data_name: ColumnName = Field(
        None, description="InfoColumn name given to a dataframe"
    )
    key_columns: list[KeyColumn] = Field(
        ..., description="Columns to get unique data from"
    )
    sort_column: ColumnName = Field(..., description="Columns to sort by")
    ascending: bool = Field(
        True, description="Sort in ascending order if True, descending if False"
    )
    number_to_return: int = Field(1, description="Number of rows to return")

    def filter(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        if self.sort_column in self.key_columns:
            # If the sort column is also a key column, we need to remove it from the key columns
            self.key_columns.remove(self.sort_column)

        if self.sort_column.name not in dataframe.columns:
            raise ValueError(
                f"Column '{self.sort_column.name}' not found in DataFrame."
            )
        df = (
            dataframe.sort_values(self.sort_column.name, ascending=self.ascending)
            .groupby([key.name for key in self.key_columns])
            .head(self.number_to_return)
        )
        return df


class DockingDataModel:
    def __init__(
        self,
        evaluation_key_columns: list[KeyColumn],
        reference_data: ReferenceData,
        query_data: QueryData,
        pose_data: PoseData,
        chemical_similarity_data: list[ChemicalSimilarityData] | None = None,
    ):
        self.evaluation_key_columns = evaluation_key_columns
        self.reference_data = reference_data
        self.query_data = query_data
        self.pose_data = pose_data
        self.chemical_similarity_data = chemical_similarity_data or []

    def __repr__(self):
        return f"DockingDataModel(reference_data={self.reference_data}, query_data={self.query_data}, pose_data={self.pose_data}, chemical_similarity_data={self.chemical_similarity_data})"

    def get_dataframe_list(self):
        return [
            self.reference_data,
            self.query_data,
            self.pose_data,
        ] + self.chemical_similarity_data

    def get_data_as_dict(self) -> dict[str, DataFrameModel]:
        dataframes = {}
        for data in self.get_dataframe_list():
            dataframes[str(data.name)] = data
        return dataframes

    def get_key_columns(self, as_str: bool = True):
        key_columns = []
        for data in self.get_dataframe_list():
            key_columns.extend(
                [col for col in data.get_columns(ColumnType.KEY, as_str=as_str)]
            )
        return set(key_columns)

    def get_value_columns(self, as_str: bool = True):
        key_columns = []
        for data in self.get_dataframe_list():
            key_columns.extend(
                [col for col in data.get_columns(ColumnType.VALUE, as_str=as_str)]
            )
        return set(key_columns)

    def get_combined_similarity_data(self) -> pd.DataFrame:
        if self.chemical_similarity_data is None:
            return pd.DataFrame()
        combined_similarity_data = self.chemical_similarity_data[0].dataframe.copy()

        if len(self.chemical_similarity_data) == 1:
            return combined_similarity_data

        # Get common columns for merging
        similarity_columns = get_common_columns(self.chemical_similarity_data)
        for similarity_data in self.chemical_similarity_data[1:]:
            combined_similarity_data = pd.merge(
                combined_similarity_data,
                similarity_data.dataframe,
                on=similarity_columns,
                how="outer",
            )
        return combined_similarity_data

    def get_combined_dataframe(self):
        # Ensure we have the required base data
        if self.pose_data.dataframe.empty:
            raise ValueError("Pose data is empty")

        # Start with pose data
        combined_df = self.pose_data.dataframe.copy()

        # Merge reference and query data
        for data_model in [self.reference_data, self.query_data]:
            if data_model.dataframe.empty:
                raise ValueError(f"{data_model.__class__.__name__} is empty")

            combined_df = pd.merge(
                combined_df,
                data_model.dataframe,
                on=data_model.get_columns(column_type=(ColumnType.KEY), as_str=True),
                how="inner",
            )

        # Merge similarity data if exists
        if self.chemical_similarity_data:
            try:
                similarity_df = self.get_combined_similarity_data()
                if not similarity_df.empty:
                    key_cols = self.chemical_similarity_data[0].get_columns(
                        "key", as_str=True
                    )
                    combined_df = pd.merge(
                        combined_df,
                        similarity_df,
                        on=key_cols,
                        how="inner",
                    )
            except (IndexError, KeyError) as e:
                print(f"Warning: Could not merge similarity data: {e}")

        return combined_df

    def copy(self) -> "DockingDataModel":
        # Create new instances of each data model with copied dataframes
        new_reference = ReferenceData(
            dataframe=self.reference_data.dataframe.copy(),
            name=self.reference_data.name,
            reference_column=self.reference_data.reference_column,
            other_columns=self.reference_data.other_columns,
        )
        new_query = QueryData(
            dataframe=self.query_data.dataframe.copy(),
            name=self.query_data.name,
            query_column=self.query_data.query_column,
            other_columns=self.query_data.other_columns,
        )
        new_pose = PoseData(
            dataframe=self.pose_data.dataframe.copy(),
            name=self.pose_data.name,
            reference_column=self.pose_data.reference_column,
            query_column=self.pose_data.query_column,
            pose_id_column=self.pose_data.pose_id_column,
            rmsd_column=self.pose_data.rmsd_column,
            other_columns=self.pose_data.other_columns,
        )
        new_similarity = (
            [
                ChemicalSimilarityData(
                    dataframe=data.dataframe.copy(),
                    name=data.name,
                    reference_column=data.reference_column,
                    query_column=data.query_column,
                    tanimoto_column=data.tanimoto_column,
                    other_columns=data.other_columns,
                )
                for data in self.chemical_similarity_data
            ]
            if self.chemical_similarity_data
            else None
        )

        # Create new DockingDataModel
        return DockingDataModel(
            evaluation_key_columns=self.evaluation_key_columns,
            reference_data=new_reference,
            query_data=new_query,
            pose_data=new_pose,
            chemical_similarity_data=new_similarity,
        )

    def apply_filters(self, filters: list[ColumnFilter]) -> "DockingDataModel":
        """
        Create a new DockingDataModel with filtered DataFrameModels.

        Args:
            filters: List of ColumnFilter objects to apply

        Returns:
            New DockingDataModel instance with filtered data
        """
        new_data = self.copy()
        data_models = new_data.get_dataframe_list()

        for cf in filters:
            # Apply filter to matching data models
            updated_models = []
            for data in data_models:
                should_apply = (
                    (
                        cf.data_type is None
                        and cf.data_name is None
                        and cf.column.name in data.get_columns(as_str=True)
                    )
                    or (cf.data_type is not None and cf.data_type == data.type)
                    or (cf.data_name is not None and cf.data_name == data.name)
                )

                if should_apply:
                    data.dataframe = cf.filter(data.dataframe)
                    updated_models.append(data)

            # Propagate filtering to related models
            if updated_models:
                for data in data_models:
                    if data not in updated_models:
                        filtered_df = data.dataframe
                        for updated_data in updated_models:
                            common_cols = get_common_columns([data, updated_data])
                            if common_cols:
                                # Create mask using common columns
                                mask = (
                                    pd.concat(
                                        [
                                            filtered_df[common_cols].merge(
                                                updated_data.dataframe[common_cols],
                                                how="inner",
                                                on=common_cols,
                                            ),
                                            filtered_df[common_cols],
                                        ]
                                    )
                                    .drop_duplicates(keep=False)
                                    .index
                                )

                                # Apply mask to keep only rows that match in common columns
                                filtered_df = filtered_df[~filtered_df.index.isin(mask)]

                        data.dataframe = filtered_df

        return new_data

    def get_unique_refs(self) -> set:
        """
        Get unique reference structures from the reference data.
        """
        ref_data = self.reference_data
        return set(ref_data.dataframe[ref_data.reference_column.name].unique().tolist())

    def get_unique_ligs(self) -> set:
        """
        Get unique query ligands from the query data.
        """
        query_data = self.query_data
        return set(query_data.dataframe[query_data.query_column.name].unique().tolist())


class ModelBase(BaseModel):
    type_: str = Field(..., description="Type of model")

    @abc.abstractmethod
    def plot_name(self) -> str:
        pass

    @abc.abstractmethod
    def get_records(self) -> dict:
        pass


class EmptyModel(ModelBase):
    type_: str = Field("EmptyModel", description="Empty model")

    def plot_name(self) -> str:
        return ""

    def get_records(self) -> dict:
        return {}


class EmptyDataframeModel(EmptyModel):
    """
    A model that does nothing to the dataframe.
    """

    type_: str = Field("EmptyDataframeModel", description="Empty dataframe model")

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        return df


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
        return f"{self.name}"

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
    n_reference_structures: int = Field(
        None, description="Number of values per split to generate"
    )

    @abc.abstractmethod
    def run(self, df: pd.DataFrame) -> [pd.DataFrame]:
        pass

    def _get_records(self) -> dict:
        return {
            "Split": self.name,
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
        docking_model = data.copy()
        ref_list = list(docking_model.get_unique_refs())
        if self.n_reference_structures is None:
            return [docking_model for _ in range(bootstraps)]
        else:
            random_ref_samples = generate_random_samples(
                ref_list, n_values=self.n_reference_structures, n_samples=bootstraps
            )
        filters = [
            ColumnFilter(
                data_type=DataFrameType.REFERENCE,
                column=KeyColumn(name=self.reference_structure_column),
                value=sample,
                operator=Operator.IN,
            )
            for sample in random_ref_samples
        ]

        return [docking_model.apply_filters(filters=[cf]) for cf in filters]


class DateSplit(ReferenceStructureSplitBase):
    """
    Splits the data by date.
    """

    name: str = "DateSplit"
    type_: str = "DateSplit"
    date_column: str = Field(
        ...,
        description="Dictionary of dates to split the data by of the form dict[str, str] where the key is the structure name and the value is the date",
    )
    randomize_by_n_days: int = Field(
        0,
        description="Randomize the structures by n days. If 0 no randomization is done. If 1 or greater, for each structure, it can be randomly replaced by any other structure collected on that day or n-1 days from it's collection date.",
    )

    def run(self, data: DockingDataModel, bootstraps=1) -> [DockingDataModel]:
        docking_model = data.copy()

        if self.n_reference_structures is None:
            return [docking_model for _ in range(bootstraps)]

        candidate_lists = get_unique_structures_randomized_by_date(
            docking_model.reference_data.dataframe,
            self.reference_structure_column,
            self.date_column,
            self.n_reference_structures,
            self.randomize_by_n_days,
            bootstraps=bootstraps,
        )
        cfs = [
            ColumnFilter(
                data_type=DataFrameType.REFERENCE,
                column=KeyColumn(name=self.reference_structure_column),
                value=candidate_list,
                operator=Operator.IN,
            )
            for candidate_list in candidate_lists
        ]
        return [docking_model.apply_filters(filters=[cf]) for cf in cfs]


class ScaffoldSplitOptions(StrEnum):
    """Enum for scaffold split options."""

    X_TO_ALL = "x_to_all"  # Split reference by scaffold X, test against all queries
    ALL_TO_X = "all_to_x"  # Split query by scaffold X, test against all references
    X_TO_NOT_X = (
        "x_to_not_x"  # Split reference by scaffold X, test against queries not in X
    )
    NOT_X_TO_X = (
        "not_x_to_x"  # Split query by scaffold X, test against references not in X
    )
    X_TO_Y = "x_to_y"  # Split both reference and query by scaffold X


class ScaffoldSplit(SplitBase):
    """Split data based on molecular scaffolds."""

    name: str = "ScaffoldSplit"
    type_: str = "ScaffoldSplit"
    query_scaffold_id_column: str = Field(
        ..., description="Column containing query scaffold IDs"
    )
    reference_scaffold_id_column: str = Field(
        ..., description="Column containing reference scaffold IDs"
    )
    split_option: ScaffoldSplitOptions = Field(
        ..., description="Type of scaffold split to perform"
    )
    reference_scaffold_id_subset: list[str] | None = Field(
        None, description="Subset of reference scaffold IDs to use"
    )
    query_scaffold_id_subset: list[str] | None = Field(
        None, description="Subset of query scaffold IDs to use"
    )
    deterministic: bool = Field(True, description="Deterministic split")

    def _get_records(self) -> dict:
        return_dict = {
            "Split": self.name,
            "Query_Scaffold_ID_Column": self.query_scaffold_id_column,
            "Reference_Scaffold_ID_Column": self.reference_scaffold_id_column,
            "Split_Option": self.split_option,
            "Query_Scaffold_ID_Subset": self.query_scaffold_id_subset,
            "Reference_Scaffold_ID_Subset": self.reference_scaffold_id_subset,
        }
        return return_dict

    def run(self, data: DockingDataModel) -> list[DockingDataModel]:
        """Run scaffold split on the data."""
        docking_model = data.copy()

        # set scaffold subsets if not provided
        if self.reference_scaffold_id_subset is None:
            self.reference_scaffold_id_subset = (
                docking_model.reference_data.dataframe[
                    self.reference_scaffold_id_column
                ]
                .unique()
                .tolist()
            )
        if self.query_scaffold_id_subset is None:
            self.query_scaffold_id_subset = (
                docking_model.query_data.dataframe[self.query_scaffold_id_column]
                .unique()
                .tolist()
            )

        filters = [
            ColumnFilter(
                column=KeyColumn(name=self.query_scaffold_id_column),
                value=self.query_scaffold_id_subset,
                operator=Operator.IN,
            ),
            ColumnFilter(
                column=KeyColumn(name=self.reference_scaffold_id_column),
                value=self.reference_scaffold_id_subset,
                operator=Operator.IN,
            ),
        ]
        if self.split_option in (
            ScaffoldSplitOptions.X_TO_NOT_X,
            ScaffoldSplitOptions.NOT_X_TO_X,
            ScaffoldSplitOptions.X_TO_Y,
        ):
            filters.append(
                ColumnFilter(
                    data_name=InfoColumn(name="ScaffoldMatch"),
                    column=ValueColumn(name="Tanimoto"),
                    value=0,
                    operator=Operator.EQ,
                )
            )
        return [docking_model.apply_filters(filters=filters)]


DatasetSplitType = RandomSplit | DateSplit | ScaffoldSplit
ReferenceSplitType = RandomSplit | DateSplit
PairwiseSplitType = ScaffoldSplit


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
        key_columns = data.pose_data.get_columns(column_type=ColumnType.KEY)
        sf = ColumnSortFilter(
            data_type=DataFrameType.POSE,
            sort_column=KeyColumn(name=self.variable),
            key_columns=key_columns,
            ascending=self.ascending,
            number_to_return=self.number_to_return,
        )

        return data.apply_filters([sf])


class Scorer(SorterBase):
    category: str = "Score"
    type_: str = "Scorer"

    def run(self, data: DockingDataModel) -> DockingDataModel:
        key_columns = data.evaluation_key_columns
        sf = ColumnSortFilter(
            data_type=DataFrameType.POSE,
            sort_column=KeyColumn(name=self.variable),
            key_columns=key_columns,
            ascending=self.ascending,
            number_to_return=self.number_to_return,
        )

        return data.apply_filters([sf])


class POSITScorer(Scorer):
    name: str = "POSIT_Probability"
    variable: str = "docking-confidence-POSIT"
    ascending: bool = False
    number_to_return: int = 1


class RMSDScorer(Scorer):
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
        df = data.pose_data.dataframe
        total = len(df.groupby([ky.name for ky in data.evaluation_key_columns]))
        if total < len(df):
            raise ValueError(
                f"There are more rows in your dataframe ({len(df)}) than can be selected by {data.evaluation_key_columns}, ({total})"
            )
        if total == 0:
            return SuccessRate(total=0, fraction=0)
        if self.below_cutoff_is_good:
            fraction = df[self.variable].apply(lambda x: x <= self.cutoff).sum() / total
        else:
            fraction = df[self.variable].apply(lambda x: x >= self.cutoff).sum() / total
        return SuccessRate(total=total, fraction=fraction)

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
        case "ScaffoldSplit":
            return ScaffoldSplit
        case "Scorer":
            return Scorer
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
    dataset_split: DatasetSplitType = Field(..., description="Dataset split")
    extra_splits: Optional[list[DatasetSplitType]] = Field(
        None, description="Additional dataset splits to be run after the first one"
    )
    scorer: Scorer = Field(..., description="How to score and rank resulting poses")
    evaluator: BinaryEvaluation = Field(
        ..., description="How to determine how good the results are"
    )
    n_bootstraps: int = Field(1, description="Number of bootstrap replicates to run")

    def run(self, data: DockingDataModel) -> SuccessRate:
        data = self.pose_selector.run(data)
        data_splits: list[DockingDataModel] = self.dataset_split.run(
            data, bootstraps=self.n_bootstraps
        )
        if self.extra_splits:
            if self.extra_splits:
                for split in self.extra_splits:
                    if split is not None:
                        data_splits: list[DockingDataModel] = [
                            split.run(data_) for data_ in data_splits
                        ]
        data_splits = [self.scorer.run(data_) for data_ in data_splits]
        results = [self.evaluator.run(data_) for data_ in data_splits]
        return SuccessRate.from_replicates(results)

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

    @model_validator(mode="after")
    def update_split_level(self) -> Self:
        self.dataset_split.split_level = 0
        if self.extra_splits:
            for level, split in enumerate(self.extra_splits, start=1):
                if split is not None:
                    split.split_level = level
        return self

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
            for model in [self.dataset_split, self.structure_choice, self.scorer]
            if model is not None
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
            if container is not None:
                mydict.update(container.get_records())
        if self.extra_splits:
            for split in self.extra_splits:
                mydict.update(split.get_records())
        return mydict


class Results(BaseModel):
    evaluator: Evaluator
    fraction_good: SuccessRate

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
