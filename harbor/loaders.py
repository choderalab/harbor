from pydantic import BaseModel, Field, field_validator
from pathlib import Path


class Loader(BaseModel):
    """
    Base class for a loader
    """

    filename: str = Field(..., description="Path to the file")

    @field_validator("filename")
    @classmethod
    def check_file_exists(cls, v):
        if not Path(v).exists():
            raise ValueError(f"File {v} does not exist")
        return v
