from pydantic import BaseModel, Field
from openff.units import unit


class MeasurementType(BaseModel):
    """
    Base class for defining what kinds of measurements are available.
    """

    class Config:
        arbitrary_types_allowed = True

    name: str = Field(..., description="Name")
    description: str = Field(..., description="Description")
    units: unit = Field(
        None,
        description="Units for the measurement type. If None assumed to be unitless",
    )
    higher_is_better: bool = Field(
        ..., description="Whether higher values of this score are better."
    )
    logarithmic: bool = Field(
        ..., description="Whether this score should be plotted on a log scale."
    )
    predicted: bool = Field(..., description="Whether this score is predicted.")


class ExperimentalMeasurementType(MeasurementType):
    predicted: bool = Field(False, frozen=True)


class PredictedMeasurementType(MeasurementType):
    predicted: bool = Field(True, frozen=True)


RelativeActivity = ExperimentalMeasurementType(
    name="relative_activity",
    description="Relative activity over a control. Scales logarithmically with concentration.",
    logarithmic=True,
    higher_is_better=True,
)

DockingScore = PredictedMeasurementType(
    name="docking_score",
    description="Docking score",
    logarithmic=False,
    higher_is_better=False,
)

pIC50 = ExperimentalMeasurementType(
    name="pIC50",
    description="The negative log of the concentration at which the inhibition is 50%.",
    logarithmic=True,
    higher_is_better=True,
)

IsActive = ExperimentalMeasurementType(
    name="active",
    description="Binary Active or Inactive measurement type. Higher is better because 1 > 0.",
    units=bool,
    logarithmic=False,
    higher_is_better=True,
)
