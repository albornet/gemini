from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


class ClinicalEventType(str, Enum):
    """
    Different types of clinical dates
    """
    ADMISSION = "Admission"
    DISCHARGE = "Discharge"
    PROCEDURE = "Procedure"


class ClinicalEventDate(BaseModel):
    """
    Specific date along with its clinical type
    """
    event_date: date = Field(description="The calendar date of the event (YYYY-MM-DD).")
    event_type: ClinicalEventType = Field(description="The type of clinical event.")


class PatientInfoSchema(BaseModel):
    """
    Schema for extracting key patient information
    """
    mRS: int = Field(
        title="Score mRS",
        description="Score sur l'échelle de Rankin Modifiée (0-6)",
        default=-1,
        ge=0,
        le=6,
    )
    # clinical_dates: List[ClinicalEventDate] = Field(
    #     title="Clinical Dates",
    #     description="A list of important clinical dates for the patient.",
    #     default_factory=list,
    # )
