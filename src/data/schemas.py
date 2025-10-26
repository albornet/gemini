from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


# Regular expression pattern for a strict YYYY-MM-DD date format
ISO_8601_DATE_REGEX = r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2][0-9]|3[0-1])$"


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
    event_date: str = Field(pattern=ISO_8601_DATE_REGEX)
    event_type: ClinicalEventType = Field()


class PatientInfoSchema(BaseModel):
    """
    Schema for extracting key patient information
    """
    mRS: int = Field(
        default=-1,
        ge=0,
        le=6,
    )
    # clinical_dates: List[ClinicalEventDate] = Field(
    #     default_factory=list,
    # )
