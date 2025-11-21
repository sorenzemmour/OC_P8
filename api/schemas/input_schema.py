from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    EXT_SOURCE_3: float = Field(..., description="Score externe 3 (float)")
    EXT_SOURCE_2: float = Field(..., description="Score externe 2 (float)")
    EXT_SOURCE_1: float = Field(..., description="Score externe 1 (float)")
    REG_CITY_NOT_WORK_CITY: int = Field(..., description="1 si le client ne travaille pas dans sa ville de résidence")
    DAYS_ID_PUBLISH: int = Field(..., description="Nombre de jours depuis l'émission du document d'identité")
    DAYS_LAST_PHONE_CHANGE: float = Field(..., description="Nombre de jours depuis le changement de téléphone")
    REGION_RATING_CLIENT: int = Field(..., description="Notation de la région du client")
    REGION_RATING_CLIENT_W_CITY: int = Field(..., description="Notation de la région pondérée par la ville")
    DAYS_EMPLOYED: float = Field(..., description="Nombre de jours d'emploi du client")
    DAYS_BIRTH: int = Field(..., description="Âge du client exprimé en jours (négatif)")

FEATURE_ORDER = [
    "EXT_SOURCE_3",
    "EXT_SOURCE_2",
    "EXT_SOURCE_1",
    "REG_CITY_NOT_WORK_CITY",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_RATING_CLIENT",
    "REGION_RATING_CLIENT_W_CITY",
    "DAYS_EMPLOYED",
    "DAYS_BIRTH"
]

