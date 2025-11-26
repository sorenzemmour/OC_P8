from pydantic import BaseModel, Field
from typing import Optional

class CustomerFeatures(BaseModel):
    EXT_SOURCE_3: Optional[float] = Field(None, description="Score externe 3 (float)")
    EXT_SOURCE_2: Optional[float] = Field(None, description="Score externe 2 (float)")
    EXT_SOURCE_1: Optional[float] = Field(None, description="Score externe 1 (float)")
    REG_CITY_NOT_WORK_CITY: Optional[int] = Field(None, description="1 si le client ne travaille pas dans sa ville de résidence")
    DAYS_ID_PUBLISH: Optional[int] = Field(None, description="Nombre de jours depuis l'émission du document d'identité")
    DAYS_LAST_PHONE_CHANGE: Optional[float] = Field(None, description="Nombre de jours depuis le changement de téléphone")
    REGION_RATING_CLIENT: Optional[int] = Field(None, description="Notation de la région du client")
    REGION_RATING_CLIENT_W_CITY: Optional[int] = Field(None, description="Notation de la région pondérée par la ville")
    DAYS_EMPLOYED: Optional[float] = Field(None, description="Nombre de jours d'emploi du client")
    DAYS_BIRTH: Optional[int] = Field(None, description="Âge du client exprimé en jours (négatif)")


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

