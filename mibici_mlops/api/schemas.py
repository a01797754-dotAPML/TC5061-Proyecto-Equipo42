from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hora_dia: int = Field(ge=0, le=23)
    dia_semana: int = Field(ge=0, le=6)
    es_fin_semana: int = Field(ge=0, le=1)
    mes: int = Field(ge=1, le=12)

    capacidad: float = Field(gt=0)

    salidas_lag_1: float = Field(ge=0)
    salidas_lag_2: float = Field(ge=0)
    salidas_lag_24: float = Field(ge=0)
    salidas_lag_168: float = Field(ge=0)

    salidas_media_24h: float = Field(ge=0)
    salidas_media_4sem: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: float


class HealthResponse(BaseModel):
    status: str