from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException
import pandas as pd

from mibici_mlops.api.schemas import (
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

app = FastAPI(
    title="MiBici Equipo 42 API",
    description="Servicio REST de inferencia del proyecto MiBici",
    version="1.0.0",
)


def get_model():
    from mibici_mlops.api.model_service import get_model as load_model

    return load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(
    request: PredictionRequest,
    model: Annotated[Any, Depends(get_model)],
) -> PredictionResponse:
    data = pd.DataFrame([request.model_dump()]).astype("float32")

    try:
        prediction = model.predict(data)
        value = float(prediction[0])
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="No fue posible ejecutar la predicción.",
        ) from exc

    return PredictionResponse(prediction=value)