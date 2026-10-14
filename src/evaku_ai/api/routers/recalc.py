from fastapi import APIRouter, HTTPException, status

from evaku_ai.api.responses import NOT_IMPLEMENTED
from evaku_ai.api.schemas import RecalcRequest, SimulationResponse

router = APIRouter(tags=["recalc"])


@router.post("/recalc", response_model=SimulationResponse, responses=NOT_IMPLEMENTED)
def recalc(request: RecalcRequest) -> SimulationResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Recalc not implemented")
