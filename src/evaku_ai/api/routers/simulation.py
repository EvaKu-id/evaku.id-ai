from fastapi import APIRouter, HTTPException, status

from evaku_ai.api.responses import NOT_IMPLEMENTED
from evaku_ai.api.schemas import SimulationRequest, SimulationResponse

router = APIRouter(tags=["simulation"])


@router.post("/simulation", response_model=SimulationResponse, responses=NOT_IMPLEMENTED)
def simulate(request: SimulationRequest) -> SimulationResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Simulation not implemented")
