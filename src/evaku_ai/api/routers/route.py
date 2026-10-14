from fastapi import APIRouter, HTTPException, status

from evaku_ai.api.responses import NOT_IMPLEMENTED
from evaku_ai.api.schemas import RouteRequest, RouteResponse

router = APIRouter(tags=["route"])


@router.post("/route", response_model=RouteResponse, responses=NOT_IMPLEMENTED)
def route(request: RouteRequest) -> RouteResponse:
    raise HTTPException(status.HTTP_501_NOT_IMPLEMENTED, "Routing not implemented")
