import logging
from enum import StrEnum

from evaku_ai.pipeline.stages import export, forecast, notify, radar, write

log = logging.getLogger(__name__)


class Mode(StrEnum):
    REPLAY = "replay"
    LIVE = "live"


def run(mode: Mode) -> None:
    log.info("pipeline start (%s)", mode)
    forecast_results = forecast.run(mode)
    radar_results = radar.run(mode)
    run_ids = write.run(forecast_results, radar_results)
    export.run(run_ids)
    notify.run(run_ids)
    log.info("pipeline done: %s", run_ids)
