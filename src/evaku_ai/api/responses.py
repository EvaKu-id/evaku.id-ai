NOT_IMPLEMENTED: dict[int | str, dict[str, str]] = {501: {"description": "Not implemented"}}
DATA_NOT_LOADED: dict[int | str, dict[str, str]] = {
    503: {"description": "Static data is not loaded"},
}
ROUTE_ERRORS: dict[int | str, dict[str, str]] = {
    404: {"description": "Unknown village_id, or param_hash not cached and no simulation given"},
    501: {"description": "Routing on a published run (run_id) is not available yet"},
    503: {"description": "Static data is not loaded"},
}
