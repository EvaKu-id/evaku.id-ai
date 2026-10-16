from enum import IntEnum, StrEnum


class RoadCondition(IntEnum):
    SAFE = 0
    LIMITED = 1
    CUT = 2
    UNKNOWN = 3


class NodeImpact(IntEnum):
    NORMAL = 0
    DELAYED = 1
    TRUCK_ONLY = 2
    CUT = 3


class VillageAccessStatus(StrEnum):
    REACHABLE = "reachable"
    PARTIALLY_ISOLATED = "partially_isolated"
    FULLY_ISOLATED = "fully_isolated"


class Vehicle(StrEnum):
    CAR = "car"
    TRUCK = "truck"


class RouteMode(StrEnum):
    LAND = "land"
    LAND_BOAT = "land_boat"


class RunType(StrEnum):
    RADAR = "radar"
    FORECAST = "forecast"
    SIMULATION = "simulation"


class RunStatus(StrEnum):
    DRAFT = "draft"
    PUBLISHED = "published"


class FacilityType(StrEnum):
    HEALTH_CENTER = "health_center"
    HOSPITAL = "hospital"
    SCHOOL = "school"
    PLACE_OF_WORSHIP = "place_of_worship"
    VILLAGE_OFFICE = "village_office"
    FIELD = "field"


class LocationSource(StrEnum):
    OSM = "osm"
    BPBD = "bpbd"
    OVERTURE = "overture"


class HubType(StrEnum):
    REGENT_OFFICE = "regent_office"
    BPBD_OFFICE = "bpbd_office"
    AIRPORT = "airport"
    PORT = "port"
    COMMAND_POST = "command_post"
    COMMAND_POST_ESTIMATED = "command_post_estimated"
