"""
utils/registry.py

Station ID -> geospatial metadata lookup (river, district, lat/lon,
datum, associated engine params). Central place other modules query
instead of hardcoding station identifiers.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class StationMeta:
    station_id: str
    name: str
    river: str
    district: str
    lat: float
    lon: float
    bed_level_m: Optional[float] = None   # gauge-zero / bed datum, for rating-curve h0
    manning_n: float = 0.030


# Seed registry -- extend with the full CWC/WB-SWD station list.
STATIONS: dict[str, StationMeta] = {
    "DURGAPUR_BARRAGE": StationMeta(
        station_id="DURGAPUR_BARRAGE", name="Durgapur Barrage",
        river="Damodar", district="Bardhaman", lat=23.5204, lon=87.3119,
        bed_level_m=52.0,
    ),
    "PANCHET_DAM": StationMeta(
        station_id="PANCHET_DAM", name="Panchet Dam (DVC-SCADA)",
        river="Damodar", district="Purba Bardhaman", lat=23.6300, lon=86.7300,
        bed_level_m=120.0,
    ),
}


def get_station(station_id: str) -> Optional[StationMeta]:
    return STATIONS.get(station_id)


def stations_in_district(district: str) -> list[StationMeta]:
    return [s for s in STATIONS.values() if s.district == district]
