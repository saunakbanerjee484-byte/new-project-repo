"""
geospatial/inundation_map.py

Flood-extent and depth mapping utilities: converts a depth grid (from
engines.dam_break.SaintVenant2D or a SWMM surcharge summary) into
GIS-ready layers (GeoJSON polygons of inundation extent, depth rasters)
for the Folium/PyDeck Command Center map in app.py.
"""

import numpy as np


def depth_grid_to_geojson(depth_grid: np.ndarray, origin_lat: float, origin_lon: float,
                           cell_size_deg: float, depth_threshold_m: float = 0.1) -> dict:
    """
    Convert a 2D depth array (from SaintVenant2D.run()['h'][-1] or a
    resampled SWMM flood-volume surface) into a GeoJSON FeatureCollection
    of inundated grid cells, each tagged with its depth for choropleth
    styling in Folium.
    """
    features = []
    ny, nx = depth_grid.shape
    for j in range(ny):
        for i in range(nx):
            depth = float(depth_grid[j, i])
            if depth < depth_threshold_m:
                continue
            lat0 = origin_lat + j * cell_size_deg
            lon0 = origin_lon + i * cell_size_deg
            lat1, lon1 = lat0 + cell_size_deg, lon0 + cell_size_deg
            features.append({
                "type": "Feature",
                "properties": {"depth_m": round(depth, 2)},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon0, lat0], [lon1, lat0], [lon1, lat1], [lon0, lat1], [lon0, lat0],
                    ]],
                },
            })
    return {"type": "FeatureCollection", "features": features}


def inundation_extent_stats(depth_grid: np.ndarray, cell_area_m2: float,
                             depth_threshold_m: float = 0.1) -> dict:
    """Summary stats (area, max depth, mean depth of wetted cells) for a dashboard KPI card."""
    wetted = depth_grid[depth_grid >= depth_threshold_m]
    return {
        "inundated_area_km2": float(wetted.size * cell_area_m2 / 1e6),
        "max_depth_m": float(depth_grid.max()) if depth_grid.size else 0.0,
        "mean_wetted_depth_m": float(wetted.mean()) if wetted.size else 0.0,
    }
