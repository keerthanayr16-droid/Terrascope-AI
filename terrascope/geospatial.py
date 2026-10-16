import math
import numpy as np
from typing import Dict, List, Tuple, Optional

class GeospatialEngine:
    """
    Handles geographical coordinate transformations, bounding box math,
    and area conversions for satellite image tiles.
    """
    def __init__(self, pixel_resolution_m: float = 10.0):
        self.pixel_resolution_m = pixel_resolution_m

    def compute_tile_bounds(self, center_lat: float, center_lon: float, width_px: int = 256, height_px: int = 256) -> Dict[str, float]:
        """
        Calculates geographic bounding box (min_lat, min_lon, max_lat, max_lon)
        given a center lat/lon and pixel dimensions at 10m/pixel spatial resolution.
        1 degree lat approx = 111,000 meters.
        1 degree lon approx = 111,000 * cos(lat) meters.
        """
        lat_span_m = height_px * self.pixel_resolution_m
        lon_span_m = width_px * self.pixel_resolution_m

        delta_lat = (lat_span_m / 2.0) / 111000.0
        delta_lon = (lon_span_m / 2.0) / (111000.0 * math.cos(math.radians(center_lat)))

        return {
            "min_lat": round(center_lat - delta_lat, 6),
            "max_lat": round(center_lat + delta_lat, 6),
            "min_lon": round(center_lon - delta_lon, 6),
            "max_lon": round(center_lon + delta_lon, 6),
            "center_lat": round(center_lat, 6),
            "center_lon": round(center_lon, 6)
        }

    def pixel_count_to_area(self, pixel_count: int) -> Tuple[float, float]:
        """
        Converts pixel count to square meters and hectares.
        At 10m/px, 1 pixel = 100 sq meters = 0.01 hectares.
        """
        sq_meters = pixel_count * (self.pixel_resolution_m ** 2)
        hectares = sq_meters / 10000.0
        return sq_meters, hectares

    def generate_geojson_polygon(self, bounds: Dict[str, float], properties: Optional[Dict] = None) -> Dict:
        """
        Generates GeoJSON Feature format for interactive map visualization in Folium/Leaflet.
        """
        coordinates = [[
            [bounds["min_lon"], bounds["min_lat"]],
            [bounds["max_lon"], bounds["min_lat"]],
            [bounds["max_lon"], bounds["max_lat"]],
            [bounds["min_lon"], bounds["max_lat"]],
            [bounds["min_lon"], bounds["min_lat"]]
        ]]

        return {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            },
            "properties": properties or {}
        }
