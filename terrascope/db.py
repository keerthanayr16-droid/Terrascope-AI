import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

class DatabaseManager:
    def __init__(self, db_path: str = "terrascope.db"):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Table for satellite image metadata
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tiles (
                    tile_id TEXT PRIMARY KEY,
                    location_name TEXT,
                    lat REAL,
                    lon REAL,
                    acquisition_date TEXT,
                    year INTEGER,
                    sensor TEXT,
                    cloud_cover_pct REAL,
                    sun_elevation REAL,
                    spatial_resolution_m REAL,
                    alignment_error_px REAL,
                    rgb_path TEXT,
                    bands_path TEXT,
                    land_cover_summary TEXT,
                    created_at TEXT
                )
            ''')

            # Table for analyst reviews and audit history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS analyst_reviews (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tile_id TEXT,
                    pair_tile_id TEXT,
                    search_query TEXT,
                    detected_change_type TEXT,
                    estimated_area_sqm REAL,
                    veg_loss_pct REAL,
                    builtup_gain_pct REAL,
                    confidence_score REAL,
                    quality_warning TEXT,
                    analyst_verdict TEXT,
                    notes TEXT,
                    timestamp TEXT,
                    FOREIGN KEY (tile_id) REFERENCES tiles (tile_id)
                )
            ''')

            # Table for research evaluation benchmark logs
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evaluation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    benchmark_name TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    category TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()

    def insert_tile(self, tile_data: Dict[str, Any]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO tiles (
                    tile_id, location_name, lat, lon, acquisition_date, year,
                    sensor, cloud_cover_pct, sun_elevation, spatial_resolution_m,
                    alignment_error_px, rgb_path, bands_path, land_cover_summary, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                tile_data["tile_id"],
                tile_data.get("location_name", "Bengaluru Region"),
                tile_data["lat"],
                tile_data["lon"],
                tile_data["acquisition_date"],
                tile_data["year"],
                tile_data.get("sensor", "Sentinel-2 L2A"),
                tile_data.get("cloud_cover_pct", 0.0),
                tile_data.get("sun_elevation", 55.0),
                tile_data.get("spatial_resolution_m", 10.0),
                tile_data.get("alignment_error_px", 0.1),
                tile_data["rgb_path"],
                tile_data.get("bands_path", ""),
                json.dumps(tile_data.get("land_cover_summary", {})),
                datetime.now().isoformat()
            ))
            conn.commit()

    def get_all_tiles(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tiles ORDER BY year DESC, location_name ASC")
            rows = cursor.fetchall()
            result = []
            for r in rows:
                item = dict(r)
                if item.get("land_cover_summary"):
                    item["land_cover_summary"] = json.loads(item["land_cover_summary"])
                result.append(item)
            return result

    def get_tile_by_id(self, tile_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tiles WHERE tile_id = ?", (tile_id,))
            r = cursor.fetchone()
            if r:
                item = dict(r)
                if item.get("land_cover_summary"):
                    item["land_cover_summary"] = json.loads(item["land_cover_summary"])
                return item
            return None

    def record_review(self, review_data: Dict[str, Any]) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO analyst_reviews (
                    tile_id, pair_tile_id, search_query, detected_change_type,
                    estimated_area_sqm, veg_loss_pct, builtup_gain_pct,
                    confidence_score, quality_warning, analyst_verdict, notes, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                review_data["tile_id"],
                review_data.get("pair_tile_id", ""),
                review_data.get("search_query", ""),
                review_data.get("detected_change_type", "Vegetation Loss & Urban Growth"),
                review_data.get("estimated_area_sqm", 0.0),
                review_data.get("veg_loss_pct", 0.0),
                review_data.get("builtup_gain_pct", 0.0),
                review_data.get("confidence_score", 0.95),
                review_data.get("quality_warning", "None"),
                review_data.get("analyst_verdict", "Accepted"),
                review_data.get("notes", ""),
                datetime.now().isoformat()
            ))
            conn.commit()
            return cursor.lastrowid

    def get_reviews(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT r.*, t1.location_name, t1.acquisition_date as date1, t2.acquisition_date as date2
                FROM analyst_reviews r
                LEFT JOIN tiles t1 ON r.tile_id = t1.tile_id
                LEFT JOIN tiles t2 ON r.pair_tile_id = t2.tile_id
                ORDER BY r.id DESC
            ''')
            return [dict(r) for r in cursor.fetchall()]

    def log_evaluation_metric(self, benchmark_name: str, metric_name: str, metric_value: float, category: str = "General"):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO evaluation_metrics (benchmark_name, metric_name, metric_value, category, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (benchmark_name, metric_name, metric_value, category, datetime.now().isoformat()))
            conn.commit()
