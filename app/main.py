import os
import json
import numpy as np
import cv2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from terrascope.db import DatabaseManager
from terrascope.geospatial import GeospatialEngine
from terrascope.retrieval import SemanticRetrievalEngine
from terrascope.change_detection import ChangeDetectionPipeline
from terrascope.benchmarks import ResearchBenchmarkEvaluator
from terrascope.data_generator import create_dataset_and_seed_db

app = FastAPI(
    title="TerraScope AI API",
    description="Semantic Satellite Search and Environmental Change Intelligence API",
    version="1.0.0"
)

# Enable CORS for Streamlit / Web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
db = DatabaseManager("terrascope.db")
geo_engine = GeospatialEngine(pixel_resolution_m=10.0)
retrieval_engine = SemanticRetrievalEngine()
change_pipeline = ChangeDetectionPipeline(pixel_resolution_m=10.0)
benchmark_evaluator = ResearchBenchmarkEvaluator()

# Ensure dataset exists on startup
if not os.path.exists("data/bengaluru_tiles"):
    create_dataset_and_seed_db()

# Index tiles in FAISS engine
all_tiles = db.get_all_tiles()
retrieval_engine.build_index(all_tiles)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    year_filter: Optional[int] = None

class ChangeRequest(BaseModel):
    tile_id_t1: str
    tile_id_t2: str
    apply_quality_filter: bool = True

class ReviewRequest(BaseModel):
    tile_id: str
    pair_tile_id: Optional[str] = ""
    search_query: Optional[str] = ""
    detected_change_type: str
    estimated_area_sqm: float
    veg_loss_pct: float
    builtup_gain_pct: float
    confidence_score: float
    quality_warning: str
    analyst_verdict: str
    notes: Optional[str] = ""

@app.get("/health")
def health_check():
    return {"status": "online", "system": "TerraScope AI", "faiss_active": True}

@app.post("/api/init_dataset")
def init_dataset():
    create_dataset_and_seed_db()
    tiles = db.get_all_tiles()
    retrieval_engine.build_index(tiles)
    return {"status": "success", "tiles_loaded": len(tiles)}

@app.get("/api/tiles")
def get_tiles():
    tiles = db.get_all_tiles()
    # Enrich with geospatial bounding boxes
    for t in tiles:
        bounds = geo_engine.compute_tile_bounds(t["lat"], t["lon"])
        t["bounds"] = bounds
    return tiles

@app.get("/api/tile/{tile_id}")
def get_tile(tile_id: str):
    tile = db.get_tile_by_id(tile_id)
    if not tile:
        raise HTTPException(status_code=404, detail="Tile not found")
    tile["bounds"] = geo_engine.compute_tile_bounds(tile["lat"], tile["lon"])
    return tile

@app.post("/api/search")
def search_tiles(req: SearchRequest):
    tiles_list = db.get_all_tiles()
    tiles_dict = {t["tile_id"]: t for t in tiles_list}

    raw_results = retrieval_engine.search_text(
        query=req.query,
        top_k=req.top_k,
        year_filter=req.year_filter,
        tiles_dict=tiles_dict
    )

    enriched_results = []
    for tile_id, score in raw_results:
        if tile_id in tiles_dict:
            t = dict(tiles_dict[tile_id])
            t["similarity_score"] = round(float(score), 4)
            t["bounds"] = geo_engine.compute_tile_bounds(t["lat"], t["lon"])
            t["geojson"] = geo_engine.generate_geojson_polygon(t["bounds"], {"tile_id": tile_id, "score": t["similarity_score"]})
            enriched_results.append(t)

    return {"query": req.query, "count": len(enriched_results), "results": enriched_results}

@app.post("/api/analyze_change")
def analyze_change(req: ChangeRequest):
    tile_t1 = db.get_tile_by_id(req.tile_id_t1)
    tile_t2 = db.get_tile_by_id(req.tile_id_t2)

    if not tile_t1 or not tile_t2:
        raise HTTPException(status_code=404, detail="One or both tiles not found")

    bands_t1 = np.load(tile_t1["bands_path"])
    bands_t2 = np.load(tile_t2["bands_path"])

    rgb_t1 = np.array(cv2.imread(tile_t1["rgb_path"])[:, :, ::-1])
    rgb_t2 = np.array(cv2.imread(tile_t2["rgb_path"])[:, :, ::-1])

    result = change_pipeline.analyze_change(
        bands_t1=bands_t1,
        rgb_t1=rgb_t1,
        tile_meta_t1=tile_t1,
        bands_t2=bands_t2,
        rgb_t2=rgb_t2,
        tile_meta_t2=tile_t2,
        apply_quality_filter=req.apply_quality_filter
    )

    # Convert overlay image to serializable base64 or temporary file path
    overlay_path = f"data/bengaluru_tiles/change_{req.tile_id_t1}_{req.tile_id_t2}.jpg"
    cv2.imwrite(overlay_path, result["change_overlay_rgb"][:, :, ::-1])
    result["overlay_image_path"] = overlay_path

    # Clean non-serializable ndarray items for JSON response
    del result["builtup_expansion_mask"]
    del result["veg_loss_mask"]
    del result["change_overlay_rgb"]
    del result["delta_ndvi"]
    del result["delta_ndbi"]

    result["bounds"] = geo_engine.compute_tile_bounds(tile_t1["lat"], tile_t1["lon"])
    result["geojson"] = geo_engine.generate_geojson_polygon(result["bounds"], {
        "location": tile_t1["location_name"],
        "builtup_gain_ha": result["builtup_gain_ha"],
        "veg_loss_ha": result["veg_loss_ha"],
        "quality": result["quality_grade"]
    })

    return result

@app.get("/api/history")
def get_history():
    return db.get_reviews()

@app.post("/api/record_review")
def record_review(req: ReviewRequest):
    review_id = db.record_review(req.dict())
    return {"status": "success", "review_id": review_id}

@app.get("/api/benchmarks")
def get_benchmarks():
    return {
        "rsicd_retrieval": benchmark_evaluator.evaluate_rsicd_retrieval(),
        "levir_cd_change": benchmark_evaluator.evaluate_levir_cd_change_detection(),
        "quality_ablation": benchmark_evaluator.evaluate_quality_control_ablation()
    }
