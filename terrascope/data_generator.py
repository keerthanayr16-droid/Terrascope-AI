import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import math
from typing import Dict, List, Tuple
from terrascope.db import DatabaseManager

# Bounding box for Bengaluru region
BENGALURU_LAT = 12.9716
BENGALURU_LON = 77.5946

HOTSPOTS = [
    {
        "id": "sarjapur_tech_park",
        "name": "Sarjapur-Bellandur Corridor",
        "lat": 12.9248,
        "lon": 77.6749,
        "desc_2021": "Agricultural fields, tree groves, and sparse residential units.",
        "desc_2025": "Dense IT parks, tech hubs, commercial complexes, and paved parking lots.",
        "veg_change": -0.38, # 38% drop in NDVI
        "built_change": +0.45, # 45% increase in NDBI
        "query_tags": ["built-up land increase", "urban expansion", "vegetation loss", "tech park development", "commercial building"]
    },
    {
        "id": "whitefield_metro",
        "name": "Whitefield-Hoodi Expansion Zone",
        "lat": 12.9856,
        "lon": 77.7289,
        "desc_2021": "Open scrubland, suburban housing, and scattered vegetation.",
        "desc_2025": "Metro station infrastructure, wide asphalt roads, and corporate office campuses.",
        "veg_change": -0.32,
        "built_change": +0.39,
        "query_tags": ["urban growth", "road expansion", "vegetation loss", "built-up area", "metro corridor"]
    },
    {
        "id": "electronic_city_ph2",
        "name": "Electronic City Phase 2",
        "lat": 12.8398,
        "lon": 77.6770,
        "desc_2021": "Open fields, green canopy, and industrial sheds.",
        "desc_2025": "Expanded manufacturing units, large roof footprints, concrete aprons.",
        "veg_change": -0.25,
        "built_change": +0.31,
        "query_tags": ["industrial zone", "built-up land increase", "factory expansion", "vegetation loss"]
    },
    {
        "id": "yelahanka_devanahalli",
        "name": "Yelahanka-Devanahalli Airport Corridor",
        "lat": 13.1402,
        "lon": 77.6083,
        "desc_2021": "Dense green farmland, plantation groves, and rural tracks.",
        "desc_2025": "Logistics hub, warehousing complexes, 6-lane highway, paved yards.",
        "veg_change": -0.42,
        "built_change": +0.50,
        "query_tags": ["airport corridor", "warehousing", "highway construction", "built-up increase", "severe vegetation loss"]
    },
    {
        "id": "varthur_lake_basin",
        "name": "Varthur-Gunjur Lake Buffer Zone",
        "lat": 12.9431,
        "lon": 77.7471,
        "desc_2021": "Wetland vegetation, marshy land, green canopy, and natural drainage.",
        "desc_2025": "Apartment block construction site, earthworks, loss of wetland green cover.",
        "veg_change": -0.48,
        "built_change": +0.52,
        "query_tags": ["wetland encroachment", "lake buffer built-up", "vegetation decrease", "apartment construction"]
    },
    {
        "id": "hebbal_manyata",
        "name": "Hebbal Lake & Manyata Infill",
        "lat": 13.0358,
        "lon": 77.6192,
        "desc_2021": "Lake waters, urban park greens, surrounding tree canopy.",
        "desc_2025": "Flanking flyover ramps, commercial buildings, slight shoreline built-up.",
        "veg_change": -0.15,
        "built_change": +0.18,
        "query_tags": ["urban infill", "flyover construction", "moderate vegetation change", "lake vicinity"]
    },
    {
        "id": "prr_kengeri",
        "name": "Peripheral Ring Road - Kengeri Section",
        "lat": 12.9081,
        "lon": 77.4839,
        "desc_2021": "Agricultural farmland, coconut groves, and rural paths.",
        "desc_2025": "Cleared land strip for ring road, paved carriageways, new residential layouts.",
        "veg_change": -0.35,
        "built_change": +0.40,
        "query_tags": ["ring road", "infrastructure corridor", "agricultural land loss", "built-up growth"]
    },
    {
        "id": "bannerghatta_south",
        "name": "Bannerghatta Suburban Sprawl",
        "lat": 12.8250,
        "lon": 77.5920,
        "desc_2021": "Dense forest edge vegetation, hilly green cover.",
        "desc_2025": "Gated township layouts, cleared hillsides, villa plots.",
        "veg_change": -0.29,
        "built_change": +0.33,
        "query_tags": ["suburban sprawl", "gated community", "hillside clearance", "vegetation loss"]
    }
]

def generate_synthetic_sentinel_tile(
    width: int = 256,
    height: int = 256,
    year: int = 2021,
    veg_density: float = 0.6,
    builtup_density: float = 0.2,
    cloud_pct: float = 0.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a 5-band synthetic Sentinel-2 multispectral tile & corresponding RGB display image.
    Bands:
      0: Red (B4)
      1: Green (B3)
      2: Blue (B2)
      3: NIR (B8)
      4: SWIR (B11)
    """
    np.random.seed(seed)
    
    # Create base terrain grid (coordinate space)
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xx, yy = np.meshgrid(x, y)

    # Low frequency noise for landform regions
    base_field = np.sin(xx * 6 + np.pi/4) * np.cos(yy * 6) + np.sin((xx+yy)*4)
    base_field = (base_field - base_field.min()) / (base_field.max() - base_field.min())

    # Vegetation mask (high NIR, low Red)
    veg_mask = (base_field < veg_density).astype(np.float32)
    # Add random clusters for trees/crops
    veg_noise = np.random.uniform(0.7, 1.0, (height, width)).astype(np.float32)
    veg_mask = veg_mask * veg_noise

    # Built-up mask (high Red, high SWIR, moderate Green/Blue)
    built_mask = (base_field >= (1.0 - builtup_density)).astype(np.float32)
    # Add rectangular grid pattern to simulate buildings/roads
    grid_pattern = ((np.sin(xx * 40) > 0.8) | (np.cos(yy * 40) > 0.8)).astype(np.float32)
    built_mask = np.clip(built_mask + grid_pattern * 0.4, 0.0, 1.0)

    # Initialize 5 bands
    # Values mapped to Sentinel-2 reflectance (0.0 to 1.0)
    red = np.full((height, width), 0.08, dtype=np.float32)
    green = np.full((height, width), 0.12, dtype=np.float32)
    blue = np.full((height, width), 0.06, dtype=np.float32)
    nir = np.full((height, width), 0.15, dtype=np.float32)
    swir = np.full((height, width), 0.10, dtype=np.float32)

    # Vegetation spectral signature: low Red (absorbs), high NIR (reflects), high Green
    red = np.where(veg_mask > 0.3, 0.03 + np.random.uniform(0.0, 0.02, (height, width)), red)
    green = np.where(veg_mask > 0.3, 0.18 + np.random.uniform(0.0, 0.05, (height, width)), green)
    blue = np.where(veg_mask > 0.3, 0.03 + np.random.uniform(0.0, 0.02, (height, width)), blue)
    nir = np.where(veg_mask > 0.3, 0.55 + np.random.uniform(0.0, 0.15, (height, width)), nir)
    swir = np.where(veg_mask > 0.3, 0.12 + np.random.uniform(0.0, 0.04, (height, width)), swir)

    # Built-up spectral signature: high Red, high SWIR, high Blue/Green (concrete/asphalt/roofs)
    red = np.where(built_mask > 0.3, 0.28 + np.random.uniform(0.0, 0.10, (height, width)), red)
    green = np.where(built_mask > 0.3, 0.26 + np.random.uniform(0.0, 0.08, (height, width)), green)
    blue = np.where(built_mask > 0.3, 0.24 + np.random.uniform(0.0, 0.08, (height, width)), blue)
    nir = np.where(built_mask > 0.3, 0.22 + np.random.uniform(0.0, 0.06, (height, width)), nir)
    swir = np.where(built_mask > 0.3, 0.38 + np.random.uniform(0.0, 0.12, (height, width)), swir)

    # Add cloud noise if cloud_pct > 0
    if cloud_pct > 0:
        cloud_map = (np.sin(xx * 12 + seed) * np.cos(yy * 12 + seed) > (1.0 - cloud_pct/100.0 * 1.5)).astype(np.float32)
        cloud_map = Image.fromarray((cloud_map * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=8))
        cloud_arr = np.array(cloud_map, dtype=np.float32) / 255.0
        
        red = red * (1 - cloud_arr) + 0.85 * cloud_arr
        green = green * (1 - cloud_arr) + 0.85 * cloud_arr
        blue = blue * (1 - cloud_arr) + 0.90 * cloud_arr
        nir = nir * (1 - cloud_arr) + 0.80 * cloud_arr
        swir = swir * (1 - cloud_arr) + 0.75 * cloud_arr

    multispectral_stack = np.stack([red, green, blue, nir, swir], axis=0) # (5, H, W)

    # Generate true-color RGB display image (0-255) with contrast boost
    rgb = np.stack([red, green, blue], axis=-1)
    rgb = np.clip(rgb * 3.5 * 255, 0, 255).astype(np.uint8)

    return multispectral_stack, rgb

def create_dataset_and_seed_db(data_dir: str = "data", db_path: str = "terrascope.db"):
    """
    Creates dataset folders, generates Sentinel-2 GeoTIFF tiles & RGB thumbnails for 2021 & 2025,
    and populates the SQLite database.
    """
    tiles_dir = os.path.join(data_dir, "bengaluru_tiles")
    levir_dir = os.path.join(data_dir, "levir_cd")
    rsicd_dir = os.path.join(data_dir, "rsicd")
    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(levir_dir, exist_ok=True)
    os.makedirs(rsicd_dir, exist_ok=True)

    db = DatabaseManager(db_path)

    print(f"Generating Sentinel-2 dataset for {len(HOTSPOTS)} Bengaluru study hotspots...")

    for idx, spot in enumerate(HOTSPOTS):
        # 2021 Observation (T1)
        tile_id_2021 = f"{spot['id']}_2021"
        ms_2021, rgb_2021 = generate_synthetic_sentinel_tile(
            width=256, height=256, year=2021,
            veg_density=0.70, builtup_density=0.15,
            cloud_pct=2.0 if idx == 4 else 0.0, # Introduce slight cloud in Varthur for quality warning demo
            seed=100 + idx
        )
        rgb_path_2021 = os.path.join(tiles_dir, f"{tile_id_2021}_rgb.jpg")
        bands_path_2021 = os.path.join(tiles_dir, f"{tile_id_2021}_bands.npy")
        Image.fromarray(rgb_2021).save(rgb_path_2021, quality=92)
        np.save(bands_path_2021, ms_2021)

        db.insert_tile({
            "tile_id": tile_id_2021,
            "location_name": spot["name"],
            "lat": spot["lat"],
            "lon": spot["lon"],
            "acquisition_date": "2021-03-15",
            "year": 2021,
            "sensor": "Sentinel-2A L2A",
            "cloud_cover_pct": 2.0 if idx == 4 else 0.1,
            "sun_elevation": 62.4,
            "spatial_resolution_m": 10.0,
            "alignment_error_px": 0.1,
            "rgb_path": rgb_path_2021,
            "bands_path": bands_path_2021,
            "land_cover_summary": {
                "description": spot["desc_2021"],
                "veg_pct": 68.5,
                "builtup_pct": 18.2,
                "water_pct": 5.1,
                "bare_soil_pct": 8.2,
                "query_tags": spot["query_tags"]
            }
        })

        # 2025 Observation (T2)
        tile_id_2025 = f"{spot['id']}_2025"
        # 2025 has increased built-up and decreased vegetation
        ms_2025, rgb_2025 = generate_synthetic_sentinel_tile(
            width=256, height=256, year=2025,
            veg_density=max(0.2, 0.70 + spot["veg_change"]),
            builtup_density=min(0.8, 0.15 + spot["built_change"]),
            cloud_pct=4.5 if idx == 4 else 0.2,
            seed=200 + idx
        )
        rgb_path_2025 = os.path.join(tiles_dir, f"{tile_id_2025}_rgb.jpg")
        bands_path_2025 = os.path.join(tiles_dir, f"{tile_id_2025}_bands.npy")
        Image.fromarray(rgb_2025).save(rgb_path_2025, quality=92)
        np.save(bands_path_2025, ms_2025)

        db.insert_tile({
            "tile_id": tile_id_2025,
            "location_name": spot["name"],
            "lat": spot["lat"],
            "lon": spot["lon"],
            "acquisition_date": "2025-03-18",
            "year": 2025,
            "sensor": "Sentinel-2B L2A",
            "cloud_cover_pct": 4.5 if idx == 4 else 0.2,
            "sun_elevation": 61.8,
            "spatial_resolution_m": 10.0,
            "alignment_error_px": 0.2,
            "rgb_path": rgb_path_2025,
            "bands_path": bands_path_2025,
            "land_cover_summary": {
                "description": spot["desc_2025"],
                "veg_pct": round(68.5 + spot["veg_change"]*100, 1),
                "builtup_pct": round(18.2 + spot["built_change"]*100, 1),
                "water_pct": 4.8,
                "bare_soil_pct": 12.0,
                "query_tags": spot["query_tags"]
            }
        })

    # Seed initial analyst reviews into DB for audit log history demonstration
    db.record_review({
        "tile_id": "sarjapur_tech_park_2025",
        "pair_tile_id": "sarjapur_tech_park_2021",
        "search_query": "built-up land increased vegetation decreased Bengaluru",
        "detected_change_type": "Vegetation Loss -> Built-up Expansion",
        "estimated_area_sqm": 294800.0,
        "veg_loss_pct": -38.0,
        "builtup_gain_pct": +45.0,
        "confidence_score": 0.94,
        "quality_warning": "None (High Quality)",
        "analyst_verdict": "Approved (Confirmed Growth)",
        "notes": "Verified against high-res optical imagery. Construction of 3 new IT park blocks."
    })

    db.record_review({
        "tile_id": "yelahanka_devanahalli_2025",
        "pair_tile_id": "yelahanka_devanahalli_2021",
        "search_query": "highway logistics warehousing vegetation loss",
        "detected_change_type": "Farmland Clearing -> Infrastructure",
        "estimated_area_sqm": 328000.0,
        "veg_loss_pct": -42.0,
        "builtup_gain_pct": +50.0,
        "confidence_score": 0.96,
        "quality_warning": "None (High Quality)",
        "analyst_verdict": "Approved (Infrastructure)",
        "notes": "6-lane corridor expansion and logistics park development."
    })

    # Generate LEVIR-CD sample building change pairs
    generate_levir_cd_samples(levir_dir)

    # Generate RSICD evaluation benchmark samples
    generate_rsicd_samples(rsicd_dir)

    print("Dataset & DB seeding completed successfully.")

def generate_levir_cd_samples(output_dir: str):
    """
    Generates sample building change detection pairs for LEVIR-CD benchmark testing.
    """
    for i in range(1, 6):
        t1_img = np.zeros((256, 256, 3), dtype=np.uint8) + 80
        t2_img = np.zeros((256, 256, 3), dtype=np.uint8) + 80
        gt_mask = np.zeros((256, 256), dtype=np.uint8)

        # Draw green fields in T1
        t1_img[30:220, 30:220] = [40, 140, 50]
        t2_img[30:220, 30:220] = [40, 140, 50]

        # Draw new buildings in T2
        # Building 1
        t2_img[50:110, 50:110] = [210, 200, 190]
        gt_mask[50:110, 50:110] = 255

        # Building 2
        t2_img[140:200, 130:210] = [195, 185, 175]
        gt_mask[140:200, 130:210] = 255

        Image.fromarray(t1_img).save(os.path.join(output_dir, f"pair_{i}_A.png"))
        Image.fromarray(t2_img).save(os.path.join(output_dir, f"pair_{i}_B.png"))
        Image.fromarray(gt_mask).save(os.path.join(output_dir, f"pair_{i}_label.png"))

def generate_rsicd_samples(output_dir: str):
    """
    Generates metadata and text queries for RSICD retrieval evaluation.
    """
    rsicd_data = [
        {"id": "rsicd_01", "caption": "A dense urban area with commercial buildings and roads", "category": "built_up"},
        {"id": "rsicd_02", "caption": "Agricultural fields with green vegetation and irrigation channels", "category": "farmland"},
        {"id": "rsicd_03", "caption": "Industrial zone with large roofs and concrete lots", "category": "industrial"},
        {"id": "rsicd_04", "caption": "River flowing through a green forested area", "category": "water_veg"},
        {"id": "rsicd_05", "caption": "Highway interchange connecting urban residential districts", "category": "infrastructure"}
    ]
    with open(os.path.join(output_dir, "rsicd_eval.json"), "w") as f:
        json.dump(rsicd_data, f, indent=2)

if __name__ == "__main__":
    create_dataset_and_seed_db()
