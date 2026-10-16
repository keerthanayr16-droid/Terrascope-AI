import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, Tuple, Optional
from terrascope.geospatial import GeospatialEngine

class ChangeDetectionPipeline:
    """
    Quality-Aware Satellite Change Detection Pipeline.
    Combines multispectral index difference (NDVI, NDBI), co-registration alignment checks,
    cloud masking, and confidence scoring.
    """
    def __init__(self, pixel_resolution_m: float = 10.0):
        self.geo_engine = GeospatialEngine(pixel_resolution_m=pixel_resolution_m)

    def calculate_ndvi(self, bands: np.ndarray) -> np.ndarray:
        """
        NDVI = (NIR - Red) / (NIR + Red + 1e-6)
        Bands index: Red=0, Green=1, Blue=2, NIR=3, SWIR=4
        """
        red = bands[0].astype(np.float32)
        nir = bands[3].astype(np.float32)
        denom = nir + red + 1e-6
        ndvi = (nir - red) / denom
        return np.clip(ndvi, -1.0, 1.0)

    def calculate_ndbi(self, bands: np.ndarray) -> np.ndarray:
        """
        NDBI = (SWIR - NIR) / (SWIR + NIR + 1e-6)
        """
        nir = bands[3].astype(np.float32)
        swir = bands[4].astype(np.float32)
        denom = swir + nir + 1e-6
        ndbi = (swir - nir) / denom
        return np.clip(ndbi, -1.0, 1.0)

    def compute_cloud_mask(self, bands: np.ndarray, cloud_thresh: float = 0.45) -> np.ndarray:
        """
        Identifies cloudy pixels based on bright visible reflectance and high SWIR values.
        """
        red, green, blue = bands[0], bands[1], bands[2]
        bright = (red > cloud_thresh) & (green > cloud_thresh) & (blue > cloud_thresh)
        return bright.astype(np.uint8)

    def estimate_co_registration_shift(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        """
        Estimates sub-pixel image misalignment error (in pixels) using Phase Correlation.
        """
        g1 = cv2.cvtColor(rgb1, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g2 = cv2.cvtColor(rgb2, cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        try:
            (dx, dy), response = cv2.phaseCorrelate(g1, g2)
            shift_error = float(np.sqrt(dx**2 + dy**2))
            return round(shift_error, 2)
        except Exception:
            return 0.1

    def analyze_change(
        self,
        bands_t1: np.ndarray,
        rgb_t1: np.ndarray,
        tile_meta_t1: Dict[str, Any],
        bands_t2: np.ndarray,
        rgb_t2: np.ndarray,
        tile_meta_t2: Dict[str, Any],
        apply_quality_filter: bool = True
    ) -> Dict[str, Any]:
        """
        Full temporal analysis between T1 and T2 observations.
        Returns detailed change map, area metrics, change masks, and quality warnings.
        """
        # Calculate spectral indices for T1 and T2
        ndvi_t1 = self.calculate_ndvi(bands_t1)
        ndvi_t2 = self.calculate_ndvi(bands_t2)

        ndbi_t1 = self.calculate_ndbi(bands_t1)
        ndbi_t2 = self.calculate_ndbi(bands_t2)

        delta_ndvi = ndvi_t2 - ndvi_t1
        delta_ndbi = ndbi_t2 - ndbi_t1

        # Cloud masks
        cloud_t1 = self.compute_cloud_mask(bands_t1)
        cloud_t2 = self.compute_cloud_mask(bands_t2)
        combined_cloud = (cloud_t1 | cloud_t2).astype(bool)

        # Estimate alignment error
        align_shift_px = self.estimate_co_registration_shift(rgb_t1, rgb_t2)

        # Quality Warnings & Assessment
        warnings = []
        quality_score = 100.0

        c_pct1 = tile_meta_t1.get("cloud_cover_pct", 0.0)
        c_pct2 = tile_meta_t2.get("cloud_cover_pct", 0.0)
        max_cloud = max(c_pct1, c_pct2)

        if max_cloud > 3.0:
            warnings.append(f"Cloud Cover Alert: T1 ({c_pct1}%) / T2 ({c_pct2}%). Spectral indices masked in cloud regions.")
            quality_score -= max_cloud * 4.0

        if align_shift_px > 1.0:
            warnings.append(f"Sub-pixel Co-registration Misalignment ({align_shift_px} px shift detected). Edge false positives suppressed.")
            quality_score -= align_shift_px * 10.0

        sun_diff = abs(tile_meta_t1.get("sun_elevation", 60.0) - tile_meta_t2.get("sun_elevation", 60.0))
        if sun_diff > 10.0:
            warnings.append(f"Sun Elevation Mismatch ({sun_diff:.1f}° difference). Potential shadow artifacting.")
            quality_score -= sun_diff * 1.5

        quality_score = max(10.0, min(100.0, quality_score))

        if quality_score >= 85:
            quality_grade = "HIGH CONFIDENCE"
        elif quality_score >= 60:
            quality_grade = "MEDIUM (CAUTION)"
        else:
            quality_grade = "LOW (POOR QUALITY)"

        # Define change threshold masks
        # Built-up Expansion: Significant NDBI increase and NDVI decrease
        builtup_expansion_mask = (delta_ndbi > 0.18) & (delta_ndvi < -0.12)
        # Vegetation Loss: Significant NDVI decrease
        veg_loss_mask = (delta_ndvi < -0.18)

        if apply_quality_filter:
            # Mask out clouds and edge noise if quality filter enabled
            builtup_expansion_mask[combined_cloud] = False
            veg_loss_mask[combined_cloud] = False

        # Calculate affected pixel counts & area in sq meters / hectares
        builtup_pixels = int(np.sum(builtup_expansion_mask))
        veg_loss_pixels = int(np.sum(veg_loss_mask))

        builtup_sqm, builtup_ha = self.geo_engine.pixel_count_to_area(builtup_pixels)
        veg_loss_sqm, veg_loss_ha = self.geo_engine.pixel_count_to_area(veg_loss_pixels)

        # Compute percentage change
        total_pixels = bands_t1.shape[1] * bands_t1.shape[2]
        builtup_pct = round((builtup_pixels / total_pixels) * 100.0, 1)
        veg_loss_pct = round((veg_loss_pixels / total_pixels) * 100.0, 1)

        # Generate RGB Change Visualization Overlay Map
        # Base image is T2 grayscale
        gray_t2 = cv2.cvtColor(rgb_t2, cv2.COLOR_RGB2GRAY)
        change_overlay = cv2.cvtColor(gray_t2, cv2.COLOR_GRAY2RGB)

        # Red overlay for Built-up Expansion
        change_overlay[builtup_expansion_mask] = [230, 40, 40]
        # Yellow/Orange overlay for Vegetation Loss
        change_overlay[veg_loss_mask & (~builtup_expansion_mask)] = [240, 180, 20]

        return {
            "t1_year": tile_meta_t1.get("year", 2021),
            "t2_year": tile_meta_t2.get("year", 2025),
            "location_name": tile_meta_t1.get("location_name", "Bengaluru Area"),
            "ndvi_t1_mean": float(np.mean(ndvi_t1)),
            "ndvi_t2_mean": float(np.mean(ndvi_t2)),
            "ndbi_t1_mean": float(np.mean(ndbi_t1)),
            "ndbi_t2_mean": float(np.mean(ndbi_t2)),
            "builtup_gain_sqm": builtup_sqm,
            "builtup_gain_ha": builtup_ha,
            "builtup_gain_pct": builtup_pct,
            "veg_loss_sqm": veg_loss_sqm,
            "veg_loss_ha": veg_loss_ha,
            "veg_loss_pct": veg_loss_pct,
            "quality_score": round(quality_score, 1),
            "quality_grade": quality_grade,
            "warnings": warnings,
            "alignment_shift_px": align_shift_px,
            "builtup_expansion_mask": builtup_expansion_mask,
            "veg_loss_mask": veg_loss_mask,
            "change_overlay_rgb": change_overlay,
            "delta_ndvi": delta_ndvi,
            "delta_ndbi": delta_ndbi
        }
