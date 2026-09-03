"""
change_metrics.py
=================
Member 3: Bi-Temporal Change Metrics Engine (SATQUERY AI Backend)

Mission: Answers "How much changed?", "What type of change?", and "How severe?"
Calculates quantitative metrics & remote-sensing classification from change masks:
- Change Percentage (% area changed)
- Connected Component Region Analysis (centroid, bounding box, area in px / m² / ha)
- HSV Remote-Sensing Change Taxonomy Classification (Deforestation, Urban Expansion, Water Inundation, Soil Exposure)
- Severity & Hazard Rating (CRITICAL, HIGH, MODERATE, LOW)
- GeoJSON Exporter (converts change contours to GPS coordinates for QGIS/Mapbox)
- Ground-Truth Evaluation Metrics: IoU (Intersection over Union), Precision, Recall, F1 score
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class ChangeRegion:
    """Statistics, spectral classification, and severity rating for a changed region."""
    region_id: int
    area_px: int           # area in pixels
    centroid_x: int        # centroid column
    centroid_y: int        # centroid row
    bbox: tuple            # (x, y, w, h) bounding rectangle
    category: str = "General Change"  # Deforestation, Urban Expansion, Water Inundation, Soil Exposure
    severity: str = "MODERATE"        # CRITICAL, HIGH, MODERATE, LOW
    area_m2: Optional[float] = None     # real-world area in m² if resolution is supplied
    area_ha: Optional[float] = None     # real-world area in hectares if resolution is supplied
    gps_centroid: Optional[Tuple[float, float]] = None  # (lat, lon) if bounds supplied

    def to_dict(self) -> dict:
        d = {
            "region_id": self.region_id,
            "category": self.category,
            "severity": self.severity,
            "area_px": self.area_px,
            "centroid": {"x": self.centroid_x, "y": self.centroid_y},
            "bbox": {"x": self.bbox[0], "y": self.bbox[1],
                     "w": self.bbox[2], "h": self.bbox[3]},
        }
        if self.area_m2 is not None:
            d["area_m2"] = round(self.area_m2, 2)
            d["area_ha"] = round(self.area_ha, 4)
        if self.gps_centroid is not None:
            d["gps_centroid"] = {"lat": round(self.gps_centroid[0], 6), "lon": round(self.gps_centroid[1], 6)}
        return d


class ChangeMetricsCalculator:
    """
    Computes change metrics, per-region statistics, remote-sensing HSV taxonomy,
    severity alerts, GeoJSON exporter, and ground-truth evaluation scores.
    """

    def __init__(self, min_blob_area: int = 25):
        self.min_blob_area = min_blob_area

    def compute_change_percentage(self, mask: np.ndarray) -> float:
        """Percentage of changed pixels (255) vs total pixels."""
        changed = np.count_nonzero(mask)
        total = mask.size
        return (changed / total) * 100.0 if total > 0 else 0.0

    def extract_regions(
        self,
        mask: np.ndarray,
        image_a: Optional[np.ndarray] = None,
        image_b: Optional[np.ndarray] = None,
        resolution_m_per_px: Optional[float] = None,
        top_left_lat_lon: Optional[Tuple[float, float]] = None,
    ) -> List[ChangeRegion]:
        """
        Label connected changed contours, classify change taxonomy via HSV analysis,
        calculate hazard severity rating, and map GPS coordinates.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < self.min_blob_area:
                continue
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0
            x, y, w, h = cv2.boundingRect(c)

            category = self._classify_hsv_change(image_a, image_b, mask, (x, y, w, h))

            area_m2 = None
            area_ha = None
            if resolution_m_per_px and resolution_m_per_px > 0:
                area_m2 = area * (resolution_m_per_px ** 2)
                area_ha = area_m2 / 10000.0

            severity = self._compute_severity(area, area_ha, category)
            gps_centroid = self._pixel_to_gps(cx, cy, top_left_lat_lon, resolution_m_per_px)

            regions.append(ChangeRegion(
                region_id=idx,
                area_px=int(area),
                centroid_x=cx,
                centroid_y=cy,
                bbox=(x, y, w, h),
                category=category,
                severity=severity,
                area_m2=area_m2,
                area_ha=area_ha,
                gps_centroid=gps_centroid,
            ))

        # Sort by area descending so region 0 is the primary change region
        regions.sort(key=lambda r: r.area_px, reverse=True)
        for i, r in enumerate(regions):
            r.region_id = i
        return regions

    def _compute_severity(
        self, area_px: float, area_ha: Optional[float], category: str
    ) -> str:
        """Determines hazard severity rating (CRITICAL, HIGH, MODERATE, LOW)."""
        effective_area = area_ha if area_ha is not None else (area_px / 1000.0)

        is_critical_type = "Deforestation" in category or "Water Inundation" in category or "Flooding" in category

        if effective_area >= 10.0 or (is_critical_type and effective_area >= 2.0):
            return "CRITICAL"
        elif effective_area >= 3.0 or (is_critical_type and effective_area >= 0.5):
            return "HIGH"
        elif effective_area >= 0.5:
            return "MODERATE"
        else:
            return "LOW"

    def _pixel_to_gps(
        self,
        px_x: int,
        px_y: int,
        top_left_lat_lon: Optional[Tuple[float, float]],
        resolution_m_per_px: Optional[float],
    ) -> Optional[Tuple[float, float]]:
        """Converts pixel (x, y) to approximate GPS (latitude, longitude)."""
        if not top_left_lat_lon or not resolution_m_per_px:
            return None

        top_lat, left_lon = top_left_lat_lon

        # Approx conversion: 1 degree lat ~= 111,320 meters; 1 degree lon ~= 111,320 * cos(lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(top_lat))

        delta_lat = (px_y * resolution_m_per_px) / m_per_deg_lat
        delta_lon = (px_x * resolution_m_per_px) / m_per_deg_lon

        return (top_lat - delta_lat, left_lon + delta_lon)

    def export_geojson(
        self,
        regions: List[ChangeRegion],
        top_left_lat_lon: Optional[Tuple[float, float]] = (37.7749, -122.4194),
        resolution_m_per_px: float = 10.0,
    ) -> dict:
        """
        Exports change region contours as a GeoJSON FeatureCollection
        for GIS tools (QGIS, ArcGIS, Mapbox).
        """
        features = []
        for r in regions:
            x, y, w, h = r.bbox
            p1 = self._pixel_to_gps(x, y, top_left_lat_lon, resolution_m_per_px)
            p2 = self._pixel_to_gps(x + w, y, top_left_lat_lon, resolution_m_per_px)
            p3 = self._pixel_to_gps(x + w, y + h, top_left_lat_lon, resolution_m_per_px)
            p4 = self._pixel_to_gps(x, y + h, top_left_lat_lon, resolution_m_per_px)

            if p1 and p2 and p3 and p4:
                # GeoJSON coordinates are [Longitude, Latitude]
                poly_coords = [[
                    [p1[1], p1[0]],
                    [p2[1], p2[0]],
                    [p3[1], p3[0]],
                    [p4[1], p4[0]],
                    [p1[1], p1[0]],  # closed loop
                ]]

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": poly_coords,
                    },
                    "properties": r.to_dict(),
                })

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def _classify_hsv_change(
        self,
        image_a: Optional[np.ndarray],
        image_b: Optional[np.ndarray],
        mask: np.ndarray,
        bbox: tuple,
    ) -> str:
        """
        Remote Sensing HSV Spectral Taxonomy Classifier.
        Analyzes Hue, Saturation, and Value/Brightness shifts between Date 1 and Date 2.
        """
        if image_a is None or image_b is None or image_a.ndim < 3 or image_b.ndim < 3:
            return "Land Cover Change"

        x, y, w, h = bbox
        crop_a = image_a[y:y+h, x:x+w]
        crop_b = image_b[y:y+h, x:x+w]
        crop_mask = mask[y:y+h, x:x+w] > 0

        if not np.any(crop_mask):
            return "Land Cover Change"

        hsv_a = cv2.cvtColor(crop_a, cv2.COLOR_RGB2HSV).astype(float)
        hsv_b = cv2.cvtColor(crop_b, cv2.COLOR_RGB2HSV).astype(float)

        val_diff = float(np.mean(hsv_b[crop_mask, 2]) - np.mean(hsv_a[crop_mask, 2]))
        sat_diff = float(np.mean(hsv_b[crop_mask, 1]) - np.mean(hsv_a[crop_mask, 1]))

        hue_a = hsv_a[crop_mask, 0]
        hue_b = hsv_b[crop_mask, 0]
        is_green_a = np.mean((hue_a >= 30) & (hue_a <= 90)) > 0.3
        is_green_b = np.mean((hue_b >= 30) & (hue_b <= 90)) > 0.3

        if is_green_a and not is_green_b and sat_diff < -10:
            return "Deforestation / Vegetation Loss"
        elif not is_green_a and is_green_b and sat_diff > 10:
            return "Reforestation / Vegetation Growth"
        elif val_diff < -35:
            return "Water Inundation / Flooding"
        elif val_diff > 25:
            return "Urban Expansion / Construction"
        elif sat_diff < -20 and val_diff > 0:
            return "Soil Exposure / Land Clearing"
        else:
            return "Built-up / Surface Change"

    @staticmethod
    def evaluate_ground_truth(
        pred_mask: np.ndarray, target_mask: np.ndarray
    ) -> Dict[str, float]:
        """Evaluate predicted change mask against ground truth (IoU, Precision, Recall, F1)."""
        p = (pred_mask > 0).astype(bool)
        t = (target_mask > 0).astype(bool)

        intersection = np.logical_and(p, t).sum()
        union = np.logical_or(p, t).sum()
        pred_sum = p.sum()
        target_sum = t.sum()

        iou = float(intersection / union) if union > 0 else 1.0
        precision = float(intersection / pred_sum) if pred_sum > 0 else 0.0
        recall = float(intersection / target_sum) if target_sum > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "iou": round(iou, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }
