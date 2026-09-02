"""
demo_cdvqa.py
=============
Member 3: Bi-Temporal Change Detection 4-Panel Visualization Demo CLI

Usage:
  python demo_cdvqa.py --image1 path/to/date1.png --image2 path/to/date2.png --out_dir ./demo_output

If no images are supplied, generates synthetic satellite before/after test images.
Renders a high-resolution 4-Panel composite graphic:
  [ Panel 1: Image A (Before)   | Panel 2: Image B (After)      ]
  [ Panel 3: JET SSIM Heatmap   | Panel 4: Annotated Overlay    ]
"""

from __future__ import annotations

import argparse
import json
import os
import cv2
import numpy as np

from cdvqa_engine import ChangeDetectionEngine


def make_synthetic_pair():
    """Generates synthetic satellite before/after images representing deforestation & urban construction."""
    # Before: green forest area with a small river
    img_a = np.zeros((300, 300, 3), dtype=np.uint8)
    img_a[:, :] = (35, 110, 45)  # Dense green forest

    # Blue river winding through
    cv2.polylines(img_a, [np.array([[0, 150], [100, 160], [200, 140], [300, 150]])], isClosed=False, color=(180, 120, 40), thickness=15)

    # After: Deforestation (top-left patch turned soil/brown) & Urban expansion (center bright concrete)
    img_b = img_a.copy()

    # Deforestation patch (soil brown)
    cv2.rectangle(img_b, (20, 20), (120, 110), (50, 90, 140), thickness=-1)

    # Urban construction (bright concrete gray/white)
    cv2.rectangle(img_b, (160, 160), (270, 270), (210, 210, 215), thickness=-1)
    cv2.rectangle(img_b, (180, 180), (210, 210), (100, 100, 240), thickness=-1)  # Red roof building

    return img_a, img_b


def create_4panel_visualization(img_a: np.ndarray, img_b: np.ndarray, heatmap: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Combines 4 visual outputs into a clean 2x2 grid with titles."""
    h, w = img_a.shape[:2]

    # Convert grayscale to RGB if needed
    if img_a.ndim == 2:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_GRAY2RGB)
    if img_b.ndim == 2:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_GRAY2RGB)

    panel1 = img_a.copy()
    panel2 = img_b.copy()
    panel3 = heatmap.copy()
    panel4 = overlay.copy()

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    def add_header(img, title):
        cv2.rectangle(img, (0, 0), (img.shape[1], 24), (20, 20, 20), thickness=-1)
        cv2.putText(img, title, (8, 16), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return img

    p1 = add_header(panel1, "1. Date 1 (Before)")
    p2 = add_header(panel2, "2. Date 2 (After)")
    p3 = add_header(panel3, "3. SSIM Difference Heatmap (JET)")
    p4 = add_header(panel4, "4. Change Mask & Taxonomy Overlay")

    top_row = np.hstack([p1, p2])
    bottom_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bottom_row])

    return grid


def main():
    parser = argparse.ArgumentParser(description="Member 3 Bi-Temporal Change Detection 4-Panel Demo")
    parser.add_argument("--image1", type=str, default=None, help="Path to Date 1 (Before) image")
    parser.add_argument("--image2", type=str, default=None, help="Path to Date 2 (After) image")
    parser.add_argument("--out_dir", type=str, default="./demo_output", help="Output directory")
    parser.add_argument("--resolution", type=float, default=10.0, help="Resolution in meters/pixel")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.image1 and args.image2:
        img_a = cv2.cvtColor(cv2.imread(args.image1), cv2.COLOR_BGR2RGB)
        img_b = cv2.cvtColor(cv2.imread(args.image2), cv2.COLOR_BGR2RGB)
        date_a, date_b = "Date 1 (File)", "Date 2 (File)"
    else:
        print("[Demo] No image paths supplied. Generating synthetic bi-temporal satellite pair...")
        img_a, img_b = make_synthetic_pair()
        date_a, date_b = "2023-01-15 (Synthetic)", "2024-06-02 (Synthetic)"

    engine = ChangeDetectionEngine(
        ssim_win_size=7,
        change_threshold=30,
        min_blob_area=25,
        use_histogram_matching=True,
        use_otsu_thresholding=True,
    )

    print("[Demo] Running Member 3 Change Detection Engine...")
    result = engine.detect(
        img_a, img_b,
        date_a=date_a,
        date_b=date_b,
        resolution_m_per_px=args.resolution,
    )

    print("\n=======================================================")
    print("           BI-TEMPORAL CHANGE ANALYSIS REPORT          ")
    print("=======================================================")
    print(f"Change Percentage:   {result.change_percentage:.2f}%")
    print(f"SSIM Score:          {result.ssim_score:.4f}")
    print(f"Confidence Score:    {result.confidence_score:.4f}")
    print(f"Regions Detected:    {len(result.change_regions)}")
    for r in result.change_regions:
        area_str = f"{r.area_ha:.4f} ha ({r.area_px} px)" if r.area_ha else f"{r.area_px} px"
        print(f"  - Region #{r.region_id} [{r.category}]: {area_str} at centroid ({r.centroid_x}, {r.centroid_y})")
    print(f"\nVLM Prompt Generated:\n  {result.prompt_sent_to_vlm}")
    print(f"\nExplanation Response:\n  {result.explanation}")
    print("=======================================================\n")

    # Save visual outputs
    saved_paths = engine.save_outputs(result, out_dir=args.out_dir, prefix="demo")

    # Create 4-panel visual comparison graphic
    four_panel = create_4panel_visualization(img_a, img_b, result.diff_heatmap, result.overlay_image)
    grid_path = os.path.join(args.out_dir, "demo_4panel_composite.png")
    cv2.imwrite(grid_path, cv2.cvtColor(four_panel, cv2.COLOR_RGB2BGR))
    saved_paths["4panel_composite"] = grid_path

    # Save JSON analysis
    json_path = os.path.join(args.out_dir, "demo_analysis.json")
    with open(json_path, "w") as f:
        json.dump(result.to_json(), f, indent=2)
    saved_paths["analysis_json"] = json_path

    print(f"Visual evidence & 4-panel graphic saved to: {args.out_dir}")
    for k, v in saved_paths.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
