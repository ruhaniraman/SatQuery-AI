"""
app_member3.py
==============
Member 3: Bi-Temporal & Multi-Temporal Change Understanding Dashboard

Provides an interactive Gradio Web UI for Member 3's engine:
- Drag-and-drop dual satellite image input (Date 1 & Date 2)
- Interactive parameter toggles (Histogram Matching, Otsu Thresholding, Spatial Resolution)
- Multi-Panel visual output (Before, After, JET SSIM Heatmap, Annotated Overlay)
- Interactive Hazard Severity Alert badge & Spectral Taxonomy breakdown
- Downloadable GeoJSON GPS file exporter for QGIS / Mapbox
"""

from __future__ import annotations

import json
import os
import cv2
import numpy as np

from cdvqa_engine import ChangeDetectionEngine

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False


def process_change_detection(
    img_a: np.ndarray,
    img_b: np.ndarray,
    date_a: str,
    date_b: str,
    resolution_m: float,
    use_hist_match: bool,
    use_otsu: bool,
    min_area: int,
):
    """Processes images through Member 3 engine and returns visual panels & JSON audit report."""
    if img_a is None or img_b is None:
        return None, None, None, "Please upload both Date 1 and Date 2 images.", {}

    engine = ChangeDetectionEngine(
        ssim_win_size=7,
        change_threshold=30,
        min_blob_area=min_area,
        use_histogram_matching=use_hist_match,
        use_otsu_thresholding=use_otsu,
    )

    result = engine.detect(
        img_a, img_b,
        date_a=date_a or "Date 1",
        date_b=date_b or "Date 2",
        resolution_m_per_px=resolution_m if resolution_m > 0 else None,
    )

    geojson_data = engine.export_geojson(result, resolution_m_per_px=resolution_m or 10.0)

    # Save GeoJSON file for download
    os.makedirs("demo_output", exist_ok=True)
    geojson_path = "demo_output/change_regions.geojson"
    with open(geojson_path, "w") as f:
        json.dump(geojson_data, f, indent=2)

    summary_markdown = (
        f"### 🚨 Overall Hazard Alert: **{result.overall_severity}**\n"
        f"- **Total Change Area**: `{result.change_percentage:.2f}%`\n"
        f"- **Global SSIM Score**: `{result.ssim_score:.4f}`\n"
        f"- **Confidence Score**: `{result.confidence_score:.4f}`\n"
        f"- **Regions Identified**: `{len(result.change_regions)}`\n\n"
        f"**VLM Explanation Response**:\n> {result.explanation}"
    )

    return result.diff_heatmap, result.overlay_image, summary_markdown, result.to_json(), geojson_path


def launch_dashboard():
    if not GRADIO_AVAILABLE:
        print("[Dashboard] Gradio is not installed. To run the web GUI, install gradio (`pip install gradio`).")
        return

    with gr.Blocks(title="SATQUERY AI - Member 3 Change Understanding") as demo:
        gr.Markdown("# 🛰️ SATQUERY AI: Bi-Temporal & Multi-Temporal Change Understanding")
        gr.Markdown("### Member 3: Specialist Change Detection, Spectral Taxonomy & GeoJSON Exporter")

        with gr.Row():
            with gr.Column():
                img1_input = gr.Image(label="Date 1 (Before Image)", type="numpy")
                date1_input = gr.Textbox(value="2023-01-15", label="Date 1 Capture Label")
            with gr.Column():
                img2_input = gr.Image(label="Date 2 (After Image)", type="numpy")
                date2_input = gr.Textbox(value="2024-06-02", label="Date 2 Capture Label")

        with gr.Accordion("⚙️ Advanced Pipeline Parameters", open=False):
            with gr.Row():
                res_slider = gr.Slider(minimum=0.5, maximum=100.0, value=10.0, step=0.5, label="Spatial Resolution (meters/pixel)")
                min_area_slider = gr.Slider(minimum=5, maximum=500, value=25, step=5, label="Min Blob Filter Area (px)")
            with gr.Row():
                hist_match_check = gr.Checkbox(value=True, label="Illumination / Shadow Normalization (Histogram Matching)")
                otsu_check = gr.Checkbox(value=True, label="Automated Otsu Adaptive Thresholding")

        run_btn = gr.Button("🔍 Detect Changes & Generate Audit", variant="primary")

        with gr.Row():
            heatmap_output = gr.Image(label="JET SSIM Difference Heatmap")
            overlay_output = gr.Image(label="Annotated Change Mask & Taxonomy Overlay")

        summary_output = gr.Markdown(label="Analysis Summary")

        with gr.Row():
            json_output = gr.JSON(label="Structured JSON Audit Payload")
            geojson_file_output = gr.File(label="Download GeoJSON GPS Polygons (.geojson)")

        run_btn.click(
            fn=process_change_detection,
            inputs=[
                img1_input, img2_input, date1_input, date2_input,
                res_slider, hist_match_check, otsu_check, min_area_slider
            ],
            outputs=[heatmap_output, overlay_output, summary_output, json_output, geojson_file_output],
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    launch_dashboard()
