import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class ChangeDetectionResult:
    explanation: str
    overlay_image: np.ndarray
    major_regions_detected: int

class ChangeDetectionEngine:
    def detect(self, image_a: np.ndarray, image_b: np.ndarray, vlm_fn, user_query: str = "") -> ChangeDetectionResult:

        # VLM Semantic Analysis (Side-by-Side Stitching)
        stitched_image = np.concatenate((image_a, image_b), axis=1)

        prompt = (
            "This is a side-by-side satellite image of the same location. The left half is the historical past, and the right half is the present. "
            "Write a short paragraph describing the changes from the left half to the right half. "
            "You must specifically state what happened to the central water body, and what happened to the density of the surrounding buildings."
        )
        if user_query:
            prompt += f" Ensure you also address this query: {user_query}"

        # We pass ONLY the single stitched image to the VLM
        explanation = vlm_fn(prompt, stitched_image)

        # Calibrated OpenCV Spatial Differencing + Cloud Masking]
        gray1 = cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY)
        blur1 = cv2.GaussianBlur(gray1, (21, 21), 0)
        blur2 = cv2.GaussianBlur(gray2, (21, 21), 0)
        
        diff = cv2.absdiff(blur1, blur2)

        # Reintegrated Cloud Masking
        _, cloud_mask1 = cv2.threshold(gray1, 200, 255, cv2.THRESH_BINARY)
        _, cloud_mask2 = cv2.threshold(gray2, 200, 255, cv2.THRESH_BINARY)
        combined_clouds = cv2.bitwise_or(cloud_mask1, cloud_mask2)
        cloud_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
        combined_clouds = cv2.dilate(combined_clouds, cloud_kernel, iterations=2)
        
        # Zero out difference map where clouds exist
        diff[combined_clouds == 255] = 0

        # Strict Threshold & Noise Filtering 
        _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
        noise_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, noise_kernel, iterations=2)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel, iterations=2)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Prepare the BGR evidence artifact
        evidence_img = cv2.cvtColor(image_b, cv2.COLOR_RGB2BGR)
        box_count = 0
        h1, w1 = image_a.shape[:2]
        total_area = w1 * h1

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for contour in contours:
                area = cv2.contourArea(contour)
                x, y, w, h = cv2.boundingRect(contour)
                coverage_ratio = (w * h) / total_area
                
                if area > 15000 and coverage_ratio < 0.75:
                    cv2.rectangle(evidence_img, (x, y), (x + w, y + h), (0, 255, 0), thickness=3)
                    
                    # Clamp the text coordinates so they never render off-screen
                    text_x = max(x, 10)
                    text_y = max(y - 10, 25)
                    
                    cv2.putText(
                        evidence_img, "Major Change", (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA
                    )
                    box_count += 1

        return ChangeDetectionResult(
            explanation=explanation,
            overlay_image=evidence_img,
            major_regions_detected=box_count
        )