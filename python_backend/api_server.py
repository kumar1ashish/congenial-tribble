"""
Flask API Server for Palm Line Detection Pipeline

Exposes the Neuro-Symbolic palm line detection as a REST API
for integration with the React frontend.
"""

import os
import io
import base64
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import cv2

from palm_line_detector import PalmLineDetectionPipeline, LineDetectionResult
from palm_reading_bridge import (
    PalmReadingBridge,
    HandLandmarks,
    create_palm_reading
)

app = Flask(__name__)
CORS(app)

# Initialize pipeline (lazy loaded on first request)
pipeline = None


def get_pipeline():
    """Get or initialize the detection pipeline."""
    global pipeline
    if pipeline is None:
        print("Initializing palm line detection pipeline...")
        pipeline = PalmLineDetectionPipeline(
            frangi_sigmas=(1.0, 1.5, 2.0, 2.5, 3.0),
            device="cuda" if os.environ.get("USE_GPU", "false").lower() == "true" else "cpu"
        )
    return pipeline


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array.

    Args:
        base64_string: Base64 encoded image (with or without data URI prefix)

    Returns:
        RGB numpy array
    """
    # Remove data URI prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]

    # Decode base64
    image_data = base64.b64decode(base64_string)

    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_data))

    # Convert to RGB numpy array
    return np.array(image.convert("RGB"))


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """
    Encode binary mask to base64 PNG string.

    Args:
        mask: Binary mask (0 or 255)

    Returns:
        Base64 encoded PNG
    """
    # Convert to PIL Image
    image = Image.fromarray(mask)

    # Save to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Encode to base64
    return base64.b64encode(buffer.read()).decode("utf-8")


def create_overlay_image(
    original: np.ndarray,
    skeleton: np.ndarray,
    line_color: tuple = (255, 215, 0),
    line_thickness: int = 2
) -> np.ndarray:
    """
    Create overlay image with skeleton lines drawn on original.

    Args:
        original: Original RGB image
        skeleton: Binary skeleton mask
        line_color: Color for lines (default golden)
        line_thickness: Thickness of lines

    Returns:
        Overlay image
    """
    overlay = original.copy()

    # Dilate skeleton for visibility
    if line_thickness > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (line_thickness, line_thickness)
        )
        skeleton = cv2.dilate(skeleton, kernel)

    # Apply color to skeleton pixels
    overlay[skeleton > 0] = line_color

    return overlay


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "palm-line-detection"})


@app.route("/detect", methods=["POST"])
def detect_palm_lines():
    """
    Detect palm lines in an image.

    Request body (JSON):
        image: Base64 encoded image
        return_intermediate: (optional) Return intermediate results

    Response (JSON):
        success: Boolean
        skeleton: Base64 encoded skeleton overlay image
        bounding_boxes: Dictionary of detected line bounding boxes
        masks: (optional) Dictionary of line name to base64 encoded mask
        frangi_response: (optional) Base64 encoded Frangi response
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Decode image
        image_rgb = decode_base64_image(data["image"])

        # Run pipeline
        pipe = get_pipeline()
        result = pipe.process_array(image_rgb)

        # Create overlay image
        overlay = create_overlay_image(
            result.original_image,
            result.combined_skeleton,
            line_thickness=data.get("line_thickness", 2)
        )

        # Prepare response
        response = {
            "success": True,
            "skeleton_overlay": encode_mask_to_base64(overlay),
            "skeleton_mask": encode_mask_to_base64(result.combined_skeleton),
            "bounding_boxes": result.bounding_boxes,
            "detected_lines": list(result.bounding_boxes.keys())
        }

        # Optionally include intermediate results
        if data.get("return_intermediate", False):
            response["frangi_response"] = encode_mask_to_base64(
                (result.frangi_response * 255).astype(np.uint8)
            )
            response["masks"] = {
                name: encode_mask_to_base64(mask)
                for name, mask in result.segmentation_masks.items()
            }
            response["individual_skeletons"] = {
                name: encode_mask_to_base64(skeleton)
                for name, skeleton in result.skeletons.items()
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/analyze", methods=["POST"])
def analyze_palm_lines():
    """
    Detect and analyze palm lines with statistics.

    Request body (JSON):
        image: Base64 encoded image

    Response (JSON):
        success: Boolean
        analysis: Dictionary of line analysis data
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Decode image
        image_rgb = decode_base64_image(data["image"])

        # Run pipeline
        pipe = get_pipeline()
        result = pipe.process_array(image_rgb)

        # Analyze each line
        analysis = {}
        h, w = image_rgb.shape[:2]

        for name, skeleton in result.skeletons.items():
            # Find skeleton points
            points = np.argwhere(skeleton > 0)

            if len(points) == 0:
                analysis[name] = {
                    "detected": False,
                    "length": 0,
                    "coverage": 0
                }
                continue

            # Calculate line properties
            pixel_count = len(points)
            y_coords, x_coords = points[:, 0], points[:, 1]

            analysis[name] = {
                "detected": True,
                "length": pixel_count,
                "coverage": pixel_count / (h * w) * 100,  # Percentage
                "start_point": [int(x_coords.min()), int(y_coords[x_coords.argmin()])],
                "end_point": [int(x_coords.max()), int(y_coords[x_coords.argmax()])],
                "bounding_box": result.bounding_boxes.get(name, None),
                "curvature_estimate": _estimate_curvature(points) if len(points) > 10 else "insufficient_data"
            }

        # Create overlay for response
        overlay = create_overlay_image(
            result.original_image,
            result.combined_skeleton
        )

        return jsonify({
            "success": True,
            "analysis": analysis,
            "total_lines_detected": sum(1 for a in analysis.values() if a["detected"]),
            "skeleton_overlay": encode_mask_to_base64(overlay)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


def _estimate_curvature(points: np.ndarray) -> str:
    """
    Estimate line curvature from skeleton points.

    Returns a qualitative description: "straight", "slight_curve", "curved", "highly_curved"
    """
    if len(points) < 3:
        return "insufficient_data"

    # Fit a line to the points
    y_coords, x_coords = points[:, 0], points[:, 1]

    # Calculate line fit residuals
    z = np.polyfit(x_coords, y_coords, 1)
    p = np.poly1d(z)
    fitted_y = p(x_coords)
    residuals = np.abs(y_coords - fitted_y)
    mean_residual = np.mean(residuals)

    # Classify curvature
    line_length = np.sqrt((x_coords.max() - x_coords.min())**2 +
                          (y_coords.max() - y_coords.min())**2)

    if line_length == 0:
        return "point"

    normalized_residual = mean_residual / line_length

    if normalized_residual < 0.02:
        return "straight"
    elif normalized_residual < 0.05:
        return "slight_curve"
    elif normalized_residual < 0.1:
        return "curved"
    else:
        return "highly_curved"


@app.route("/configure", methods=["POST"])
def configure_pipeline():
    """
    Update pipeline configuration.

    Request body (JSON):
        frangi_sigmas: (optional) Tuple of sigma values
        device: (optional) "cuda" or "cpu"

    Response (JSON):
        success: Boolean
        config: Current configuration
    """
    global pipeline

    try:
        data = request.get_json() or {}

        # Reset pipeline with new config
        pipeline = PalmLineDetectionPipeline(
            frangi_sigmas=tuple(data.get("frangi_sigmas", [1.0, 1.5, 2.0, 2.5, 3.0])),
            device=data.get("device", "cpu")
        )

        return jsonify({
            "success": True,
            "config": {
                "frangi_sigmas": list(pipeline.frangi_detector.sigmas),
                "device": data.get("device", "cpu")
            }
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/palm-reading", methods=["POST"])
def generate_palm_reading():
    """
    Generate a complete palm reading with extracted features and LLM prompts.

    Request body (JSON):
        image: Base64 encoded image
        tone: (optional) Reading tone - "warm", "mystical", "analytical"
        tradition: (optional) Palmistry tradition - "western", "chinese", "indian"
        depth: (optional) Reading depth - "brief", "standard", "detailed"
        focus_areas: (optional) List of areas to focus on
        landmarks: (optional) MediaPipe hand landmarks

    Response (JSON):
        success: Boolean
        features: Extracted palm features
        prompts: System and user prompts for LLM
        detection_quality: Quality assessment
        lines_detected: List of detected line types
        skeleton_overlay: Base64 encoded overlay image
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Decode image
        image_rgb = decode_base64_image(data["image"])

        # Parse landmarks if provided
        landmarks = None
        if "landmarks" in data and data["landmarks"]:
            # Assuming landmarks come as a list of 21 {x, y, z} objects
            landmarks = HandLandmarks.estimate_from_image(image_rgb.shape)
            # TODO: Parse actual MediaPipe landmarks when provided

        # Run detection pipeline
        pipe = get_pipeline()
        result = pipe.process_array(image_rgb)

        # Extract reading options
        tone = data.get("tone", "warm")
        tradition = data.get("tradition", "western")
        depth = data.get("depth", "standard")
        focus_areas = data.get("focus_areas", None)

        # Create palm reading package
        reading_data = create_palm_reading(
            image_rgb,
            result,
            landmarks,
            tone=tone,
            tradition=tradition,
            depth=depth
        )

        # Create overlay for response
        overlay = create_overlay_image(
            result.original_image,
            result.combined_skeleton
        )

        return jsonify({
            "success": True,
            "features": reading_data["features"],
            "prompts": reading_data["prompts"],
            "detection_quality": reading_data["detection_quality"],
            "lines_detected": reading_data["lines_detected"],
            "skeleton_overlay": encode_mask_to_base64(overlay),
            "skeleton_mask": encode_mask_to_base64(result.combined_skeleton)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/palm-reading/features", methods=["POST"])
def extract_palm_features():
    """
    Extract palm features only (without generating prompts).
    Useful for debugging or custom prompt construction.

    Request body (JSON):
        image: Base64 encoded image

    Response (JSON):
        success: Boolean
        features: Extracted palm features as JSON
    """
    try:
        data = request.get_json()

        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        # Decode image
        image_rgb = decode_base64_image(data["image"])

        # Run detection pipeline
        pipe = get_pipeline()
        result = pipe.process_array(image_rgb)

        # Extract features using bridge
        bridge = PalmReadingBridge()
        features = bridge.process(
            image_rgb,
            result.segmentation_masks,
            None  # No landmarks
        )

        return jsonify({
            "success": True,
            "features": features._to_dict(),
            "features_json": features.to_json()
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Use port 5001 by default to avoid AirPlay conflict on macOS
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    print(f"Starting Palm Line Detection API on port {port}")
    print(f"Debug mode: {debug}")

    app.run(host="0.0.0.0", port=port, debug=debug)
