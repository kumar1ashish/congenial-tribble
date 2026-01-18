"""
DeepInsights - Neuro-Symbolic Palm Line Detection Pipeline

A modular architecture combining Traditional CV with Vision-Language Models
for precise extraction of major palm lines (Heart, Head, Life).

Phases:
1. Frangi Vesselness Filter - Ridge/tubular structure detection
2. Grounding DINO - Semantic line localization via VLM
3. SAM (Segment Anything) - Precision segmentation
4. Zhang-Suen Skeletonization - 1-pixel path extraction
"""

import cv2
import numpy as np
from skimage.filters import frangi
from skimage.morphology import skeletonize
from skimage import img_as_ubyte, img_as_float
from typing import Dict, List, Tuple, Optional
import torch
from dataclasses import dataclass
from pathlib import Path


def create_skin_mask(image: np.ndarray) -> np.ndarray:
    """
    Create skin mask using YCbCr color space.
    Works well for various skin tones.
    """
    # Convert RGB to YCbCr
    ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Skin detection thresholds in YCbCr space
    mask = np.zeros(y.shape, dtype=np.uint8)
    skin_condition = (
        (y > 80) &
        (cb > 77) & (cb < 127) &
        (cr > 133) & (cr < 173)
    )
    mask[skin_condition] = 255

    # Clean up mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def create_interior_mask(image: np.ndarray, erosion_ratio: float = 0.05) -> np.ndarray:
    """
    Create interior palm mask excluding boundaries.
    This prevents detecting the hand outline.
    """
    skin_mask = create_skin_mask(image)

    # Calculate erosion size based on image dimensions
    h, w = image.shape[:2]
    erosion_size = max(15, int(min(h, w) * erosion_ratio))

    # Erode mask to exclude boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erosion_size, erosion_size))
    interior_mask = cv2.erode(skin_mask, kernel)

    return interior_mask


@dataclass
class LineDetectionResult:
    """Container for palm line detection results"""
    original_image: np.ndarray
    frangi_response: np.ndarray
    bounding_boxes: Dict[str, Tuple[int, int, int, int]]
    segmentation_masks: Dict[str, np.ndarray]
    skeletons: Dict[str, np.ndarray]
    combined_skeleton: np.ndarray


class FrangiRidgeDetector:
    """
    Phase 1: Traditional CV Enhancement using Frangi Vesselness Filter

    Palm lines are tubular/ridge-like structures, not simple step-edges.
    Frangi filter excels at detecting such structures while suppressing noise.
    """

    def __init__(
        self,
        sigmas: Tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0),
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 15,
        black_ridges: bool = True
    ):
        """
        Initialize Frangi detector.

        Args:
            sigmas: Scale range for detecting lines of varying thickness
            alpha: Frangi correction constant for plate-like structures
            beta: Frangi correction constant for blob-like structures
            gamma: Frangi correction constant for background suppression
            black_ridges: True for dark lines on light background (palm lines)
        """
        self.sigmas = sigmas
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.black_ridges = black_ridges

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal Frangi response.

        Args:
            image: Input BGR image

        Returns:
            Preprocessed grayscale image (float)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Moderate Gaussian blur to reduce texture but preserve line structure
        blurred = cv2.GaussianBlur(enhanced, (7, 7), 1.5)

        # Convert to float [0, 1]
        return img_as_float(blurred)

    def detect(self, image: np.ndarray, interior_mask: np.ndarray = None) -> np.ndarray:
        """
        Apply Frangi vesselness filter to detect ridge structures.

        Args:
            image: Input BGR image
            interior_mask: Optional mask to exclude boundaries

        Returns:
            Vessel probability map (high values = likely palm line)
        """
        preprocessed = self.preprocess(image)

        # Apply Frangi filter
        # sigmas control the scale of detected structures
        # black_ridges=True detects dark lines on light background
        vesselness = frangi(
            preprocessed,
            sigmas=self.sigmas,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            black_ridges=self.black_ridges
        )

        # Apply interior mask to exclude hand boundaries
        if interior_mask is not None:
            vesselness = vesselness * (interior_mask / 255.0)

        # Normalize to [0, 1]
        vesselness_normalized = (vesselness - vesselness.min()) / (vesselness.max() - vesselness.min() + 1e-8)

        return vesselness_normalized

    def get_enhanced_rgb(self, vesselness: np.ndarray) -> np.ndarray:
        """
        Convert vesselness map back to RGB for VLM input.

        Args:
            vesselness: Normalized vesselness map

        Returns:
            RGB image with enhanced palm lines
        """
        # Scale to 8-bit
        enhanced_8bit = img_as_ubyte(vesselness)

        # Convert to RGB
        enhanced_rgb = cv2.cvtColor(enhanced_8bit, cv2.COLOR_GRAY2RGB)

        return enhanced_rgb


class GroundingDINOLocator:
    """
    Phase 2: Semantic Grounding using Grounding DINO

    Uses Vision-Language Model to spatially locate specific palm lines
    by querying for semantic structures.
    """

    PALM_LINE_PROMPTS = [
        "life line",
        "head line",
        "heart line"
    ]

    def __init__(
        self,
        model_config_path: str = "groundingdino/config/GroundingDINO_SwinT_OGC.py",
        model_weights_path: str = "weights/groundingdino_swint_ogc.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ):
        """
        Initialize Grounding DINO.

        Args:
            model_config_path: Path to model config
            model_weights_path: Path to model weights
            device: Computation device
            box_threshold: Confidence threshold for bounding boxes
            text_threshold: Confidence threshold for text matching
        """
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        self.model_config_path = model_config_path
        self.model_weights_path = model_weights_path

    def load_model(self):
        """Load Grounding DINO model (lazy loading)."""
        if self.model is not None:
            return

        try:
            from groundingdino.util.inference import load_model
            self.model = load_model(
                self.model_config_path,
                self.model_weights_path,
                device=self.device
            )
            print(f"Grounding DINO loaded on {self.device}")
        except ImportError:
            print("Warning: groundingdino not installed. Using fallback detection.")
            self.model = "fallback"

    def detect_lines(
        self,
        image: np.ndarray,
        prompts: Optional[List[str]] = None
    ) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Detect palm lines using semantic queries.

        Args:
            image: RGB image (enhanced or original)
            prompts: Custom prompts or use defaults

        Returns:
            Dictionary mapping line names to bounding boxes [x1, y1, x2, y2]
        """
        self.load_model()

        if prompts is None:
            prompts = self.PALM_LINE_PROMPTS

        # Fallback detection based on typical palm line positions
        if self.model == "fallback":
            return self._fallback_detection(image)

        try:
            from groundingdino.util.inference import predict

            boxes_dict = {}

            for prompt in prompts:
                # Run inference
                boxes, logits, phrases = predict(
                    model=self.model,
                    image=image,
                    caption=prompt,
                    box_threshold=self.box_threshold,
                    text_threshold=self.text_threshold,
                    device=self.device
                )

                if len(boxes) > 0:
                    # Take highest confidence box
                    best_idx = logits.argmax()
                    box = boxes[best_idx].cpu().numpy()

                    # Convert from normalized [cx, cy, w, h] to [x1, y1, x2, y2]
                    h, w = image.shape[:2]
                    cx, cy, bw, bh = box
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)

                    boxes_dict[prompt] = (x1, y1, x2, y2)

            return boxes_dict

        except Exception as e:
            print(f"Grounding DINO inference error: {e}")
            return self._fallback_detection(image)

    def _fallback_detection(self, image: np.ndarray) -> Dict[str, Tuple[int, int, int, int]]:
        """
        Fallback detection based on typical palm line anatomy.
        Covers the FULL palm area for each line type.

        Palm lines typically appear:
        - Heart line: Upper portion of palm (horizontal, runs across top)
        - Head line: Middle portion (horizontal, runs across middle)
        - Life line: Curves around thumb base (covers more area)
        """
        h, w = image.shape[:2]

        # Use full width for horizontal lines, generous coverage
        return {
            "heart line": (int(w * 0.05), int(h * 0.15), int(w * 0.95), int(h * 0.40)),
            "head line": (int(w * 0.05), int(h * 0.30), int(w * 0.95), int(h * 0.60)),
            "life line": (int(w * 0.05), int(h * 0.20), int(w * 0.95), int(h * 0.90))
        }


def link_edges_contour_tracing(edge_map: np.ndarray, min_contour_length: int = 35) -> np.ndarray:
    """
    Link edges using 8-connectivity flood fill and filter by contour length.

    This traces connected components and only keeps contours longer than
    the minimum length threshold, removing noise and short fragments.

    Args:
        edge_map: Binary edge map (0 or 255)
        min_contour_length: Minimum number of pixels to keep a contour

    Returns:
        Filtered edge map with only long contours
    """
    height, width = edge_map.shape
    visited = np.zeros((height, width), dtype=np.uint8)
    linked_edges = np.zeros((height, width), dtype=np.uint8)

    # 8-connectivity neighbor offsets
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(height):
        for x in range(width):
            if edge_map[y, x] == 255 and visited[y, x] == 0:
                # Trace contour using flood fill
                contour = []
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if cy < 0 or cy >= height or cx < 0 or cx >= width:
                        continue
                    if visited[cy, cx] == 1:
                        continue
                    if edge_map[cy, cx] != 255:
                        continue

                    visited[cy, cx] = 1
                    contour.append((cy, cx))

                    # Add 8-connected neighbors
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            if visited[ny, nx] == 0 and edge_map[ny, nx] == 255:
                                stack.append((ny, nx))

                # Only keep contours longer than minimum
                if len(contour) >= min_contour_length:
                    for cy, cx in contour:
                        linked_edges[cy, cx] = 255

    return linked_edges


def find_endpoints(edge_map: np.ndarray) -> list:
    """
    Find endpoints in the edge map (pixels with exactly 1 neighbor).

    Args:
        edge_map: Binary edge map (0 or 255)

    Returns:
        List of endpoint dictionaries with x, y, and direction
    """
    height, width = edge_map.shape
    endpoints = []

    # 8-connectivity neighbor offsets with direction info
    neighbors = [
        (-1, -1, 'nw'), (-1, 0, 'n'), (-1, 1, 'ne'),
        (0, -1, 'w'),                  (0, 1, 'e'),
        (1, -1, 'sw'), (1, 0, 's'), (1, 1, 'se')
    ]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if edge_map[y, x] == 255:
                neighbor_count = 0
                neighbor_dir = None

                for dy, dx, direction in neighbors:
                    if edge_map[y + dy, x + dx] == 255:
                        neighbor_count += 1
                        neighbor_dir = direction

                # Endpoint has exactly 1 neighbor
                if neighbor_count == 1:
                    endpoints.append({
                        'x': x,
                        'y': y,
                        'direction': neighbor_dir
                    })

    return endpoints


def directions_align(ep1: dict, ep2: dict) -> bool:
    """
    Check if two endpoints have compatible directions for bridging.

    Endpoints should point towards each other for a valid bridge.

    Args:
        ep1: First endpoint with direction
        ep2: Second endpoint with direction

    Returns:
        True if directions are compatible for bridging
    """
    # Direction mapping: which directions are opposite
    opposite_dirs = {
        'n': 's', 's': 'n', 'e': 'w', 'w': 'e',
        'ne': 'sw', 'sw': 'ne', 'nw': 'se', 'se': 'nw'
    }

    # Check if endpoints are roughly opposite
    dx = ep2['x'] - ep1['x']
    dy = ep2['y'] - ep1['y']

    # Determine relative direction of ep2 from ep1
    if abs(dx) > abs(dy):
        rel_dir = 'e' if dx > 0 else 'w'
    elif abs(dy) > abs(dx):
        rel_dir = 's' if dy > 0 else 'n'
    else:
        # Diagonal
        if dx > 0:
            rel_dir = 'se' if dy > 0 else 'ne'
        else:
            rel_dir = 'sw' if dy > 0 else 'nw'

    # ep1 should point toward ep2 (opposite of its neighbor direction)
    # ep1's neighbor is in direction ep1['direction'], so ep1 points opposite
    ep1_points = opposite_dirs.get(ep1['direction'], '')
    ep2_points = opposite_dirs.get(ep2['direction'], '')

    # Check if ep1 points roughly toward ep2
    return ep1_points == rel_dir or ep2_points == opposite_dirs.get(rel_dir, '')


def draw_line(edge_map: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    """
    Draw a line between two points using Bresenham's algorithm.

    Args:
        edge_map: Edge map to draw on (modified in place)
        x1, y1: Start point
        x2, y2: End point
    """
    cv2.line(edge_map, (x1, y1), (x2, y2), 255, 1)


def bridge_gaps(edge_map: np.ndarray, max_gap_size: int = 7) -> np.ndarray:
    """
    Bridge small gaps between edge endpoints.

    This connects nearby endpoints that have compatible directions,
    filling in small gaps that break continuous lines.

    Args:
        edge_map: Binary edge map (0 or 255)
        max_gap_size: Maximum gap distance to bridge in pixels

    Returns:
        Edge map with bridged gaps
    """
    result = edge_map.copy()
    endpoints = find_endpoints(result)

    bridged = set()  # Track which endpoints have been bridged

    for i, ep1 in enumerate(endpoints):
        if i in bridged:
            continue

        for j, ep2 in enumerate(endpoints):
            if i >= j or j in bridged:
                continue

            # Calculate distance
            distance = np.sqrt((ep1['x'] - ep2['x'])**2 + (ep1['y'] - ep2['y'])**2)

            # Bridge if close enough and directions align
            if distance <= max_gap_size and directions_align(ep1, ep2):
                draw_line(result, ep1['x'], ep1['y'], ep2['x'], ep2['y'])
                bridged.add(i)
                bridged.add(j)
                break  # Only bridge each endpoint once

    return result


def filter_short_contours(edge_map: np.ndarray, min_length: int = 40) -> np.ndarray:
    """
    Filter out contours shorter than the minimum length.

    Uses connected component analysis to identify and remove
    small fragments that are likely noise.

    Args:
        edge_map: Binary edge map (0 or 255)
        min_length: Minimum contour area/length in pixels

    Returns:
        Filtered edge map
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_map, connectivity=8)
    output = np.zeros_like(edge_map)

    for i in range(1, num_labels):  # Skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_length:
            output[labels == i] = 255

    return output


class SAMSegmenter:
    """
    Phase 3: Precision Segmentation using Segment Anything Model (SAM)

    Uses VLM-detected bounding boxes as geometric prompts to generate
    clean binary masks for each palm line.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: str = "weights/sam_vit_h_4b8939.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize SAM segmenter.

        Args:
            model_type: SAM model variant (vit_h, vit_l, vit_b)
            checkpoint_path: Path to SAM weights
            device: Computation device
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor = None

    def load_model(self):
        """Load SAM model (lazy loading)."""
        if self.predictor is not None:
            return

        try:
            from segment_anything import sam_model_registry, SamPredictor

            sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            sam.to(device=self.device)
            self.predictor = SamPredictor(sam)
            print(f"SAM ({self.model_type}) loaded on {self.device}")
        except ImportError:
            print("Warning: segment_anything not installed. Using fallback segmentation.")
            self.predictor = "fallback"

    def segment_lines(
        self,
        image: np.ndarray,
        bounding_boxes: Dict[str, Tuple[int, int, int, int]],
        frangi_response: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Segment palm lines using bounding box prompts.

        Args:
            image: Original RGB image
            bounding_boxes: Dictionary of line names to bounding boxes
            frangi_response: Optional Frangi response for fallback

        Returns:
            Dictionary mapping line names to binary masks
        """
        self.load_model()

        if self.predictor == "fallback":
            return self._fallback_segmentation(image, bounding_boxes, frangi_response)

        # Set image for SAM
        self.predictor.set_image(image)

        masks_dict = {}

        for line_name, box in bounding_boxes.items():
            x1, y1, x2, y2 = box
            input_box = np.array([x1, y1, x2, y2])

            # Predict mask using box prompt
            masks, scores, logits = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_box[None, :],
                multimask_output=True
            )

            # Take highest scoring mask
            best_mask = masks[scores.argmax()]
            masks_dict[line_name] = best_mask.astype(np.uint8) * 255

        return masks_dict

    def _fallback_segmentation(
        self,
        image: np.ndarray,
        bounding_boxes: Dict[str, Tuple[int, int, int, int]],
        frangi_response: Optional[np.ndarray] = None,
        interior_mask: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Enhanced edge detection pipeline for palm line detection.

        Improvements over basic approach:
        1. Stronger Gaussian blur (7x7 or double 5x5 pass) to reduce skin texture noise
        2. Lower Canny threshold ratio (0.25) for better edge connectivity
        3. Edge linking with 8-connectivity contour tracing
        4. Gap bridging to connect nearby endpoints
        5. Minimum contour length filtering to remove noise fragments
        6. Improved morphological operations

        Combines bilateral filter + adaptive threshold with improved Canny detection.
        """
        from skimage.morphology import skeletonize

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if interior_mask is None:
            interior_mask = create_interior_mask(image)

        # === METHOD 1: Bilateral filter + adaptive threshold ===
        # Bilateral smooths skin texture but keeps line edges sharp
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        # CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(bilateral)

        # Adaptive threshold - finds dark lines locally
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 21, 4
        )

        # === METHOD 2: Improved Canny edge detection ===
        # FIX 1: Use stronger Gaussian blur (7x7 or double 5x5 pass)
        # This reduces skin texture noise before edge detection
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.5)
        blurred = cv2.GaussianBlur(blurred, (5, 5), 1.0)  # Second pass for stronger smoothing

        # FIX 2: Compute adaptive thresholds with lower ratio for better connectivity
        # Using 0.25 ratio instead of 0.3 for low/high threshold
        # This allows weak edges to connect to strong edges more easily
        gradient_magnitude = cv2.Sobel(blurred, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(blurred, cv2.CV_64F, 0, 1) ** 2
        gradient_magnitude = np.sqrt(gradient_magnitude)
        max_gradient = gradient_magnitude.max()

        # Adaptive threshold based on gradient magnitude
        high_threshold = max(30, int(max_gradient * 0.15))  # More sensitive high threshold
        low_threshold = max(8, int(high_threshold * 0.25))   # Lower ratio = more connectivity

        canny = cv2.Canny(blurred, low_threshold, high_threshold)

        # === COMBINE both methods ===
        combined = cv2.bitwise_or(adaptive, canny)

        # === Apply interior mask ===
        combined = cv2.bitwise_and(combined, interior_mask)

        # === FIX 3: Morphological closing with larger kernel to connect fragments ===
        # Use 7x7 kernel for better gap filling
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_close)

        # === FIX 4: Bridge small gaps between edge endpoints ===
        # This connects nearby endpoints that have compatible directions
        combined = bridge_gaps(combined, max_gap_size=7)

        # === FIX 5: Edge linking with contour tracing and length filtering ===
        # This traces connected components and removes short fragments (noise)
        combined = link_edges_contour_tracing(combined, min_contour_length=35)

        # === FIX 6: Additional contour length filtering after bridging ===
        # Remove any remaining short fragments
        combined = filter_short_contours(combined, min_length=40)

        # Remove noise with morphological opening
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel_open)

        # === Skeletonize for clean lines ===
        skeleton = skeletonize(combined > 0)
        skeleton_img = (skeleton * 255).astype(np.uint8)

        # === Dilate for visibility ===
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        final_lines = cv2.dilate(skeleton_img, kernel_dilate, iterations=1)

        masks_dict = {}
        for line_name in bounding_boxes.keys():
            masks_dict[line_name] = final_lines.copy()

        return masks_dict


class ZhangSuenSkeletonizer:
    """
    Phase 4: Post-Processing using Zhang-Suen Thinning

    Applies skeletonization to get 1-pixel wide paths for analysis.
    """

    def __init__(self, min_branch_length: int = 10):
        """
        Initialize skeletonizer.

        Args:
            min_branch_length: Minimum branch length to keep (prunes noise)
        """
        self.min_branch_length = min_branch_length

    def skeletonize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Zhang-Suen skeletonization to a binary mask.

        Args:
            mask: Binary mask (0 or 255)

        Returns:
            Skeletonized mask (1-pixel wide paths)
        """
        # Convert to boolean
        binary = mask > 127

        # Apply skeletonization
        skeleton = skeletonize(binary)

        # Convert back to uint8
        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        # Optional: prune small branches
        if self.min_branch_length > 0:
            skeleton_uint8 = self._prune_skeleton(skeleton_uint8)

        return skeleton_uint8

    def _prune_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """
        Remove small spurious branches from skeleton.

        Uses connected component analysis to remove tiny fragments.
        """
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            skeleton, connectivity=8
        )

        # Create pruned skeleton
        pruned = np.zeros_like(skeleton)

        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_branch_length:
                pruned[labels == i] = 255

        return pruned

    def skeletonize_all(
        self,
        masks: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        Skeletonize all masks and create combined result.

        Args:
            masks: Dictionary of line names to binary masks

        Returns:
            Tuple of (individual skeletons dict, combined skeleton)
        """
        skeletons = {}
        combined = None

        for name, mask in masks.items():
            skeleton = self.skeletonize_mask(mask)
            skeletons[name] = skeleton

            if combined is None:
                combined = skeleton.copy()
            else:
                combined = cv2.bitwise_or(combined, skeleton)

        return skeletons, combined


class PalmLineDetectionPipeline:
    """
    Complete Neuro-Symbolic Pipeline for Palm Line Detection

    Orchestrates all four phases:
    1. Frangi Ridge Detection
    2. Grounding DINO Localization
    3. SAM Segmentation
    4. Zhang-Suen Skeletonization
    """

    def __init__(
        self,
        frangi_sigmas: Tuple[float, ...] = (1.5, 2.0, 2.5, 3.0, 3.5),  # Range for various line thicknesses
        grounding_dino_config: Optional[str] = None,
        grounding_dino_weights: Optional[str] = None,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_h",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize the complete pipeline.

        Args:
            frangi_sigmas: Scale range for Frangi filter
            grounding_dino_config: Path to Grounding DINO config
            grounding_dino_weights: Path to Grounding DINO weights
            sam_checkpoint: Path to SAM weights
            sam_model_type: SAM model variant
            device: Computation device
        """
        # Phase 1: Frangi Ridge Detector
        self.frangi_detector = FrangiRidgeDetector(sigmas=frangi_sigmas)

        # Phase 2: Grounding DINO
        self.dino_locator = GroundingDINOLocator(
            model_config_path=grounding_dino_config or "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_weights_path=grounding_dino_weights or "weights/groundingdino_swint_ogc.pth",
            device=device
        )

        # Phase 3: SAM Segmenter
        self.sam_segmenter = SAMSegmenter(
            model_type=sam_model_type,
            checkpoint_path=sam_checkpoint or f"weights/sam_{sam_model_type}_4b8939.pth",
            device=device
        )

        # Phase 4: Skeletonizer
        self.skeletonizer = ZhangSuenSkeletonizer(min_branch_length=40)

    def process(self, image_path: str) -> LineDetectionResult:
        """
        Run the complete palm line detection pipeline.

        Args:
            image_path: Path to input palm image

        Returns:
            LineDetectionResult with all intermediate and final outputs
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return self.process_array(image_rgb)

    def process_array(self, image_rgb: np.ndarray) -> LineDetectionResult:
        """
        Run pipeline on numpy array.

        Args:
            image_rgb: RGB image array

        Returns:
            LineDetectionResult with all outputs
        """
        # Create interior mask to exclude hand boundaries
        print("Creating interior mask to exclude hand boundaries...")
        interior_mask = create_interior_mask(image_rgb, erosion_ratio=0.07)  # Stronger erosion to exclude outline

        print("Phase 1: Applying Frangi Vesselness Filter...")
        frangi_response = self.frangi_detector.detect(image_rgb, interior_mask)
        enhanced_rgb = self.frangi_detector.get_enhanced_rgb(frangi_response)

        print("Phase 2: Localizing palm lines with Grounding DINO...")
        bounding_boxes = self.dino_locator.detect_lines(enhanced_rgb)
        print(f"  Detected {len(bounding_boxes)} lines: {list(bounding_boxes.keys())}")

        print("Phase 3: Segmenting lines with SAM...")
        # Pass interior mask to fallback segmentation
        if self.sam_segmenter.predictor == "fallback" or self.sam_segmenter.predictor is None:
            self.sam_segmenter.load_model()

        if self.sam_segmenter.predictor == "fallback":
            segmentation_masks = self.sam_segmenter._fallback_segmentation(
                image_rgb,
                bounding_boxes,
                frangi_response,
                interior_mask
            )
        else:
            segmentation_masks = self.sam_segmenter.segment_lines(
                image_rgb,
                bounding_boxes,
                frangi_response
            )

        # Skip skeletonization to preserve natural line thickness
        # Just combine the segmentation masks directly
        print("Phase 4: Combining line masks (preserving natural thickness)...")

        combined_mask = None
        for name, mask in segmentation_masks.items():
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask = cv2.bitwise_or(combined_mask, mask)

        if combined_mask is None:
            combined_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

        # Use segmentation masks as "skeletons" (they have natural thickness)
        skeletons = segmentation_masks
        combined_skeleton = combined_mask

        print("Pipeline complete!")

        return LineDetectionResult(
            original_image=image_rgb,
            frangi_response=frangi_response,
            bounding_boxes=bounding_boxes,
            segmentation_masks=segmentation_masks,
            skeletons=skeletons,
            combined_skeleton=combined_skeleton
        )

    def visualize_results(
        self,
        result: LineDetectionResult,
        output_path: Optional[str] = None,
        show: bool = True
    ) -> np.ndarray:
        """
        Create visualization of all pipeline stages.

        Args:
            result: Pipeline output
            output_path: Optional path to save visualization
            show: Whether to display with cv2.imshow

        Returns:
            Visualization image
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image
        axes[0, 0].imshow(result.original_image)
        axes[0, 0].set_title("Original Image")
        axes[0, 0].axis("off")

        # Frangi response
        axes[0, 1].imshow(result.frangi_response, cmap="hot")
        axes[0, 1].set_title("Frangi Vesselness Response")
        axes[0, 1].axis("off")

        # Bounding boxes
        img_with_boxes = result.original_image.copy()
        colors = {"heart line": (255, 0, 0), "head line": (0, 255, 0), "life line": (0, 0, 255)}
        for name, box in result.bounding_boxes.items():
            x1, y1, x2, y2 = box
            color = colors.get(name, (255, 255, 0))
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_with_boxes, name, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        axes[0, 2].imshow(img_with_boxes)
        axes[0, 2].set_title("Grounding DINO Boxes")
        axes[0, 2].axis("off")

        # Combined segmentation masks
        combined_mask = np.zeros((*result.original_image.shape[:2], 3), dtype=np.uint8)
        for name, mask in result.segmentation_masks.items():
            color = colors.get(name, (255, 255, 0))
            combined_mask[mask > 0] = color
        axes[1, 0].imshow(combined_mask)
        axes[1, 0].set_title("SAM Segmentation Masks")
        axes[1, 0].axis("off")

        # Individual skeletons
        skeleton_viz = np.zeros((*result.original_image.shape[:2], 3), dtype=np.uint8)
        for name, skeleton in result.skeletons.items():
            color = colors.get(name, (255, 255, 0))
            skeleton_viz[skeleton > 0] = color
        axes[1, 1].imshow(skeleton_viz)
        axes[1, 1].set_title("Individual Skeletons")
        axes[1, 1].axis("off")

        # Final overlay
        overlay = result.original_image.copy()
        overlay[result.combined_skeleton > 0] = [255, 215, 0]  # Golden
        axes[1, 2].imshow(overlay)
        axes[1, 2].set_title("Final Result (Skeletons on Original)")
        axes[1, 2].axis("off")

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Visualization saved to {output_path}")

        if show:
            plt.show()

        plt.close()

        return overlay


def main():
    """
    Example usage of the palm line detection pipeline.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Palm Line Detection Pipeline")
    parser.add_argument("image_path", help="Path to palm image")
    parser.add_argument("--output", "-o", help="Output visualization path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    parser.add_argument("--no-show", action="store_true", help="Don't display results")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PalmLineDetectionPipeline(device=args.device)

    # Process image
    result = pipeline.process(args.image_path)

    # Visualize
    pipeline.visualize_results(
        result,
        output_path=args.output,
        show=not args.no_show
    )

    # Print skeleton statistics
    print("\nSkeleton Statistics:")
    for name, skeleton in result.skeletons.items():
        pixel_count = np.sum(skeleton > 0)
        print(f"  {name}: {pixel_count} pixels")


if __name__ == "__main__":
    main()
