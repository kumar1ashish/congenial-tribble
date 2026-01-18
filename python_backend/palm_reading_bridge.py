"""
Palm Reading Bridge - Connects Edge Detection to LLM Interpretation

This module bridges the gap between raw edge detection output and
structured data that an LLM can use to generate palm readings.

Pipeline Stages:
1. Edge Detection (palm_line_detector.py) → Raw contours
2. Line Classification → Identify which line is which
3. Feature Extraction → Analyze palmistry-relevant attributes
4. LLM Prompt Construction → Build structured prompt
5. LLM Interpretation → Generate reading
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import math


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class LineType(Enum):
    HEART = "heart"
    HEAD = "head"
    LIFE = "life"
    FATE = "fate"
    SUN = "sun"
    MERCURY = "mercury"
    UNKNOWN = "unknown"


class LineQuality(Enum):
    FAINT = "faint"
    MEDIUM = "medium"
    DEEP = "deep"
    STRONG = "strong"


class LineClarity(Enum):
    BROKEN = "broken"
    CHAINED = "chained"
    CLEAR = "clear"
    STRONG = "strong"


class LineCurvature(Enum):
    STRAIGHT = "straight"
    SLIGHT = "slight"
    CURVED = "curved"
    WAVY = "wavy"


class LineLength(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class PalmShape(Enum):
    EARTH = "earth"
    AIR = "air"
    FIRE = "fire"
    WATER = "water"


@dataclass
class HandLandmarks:
    """MediaPipe hand landmark positions (21 points)"""
    wrist: Tuple[float, float] = (0, 0)
    thumb_cmc: Tuple[float, float] = (0, 0)
    thumb_mcp: Tuple[float, float] = (0, 0)
    thumb_ip: Tuple[float, float] = (0, 0)
    thumb_tip: Tuple[float, float] = (0, 0)
    index_mcp: Tuple[float, float] = (0, 0)
    index_pip: Tuple[float, float] = (0, 0)
    index_dip: Tuple[float, float] = (0, 0)
    index_tip: Tuple[float, float] = (0, 0)
    middle_mcp: Tuple[float, float] = (0, 0)
    middle_pip: Tuple[float, float] = (0, 0)
    middle_dip: Tuple[float, float] = (0, 0)
    middle_tip: Tuple[float, float] = (0, 0)
    ring_mcp: Tuple[float, float] = (0, 0)
    ring_pip: Tuple[float, float] = (0, 0)
    ring_dip: Tuple[float, float] = (0, 0)
    ring_tip: Tuple[float, float] = (0, 0)
    pinky_mcp: Tuple[float, float] = (0, 0)
    pinky_pip: Tuple[float, float] = (0, 0)
    pinky_dip: Tuple[float, float] = (0, 0)
    pinky_tip: Tuple[float, float] = (0, 0)

    @classmethod
    def from_mediapipe(cls, landmarks, image_width: int, image_height: int) -> 'HandLandmarks':
        """Convert MediaPipe landmarks to pixel coordinates"""
        def to_pixel(lm):
            return (int(lm.x * image_width), int(lm.y * image_height))

        return cls(
            wrist=to_pixel(landmarks[0]),
            thumb_cmc=to_pixel(landmarks[1]),
            thumb_mcp=to_pixel(landmarks[2]),
            thumb_ip=to_pixel(landmarks[3]),
            thumb_tip=to_pixel(landmarks[4]),
            index_mcp=to_pixel(landmarks[5]),
            index_pip=to_pixel(landmarks[6]),
            index_dip=to_pixel(landmarks[7]),
            index_tip=to_pixel(landmarks[8]),
            middle_mcp=to_pixel(landmarks[9]),
            middle_pip=to_pixel(landmarks[10]),
            middle_dip=to_pixel(landmarks[11]),
            middle_tip=to_pixel(landmarks[12]),
            ring_mcp=to_pixel(landmarks[13]),
            ring_pip=to_pixel(landmarks[14]),
            ring_dip=to_pixel(landmarks[15]),
            ring_tip=to_pixel(landmarks[16]),
            pinky_mcp=to_pixel(landmarks[17]),
            pinky_pip=to_pixel(landmarks[18]),
            pinky_dip=to_pixel(landmarks[19]),
            pinky_tip=to_pixel(landmarks[20])
        )

    @classmethod
    def estimate_from_image(cls, image_shape: Tuple[int, int]) -> 'HandLandmarks':
        """Estimate landmarks based on typical palm proportions when MediaPipe unavailable"""
        h, w = image_shape[:2]

        # Typical palm proportions (assuming palm fills most of image)
        return cls(
            wrist=(int(w * 0.5), int(h * 0.95)),
            thumb_cmc=(int(w * 0.25), int(h * 0.75)),
            thumb_mcp=(int(w * 0.15), int(h * 0.60)),
            thumb_ip=(int(w * 0.10), int(h * 0.45)),
            thumb_tip=(int(w * 0.08), int(h * 0.35)),
            index_mcp=(int(w * 0.30), int(h * 0.25)),
            index_pip=(int(w * 0.28), int(h * 0.15)),
            index_dip=(int(w * 0.27), int(h * 0.08)),
            index_tip=(int(w * 0.26), int(h * 0.02)),
            middle_mcp=(int(w * 0.45), int(h * 0.22)),
            middle_pip=(int(w * 0.45), int(h * 0.12)),
            middle_dip=(int(w * 0.45), int(h * 0.05)),
            middle_tip=(int(w * 0.45), int(h * 0.0)),
            ring_mcp=(int(w * 0.60), int(h * 0.25)),
            ring_pip=(int(w * 0.62), int(h * 0.15)),
            ring_dip=(int(w * 0.63), int(h * 0.08)),
            ring_tip=(int(w * 0.64), int(h * 0.02)),
            pinky_mcp=(int(w * 0.75), int(h * 0.30)),
            pinky_pip=(int(w * 0.78), int(h * 0.22)),
            pinky_dip=(int(w * 0.80), int(h * 0.16)),
            pinky_tip=(int(w * 0.82), int(h * 0.10))
        )


@dataclass
class LineMarking:
    """Special marking on a palm line"""
    type: str  # cross, star, island, branch_up, branch_down, break, chain
    position_on_line: str  # start, early, middle, late, end
    confidence: float = 0.5


@dataclass
class DetectedLine:
    """A detected and classified palm line with features"""
    type: LineType
    confidence: float
    contour_points: List[Tuple[int, int]]

    # Quality metrics
    length: LineLength = LineLength.MEDIUM
    depth: LineQuality = LineQuality.MEDIUM
    clarity: LineClarity = LineClarity.CLEAR
    curvature: LineCurvature = LineCurvature.SLIGHT

    # Line-specific attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    markings: List[LineMarking] = field(default_factory=list)


@dataclass
class MountInfo:
    """Information about a palm mount"""
    name: str  # jupiter, saturn, apollo, mercury, venus, luna, mars_positive, mars_negative
    prominence: int  # 1-5 scale
    features: List[str] = field(default_factory=list)


@dataclass
class PalmFeatures:
    """Complete extracted palm features for LLM"""
    detection_quality: str  # poor, fair, good, excellent
    overall_confidence: float
    handedness: str  # left, right, unknown

    palm_shape: PalmShape = PalmShape.EARTH
    finger_length: str = "medium"
    palm_width: str = "medium"

    lines: List[DetectedLine] = field(default_factory=list)
    mounts: List[MountInfo] = field(default_factory=list)

    skin_texture: str = "medium"  # fine, medium, coarse
    line_count: str = "medium"  # few, medium, many

    def to_json(self) -> str:
        """Convert to JSON for LLM prompt"""
        return json.dumps(self._to_dict(), indent=2)

    def _to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            "detectionQuality": self.detection_quality,
            "overallConfidence": self.overall_confidence,
            "handedness": self.handedness,
            "palmShape": {
                "type": self.palm_shape.value,
                "fingerLength": self.finger_length,
                "palmWidth": self.palm_width
            },
            "lines": [
                {
                    "type": line.type.value,
                    "confidence": line.confidence,
                    "quality": {
                        "length": line.length.value,
                        "depth": line.depth.value,
                        "clarity": line.clarity.value,
                        "curvature": line.curvature.value
                    },
                    "attributes": line.attributes,
                    "markings": [
                        {
                            "type": m.type,
                            "positionOnLine": m.position_on_line,
                            "confidence": m.confidence
                        }
                        for m in line.markings
                    ]
                }
                for line in self.lines
            ],
            "mounts": [
                {
                    "name": m.name,
                    "prominence": m.prominence,
                    "features": m.features
                }
                for m in self.mounts
            ],
            "skinTexture": self.skin_texture,
            "lineCount": self.line_count
        }


# ============================================================================
# LINE CLASSIFIER
# ============================================================================

class LineClassifier:
    """
    Classifies detected contours into specific palm line types
    using hand landmarks as reference points.
    """

    def __init__(self, image_shape: Tuple[int, int], landmarks: Optional[HandLandmarks] = None):
        self.image_height, self.image_width = image_shape[:2]
        self.landmarks = landmarks or HandLandmarks.estimate_from_image(image_shape)

        # Define regions for each line type
        self._setup_regions()

    def _setup_regions(self):
        """Define expected regions for each palm line type"""
        h, w = self.image_height, self.image_width
        lm = self.landmarks

        # Heart line region: top horizontal band below fingers
        heart_top = int(min(lm.index_mcp[1], lm.middle_mcp[1], lm.ring_mcp[1], lm.pinky_mcp[1]))
        heart_bottom = int(heart_top + h * 0.15)

        # Head line region: middle horizontal band
        head_top = heart_bottom
        head_bottom = int(head_top + h * 0.20)

        # Life line region: curved area around thumb
        life_center_x = int((lm.thumb_cmc[0] + lm.wrist[0]) / 2)

        # Fate line region: vertical center
        fate_center_x = lm.middle_mcp[0]

        self.regions = {
            LineType.HEART: {
                'y_range': (heart_top, heart_bottom),
                'orientation': 'horizontal',
                'expected_start': 'right',  # Starts from pinky side
            },
            LineType.HEAD: {
                'y_range': (head_top, head_bottom),
                'orientation': 'horizontal',
                'expected_start': 'left',  # Starts from thumb side
            },
            LineType.LIFE: {
                'center_x': life_center_x,
                'orientation': 'curved',
                'curve_around': 'thumb',
            },
            LineType.FATE: {
                'center_x': fate_center_x,
                'orientation': 'vertical',
            }
        }

    def classify_contour(self, contour_points: List[Tuple[int, int]]) -> Tuple[LineType, float]:
        """
        Classify a contour as a specific palm line type.

        Returns:
            Tuple of (LineType, confidence)
        """
        if len(contour_points) < 10:
            return LineType.UNKNOWN, 0.0

        # Calculate contour properties
        points = np.array(contour_points)
        centroid = points.mean(axis=0)

        # Bounding box
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y

        # Aspect ratio indicates orientation
        aspect_ratio = width / max(height, 1)

        # Calculate curvature
        curvature = self._calculate_curvature(points)

        scores = {}

        # Score for Heart line
        heart_region = self.regions[LineType.HEART]
        if heart_region['y_range'][0] <= centroid[1] <= heart_region['y_range'][1]:
            if aspect_ratio > 1.5:  # Horizontal
                scores[LineType.HEART] = 0.7 + (0.2 if width > self.image_width * 0.3 else 0)

        # Score for Head line
        head_region = self.regions[LineType.HEAD]
        if head_region['y_range'][0] <= centroid[1] <= head_region['y_range'][1]:
            if aspect_ratio > 1.2:  # Horizontal
                scores[LineType.HEAD] = 0.65 + (0.2 if width > self.image_width * 0.25 else 0)

        # Score for Life line
        life_region = self.regions[LineType.LIFE]
        dist_to_life_center = abs(centroid[0] - life_region['center_x'])
        if dist_to_life_center < self.image_width * 0.25:
            if curvature > 0.3 and height > self.image_height * 0.2:  # Curved and tall
                scores[LineType.LIFE] = 0.6 + min(curvature * 0.3, 0.3)

        # Score for Fate line
        fate_region = self.regions[LineType.FATE]
        dist_to_fate_center = abs(centroid[0] - fate_region['center_x'])
        if dist_to_fate_center < self.image_width * 0.15:
            if aspect_ratio < 0.8:  # Vertical
                scores[LineType.FATE] = 0.5 + (0.2 if height > self.image_height * 0.2 else 0)

        if not scores:
            return LineType.UNKNOWN, 0.0

        best_type = max(scores, key=scores.get)
        return best_type, scores[best_type]

    def _calculate_curvature(self, points: np.ndarray) -> float:
        """Calculate average curvature of a contour"""
        if len(points) < 3:
            return 0.0

        # Sample points along the contour
        n_samples = min(20, len(points))
        indices = np.linspace(0, len(points) - 1, n_samples, dtype=int)
        sampled = points[indices]

        # Calculate angles between consecutive segments
        angles = []
        for i in range(1, len(sampled) - 1):
            v1 = sampled[i] - sampled[i-1]
            v2 = sampled[i+1] - sampled[i]

            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)

        if not angles:
            return 0.0

        # Higher values = more curved
        return np.mean(angles) / np.pi


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extracts palmistry-relevant features from detected lines.
    """

    def __init__(self, image: np.ndarray, landmarks: Optional[HandLandmarks] = None):
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        self.height, self.width = image.shape[:2]
        self.landmarks = landmarks or HandLandmarks.estimate_from_image(image.shape)

    def extract_line_features(self, contour_points: List[Tuple[int, int]],
                              line_type: LineType) -> DetectedLine:
        """Extract all features for a single line"""
        points = np.array(contour_points)

        # Basic metrics
        length = self._classify_length(points)
        depth = self._analyze_depth(points)
        clarity = self._analyze_clarity(points)
        curvature = self._analyze_curvature(points)

        # Line-specific attributes
        attributes = self._get_line_attributes(points, line_type)

        # Detect markings
        markings = self._detect_markings(points)

        return DetectedLine(
            type=line_type,
            confidence=0.0,  # Set by classifier
            contour_points=contour_points,
            length=length,
            depth=depth,
            clarity=clarity,
            curvature=curvature,
            attributes=attributes,
            markings=markings
        )

    def _classify_length(self, points: np.ndarray) -> LineLength:
        """Classify line length relative to palm size"""
        if len(points) < 2:
            return LineLength.SHORT

        # Calculate arc length
        arc_length = 0
        for i in range(1, len(points)):
            arc_length += np.linalg.norm(points[i] - points[i-1])

        # Normalize by palm width
        relative_length = arc_length / self.width

        if relative_length < 0.25:
            return LineLength.SHORT
        elif relative_length < 0.45:
            return LineLength.MEDIUM
        else:
            return LineLength.LONG

    def _analyze_depth(self, points: np.ndarray) -> LineQuality:
        """Analyze line depth based on intensity gradient"""
        if len(points) < 5:
            return LineQuality.MEDIUM

        # Sample intensity along the line
        # Points are in (x, y) format, but image access is [y, x]
        intensities = []
        for x, y in points[::max(1, len(points)//20)]:
            if 0 <= y < self.height and 0 <= x < self.width:
                intensities.append(self.gray[int(y), int(x)])

        if not intensities:
            return LineQuality.MEDIUM

        # Calculate contrast with surrounding area
        line_intensity = np.mean(intensities)

        # Sample nearby pixels for comparison
        surrounding = []
        for x, y in points[::max(1, len(points)//10)]:
            for dy in [-3, 3]:
                ny = int(y + dy)
                if 0 <= ny < self.height and 0 <= x < self.width:
                    surrounding.append(self.gray[ny, int(x)])

        if surrounding:
            surrounding_intensity = np.mean(surrounding)
            contrast = abs(surrounding_intensity - line_intensity)

            if contrast < 15:
                return LineQuality.FAINT
            elif contrast < 30:
                return LineQuality.MEDIUM
            elif contrast < 50:
                return LineQuality.DEEP
            else:
                return LineQuality.STRONG

        return LineQuality.MEDIUM

    def _analyze_clarity(self, points: np.ndarray) -> LineClarity:
        """Analyze line clarity (broken, chained, clear, strong)"""
        if len(points) < 10:
            return LineClarity.BROKEN

        # Check for gaps in the contour
        gaps = 0
        for i in range(1, len(points)):
            dist = np.linalg.norm(points[i] - points[i-1])
            if dist > 5:  # Gap detected
                gaps += 1

        gap_ratio = gaps / len(points)

        # Check for chain-like pattern (waviness in small scale)
        local_variance = self._calculate_local_variance(points)

        if gap_ratio > 0.1:
            return LineClarity.BROKEN
        elif local_variance > 0.5:
            return LineClarity.CHAINED
        elif local_variance < 0.2:
            return LineClarity.STRONG
        else:
            return LineClarity.CLEAR

    def _calculate_local_variance(self, points: np.ndarray) -> float:
        """Calculate local variance to detect chain-like patterns"""
        if len(points) < 10:
            return 0.0

        # Fit a smooth curve and measure deviation
        window = 5
        deviations = []

        for i in range(window, len(points) - window):
            local_mean = points[i-window:i+window].mean(axis=0)
            deviation = np.linalg.norm(points[i] - local_mean)
            deviations.append(deviation)

        return np.std(deviations) if deviations else 0.0

    def _analyze_curvature(self, points: np.ndarray) -> LineCurvature:
        """Analyze overall line curvature"""
        if len(points) < 3:
            return LineCurvature.STRAIGHT

        # Fit line and measure deviation
        start = points[0]
        end = points[-1]
        line_vec = end - start
        line_length = np.linalg.norm(line_vec)

        if line_length < 1:
            return LineCurvature.STRAIGHT

        # Calculate max deviation from straight line
        max_deviation = 0
        deviations = []

        for point in points:
            # Distance from point to line
            t = np.dot(point - start, line_vec) / (line_length ** 2)
            closest_on_line = start + t * line_vec
            deviation = np.linalg.norm(point - closest_on_line)
            deviations.append(deviation)
            max_deviation = max(max_deviation, deviation)

        relative_curve = max_deviation / line_length
        deviation_std = np.std(deviations)

        # Check for waviness
        is_wavy = deviation_std > max_deviation * 0.3

        if relative_curve < 0.05:
            return LineCurvature.STRAIGHT
        elif relative_curve < 0.15:
            return LineCurvature.SLIGHT
        elif is_wavy:
            return LineCurvature.WAVY
        else:
            return LineCurvature.CURVED

    def _get_line_attributes(self, points: np.ndarray,
                             line_type: LineType) -> Dict[str, Any]:
        """Get line-type specific attributes"""
        if len(points) < 2:
            return {}

        start = points[0]
        end = points[-1]

        if line_type == LineType.HEART:
            return {
                "endsToward": self._get_finger_direction(end),
                "curveDirection": "upward" if points[len(points)//2][1] < (start[1] + end[1])/2 else "downward"
            }

        elif line_type == LineType.HEAD:
            # Check connection to life line
            life_start_region = (self.landmarks.thumb_cmc[0], self.landmarks.index_mcp[1])
            connected = np.linalg.norm(start - np.array(life_start_region)) < self.width * 0.1

            # Calculate slope
            slope = (end[1] - start[1]) / max(end[0] - start[0], 1)
            slope_type = "level" if abs(slope) < 0.1 else ("downward" if slope > 0 else "upward")

            return {
                "connectedToLife": connected,
                "slope": slope_type
            }

        elif line_type == LineType.LIFE:
            # Arc width
            arc_width = max(points[:, 0]) - min(points[:, 0])
            arc_type = "wide" if arc_width > self.width * 0.25 else "narrow"

            # Extension
            extends_to_wrist = max(points[:, 1]) > self.height * 0.8

            return {
                "extensionLength": "to_wrist" if extends_to_wrist else "mid_palm",
                "arcWidth": arc_type
            }

        elif line_type == LineType.FATE:
            return {
                "startPosition": "wrist" if start[1] > self.height * 0.7 else "mid_palm",
                "endPosition": "saturn_mount" if end[1] < self.height * 0.3 else "mid_palm"
            }

        return {}

    def _get_finger_direction(self, point: Tuple[float, float]) -> str:
        """Determine which finger a point is directed toward"""
        x = point[0]

        fingers = [
            ("index_finger", self.landmarks.index_mcp[0]),
            ("middle_finger", self.landmarks.middle_mcp[0]),
            ("ring_finger", self.landmarks.ring_mcp[0]),
            ("pinky_finger", self.landmarks.pinky_mcp[0])
        ]

        closest = min(fingers, key=lambda f: abs(f[1] - x))
        return closest[0]

    def _detect_markings(self, points: np.ndarray) -> List[LineMarking]:
        """Detect special markings on the line"""
        markings = []

        if len(points) < 20:
            return markings

        # Divide line into sections
        sections = ["start", "early", "middle", "late", "end"]
        section_size = len(points) // 5

        for i, section in enumerate(sections):
            section_points = points[i*section_size:(i+1)*section_size]

            # Detect breaks
            if self._has_break(section_points):
                markings.append(LineMarking(
                    type="break",
                    position_on_line=section,
                    confidence=0.6
                ))

            # Detect islands (loops)
            if self._has_island(section_points):
                markings.append(LineMarking(
                    type="island",
                    position_on_line=section,
                    confidence=0.5
                ))

            # Detect branches
            branch = self._detect_branch(section_points, points)
            if branch:
                markings.append(LineMarking(
                    type=branch,
                    position_on_line=section,
                    confidence=0.55
                ))

        return markings

    def _has_break(self, points: np.ndarray) -> bool:
        """Check if section has a significant break"""
        if len(points) < 3:
            return False

        for i in range(1, len(points)):
            if np.linalg.norm(points[i] - points[i-1]) > 8:
                return True
        return False

    def _has_island(self, points: np.ndarray) -> bool:
        """Check for island/loop formation"""
        if len(points) < 10:
            return False

        # Check if points form a loop-like pattern
        local_variance = self._calculate_local_variance(points)
        return local_variance > 1.0

    def _detect_branch(self, section_points: np.ndarray,
                       all_points: np.ndarray) -> Optional[str]:
        """Detect if there's a branch in this section"""
        # Simplified branch detection based on local direction changes
        if len(section_points) < 5:
            return None

        # Check for sudden direction changes
        for i in range(2, len(section_points) - 2):
            v1 = section_points[i] - section_points[i-2]
            v2 = section_points[i+2] - section_points[i]

            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                if cos_angle < 0.5:  # Sharp turn
                    # Determine direction
                    cross = v1[0] * v2[1] - v1[1] * v2[0]
                    return "branch_up" if cross > 0 else "branch_down"

        return None

    def analyze_palm_shape(self) -> Tuple[PalmShape, str, str]:
        """Analyze overall palm shape"""
        lm = self.landmarks

        # Calculate palm dimensions
        palm_width = abs(lm.pinky_mcp[0] - lm.index_mcp[0])
        palm_height = abs(lm.wrist[1] - lm.middle_mcp[1])

        # Estimate finger length (middle finger)
        finger_length = abs(lm.middle_tip[1] - lm.middle_mcp[1])

        # Palm ratio
        palm_ratio = palm_width / max(palm_height, 1)
        finger_ratio = finger_length / max(palm_height, 1)

        # Classify palm shape
        if palm_ratio > 0.9 and finger_ratio < 0.8:
            shape = PalmShape.EARTH  # Square palm, short fingers
        elif palm_ratio > 0.9 and finger_ratio >= 0.8:
            shape = PalmShape.AIR  # Square palm, long fingers
        elif palm_ratio <= 0.9 and finger_ratio < 0.8:
            shape = PalmShape.FIRE  # Rectangular palm, short fingers
        else:
            shape = PalmShape.WATER  # Rectangular palm, long fingers

        finger_desc = "long" if finger_ratio >= 0.8 else "short" if finger_ratio < 0.6 else "medium"
        palm_desc = "wide" if palm_ratio > 0.95 else "narrow" if palm_ratio < 0.8 else "medium"

        return shape, finger_desc, palm_desc

    def analyze_mounts(self) -> List[MountInfo]:
        """Analyze palm mounts (elevated regions)"""
        mounts = []
        lm = self.landmarks

        # Define mount regions
        mount_regions = {
            "jupiter": (lm.index_mcp[0], lm.index_mcp[1] + 20),
            "saturn": (lm.middle_mcp[0], lm.middle_mcp[1] + 20),
            "apollo": (lm.ring_mcp[0], lm.ring_mcp[1] + 20),
            "mercury": (lm.pinky_mcp[0], lm.pinky_mcp[1] + 20),
            "venus": ((lm.thumb_cmc[0] + lm.wrist[0]) // 2, (lm.thumb_cmc[1] + lm.wrist[1]) // 2),
            "luna": (int(self.width * 0.8), int(self.height * 0.7))
        }

        for mount_name, (x, y) in mount_regions.items():
            prominence = self._estimate_mount_prominence(int(x), int(y))
            features = []

            if prominence >= 4:
                features.append("well-developed")
            elif prominence <= 2:
                features.append("flat")

            mounts.append(MountInfo(
                name=mount_name,
                prominence=prominence,
                features=features
            ))

        return mounts

    def _estimate_mount_prominence(self, x: int, y: int, radius: int = 20) -> int:
        """Estimate mount prominence based on local intensity variance"""
        # In a real implementation, this would use depth/3D information
        # Here we use intensity variance as a proxy

        x = max(radius, min(x, self.width - radius - 1))
        y = max(radius, min(y, self.height - radius - 1))

        region = self.gray[y-radius:y+radius, x-radius:x+radius]

        if region.size == 0:
            return 3

        variance = np.var(region)

        # Map variance to 1-5 scale
        if variance < 100:
            return 2
        elif variance < 300:
            return 3
        elif variance < 600:
            return 4
        else:
            return 5


# ============================================================================
# LLM PROMPT BUILDER
# ============================================================================

class PalmReadingPromptBuilder:
    """Builds structured prompts for LLM palm reading interpretation"""

    SYSTEM_PROMPT = """You are a skilled palm reader providing insightful, thoughtful readings.

IMPORTANT GUIDELINES:
1. Never make specific predictions about death, serious illness, or exact timelines
2. Frame insights as tendencies, potentials, and reflections - not certainties
3. Encourage self-reflection rather than dependency on the reading
4. Acknowledge that palmistry is a traditional practice for self-discovery, not science
5. Be warm and supportive while remaining honest about what you observe
6. Focus on character traits, tendencies, and potentials rather than fortune-telling

READING STRUCTURE:
1. Brief overview of palm characteristics
2. Analysis of each major line present (heart, head, life, fate if visible)
3. Notable features and what they traditionally suggest
4. Reflection questions for the querent
5. Closing disclaimer about palmistry being for reflection, not prediction

TONE: {tone}
TRADITION: {tradition}
DEPTH: {depth}"""

    USER_PROMPT_TEMPLATE = """Please provide a palm reading based on the following detected features:

{features_json}

The querent would like a {depth} reading focusing on: {focus_areas}

Please structure your reading with clear sections and include reflection questions."""

    @classmethod
    def build_system_prompt(cls,
                           tone: str = "warm",
                           tradition: str = "western",
                           depth: str = "standard") -> str:
        """Build the system prompt with customization"""
        return cls.SYSTEM_PROMPT.format(
            tone=tone,
            tradition=tradition,
            depth=depth
        )

    @classmethod
    def build_user_prompt(cls,
                         features: PalmFeatures,
                         depth: str = "standard",
                         focus_areas: Optional[List[str]] = None) -> str:
        """Build the user prompt with extracted features"""
        if focus_areas is None:
            focus_areas = ["love and relationships", "career", "life path"]

        return cls.USER_PROMPT_TEMPLATE.format(
            features_json=features.to_json(),
            depth=depth,
            focus_areas=", ".join(focus_areas)
        )


# ============================================================================
# MAIN BRIDGE PIPELINE
# ============================================================================

class PalmReadingBridge:
    """
    Main bridge connecting edge detection output to LLM interpretation.

    Usage:
        bridge = PalmReadingBridge()
        features = bridge.process(image, detection_result, landmarks)
        prompts = bridge.build_prompts(features)
    """

    def __init__(self):
        self.classifier = None
        self.extractor = None

    def process(self,
                image: np.ndarray,
                detection_masks: Dict[str, np.ndarray],
                landmarks: Optional[HandLandmarks] = None) -> PalmFeatures:
        """
        Process detection results into structured palm features.

        Args:
            image: Original RGB image
            detection_masks: Dictionary of line masks from edge detection
            landmarks: Optional MediaPipe hand landmarks

        Returns:
            PalmFeatures object ready for LLM interpretation
        """
        # Initialize processors
        self.classifier = LineClassifier(image.shape, landmarks)
        self.extractor = FeatureExtractor(image, landmarks)

        # Extract contours from masks
        all_contours = self._extract_contours(detection_masks)

        # Classify and extract features for each line
        detected_lines = []
        for contour_points in all_contours:
            if len(contour_points) < 20:
                continue

            # Classify the line
            line_type, confidence = self.classifier.classify_contour(contour_points)

            if line_type == LineType.UNKNOWN or confidence < 0.4:
                continue

            # Extract features
            line = self.extractor.extract_line_features(contour_points, line_type)
            line.confidence = confidence
            detected_lines.append(line)

        # Remove duplicates (keep highest confidence for each type)
        detected_lines = self._deduplicate_lines(detected_lines)

        # Analyze palm shape and mounts
        palm_shape, finger_length, palm_width = self.extractor.analyze_palm_shape()
        mounts = self.extractor.analyze_mounts()

        # Calculate overall quality
        detection_quality = self._assess_detection_quality(detected_lines)
        overall_confidence = np.mean([l.confidence for l in detected_lines]) if detected_lines else 0.0

        return PalmFeatures(
            detection_quality=detection_quality,
            overall_confidence=round(overall_confidence, 2),
            handedness="unknown",  # Would need additional logic
            palm_shape=palm_shape,
            finger_length=finger_length,
            palm_width=palm_width,
            lines=detected_lines,
            mounts=mounts,
            skin_texture="medium",
            line_count=self._count_line_density(all_contours)
        )

    def _extract_contours(self, masks: Dict[str, np.ndarray]) -> List[List[Tuple[int, int]]]:
        """Extract contour points from detection masks"""
        all_contours = []

        for name, mask in masks.items():
            if mask is None or mask.size == 0:
                continue

            # Ensure binary
            binary = (mask > 127).astype(np.uint8) * 255

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            for contour in contours:
                points = [(int(p[0][0]), int(p[0][1])) for p in contour]  # (x, y) format
                if len(points) >= 20:
                    all_contours.append(points)

        return all_contours

    def _deduplicate_lines(self, lines: List[DetectedLine]) -> List[DetectedLine]:
        """Keep only the highest confidence line of each type"""
        best_by_type = {}

        for line in lines:
            if line.type not in best_by_type or line.confidence > best_by_type[line.type].confidence:
                best_by_type[line.type] = line

        return list(best_by_type.values())

    def _assess_detection_quality(self, lines: List[DetectedLine]) -> str:
        """Assess overall detection quality"""
        if not lines:
            return "poor"

        # Check for major lines
        has_heart = any(l.type == LineType.HEART for l in lines)
        has_head = any(l.type == LineType.HEAD for l in lines)
        has_life = any(l.type == LineType.LIFE for l in lines)

        major_count = sum([has_heart, has_head, has_life])
        avg_confidence = np.mean([l.confidence for l in lines])

        if major_count >= 3 and avg_confidence > 0.7:
            return "excellent"
        elif major_count >= 2 and avg_confidence > 0.5:
            return "good"
        elif major_count >= 1:
            return "fair"
        else:
            return "poor"

    def _count_line_density(self, contours: List) -> str:
        """Classify line count density"""
        count = len(contours)

        if count < 3:
            return "few"
        elif count < 8:
            return "medium"
        else:
            return "many"

    def build_prompts(self,
                     features: PalmFeatures,
                     tone: str = "warm",
                     tradition: str = "western",
                     depth: str = "standard",
                     focus_areas: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Build LLM prompts from extracted features.

        Returns:
            Dictionary with 'system' and 'user' prompts
        """
        return {
            "system": PalmReadingPromptBuilder.build_system_prompt(tone, tradition, depth),
            "user": PalmReadingPromptBuilder.build_user_prompt(features, depth, focus_areas)
        }


# ============================================================================
# INTEGRATION HELPER
# ============================================================================

def create_palm_reading(image: np.ndarray,
                       detection_result: Any,
                       landmarks: Optional[HandLandmarks] = None,
                       tone: str = "warm",
                       tradition: str = "western",
                       depth: str = "standard") -> Dict[str, Any]:
    """
    Convenience function to create a complete palm reading package.

    Args:
        image: Original RGB image
        detection_result: Result from PalmLineDetectionPipeline
        landmarks: Optional MediaPipe landmarks
        tone: Reading tone (warm, mystical, analytical)
        tradition: Palmistry tradition (western, chinese, indian)
        depth: Reading depth (brief, standard, detailed)

    Returns:
        Dictionary containing features JSON and prompts
    """
    bridge = PalmReadingBridge()

    # Extract features
    features = bridge.process(
        image,
        detection_result.segmentation_masks,
        landmarks
    )

    # Build prompts
    prompts = bridge.build_prompts(features, tone, tradition, depth)

    return {
        "features": features._to_dict(),
        "features_json": features.to_json(),
        "prompts": prompts,
        "detection_quality": features.detection_quality,
        "lines_detected": [l.type.value for l in features.lines]
    }
