"""
gesture_arm.vision.tracker
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hand detection and normalized feature extraction.

Wraps cvzone / MediaPipe and exposes a clean, typed interface so the
rest of the system never touches raw landmark lists directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from cvzone.HandTrackingModule import HandDetector

logger = logging.getLogger(__name__)

# MediaPipe landmark indices (for documentation clarity)
WRIST           = 0
INDEX_MCP       = 5
MIDDLE_MCP      = 9
THUMB_TIP       = 4
INDEX_TIP       = 8
MIDDLE_TIP      = 12
NUM_LANDMARKS   = 21


@dataclass(frozen=True)
class HandState:
    """
    Immutable snapshot of one detected hand.

    Attributes:
        hand_type:      "Left" or "Right"
        landmarks:      (21, 3) float32 array — raw pixel (x, y, z) coords
        features:       (42,) float32 array — normalized (x/W, y/H) for LSTM input
        palm_center:    (x, y) pixel position of landmark 9 (palm centre)
        pinch_distance: Euclidean pixel distance between thumb tip and index tip
        fingers_up:     list[bool] — [thumb, index, middle, ring, pinky] extended
        is_fist:        True when all five fingers are closed
    """
    hand_type: str
    landmarks: np.ndarray
    features: np.ndarray
    palm_center: Tuple[float, float]
    pinch_distance: float
    fingers_up: List[bool]
    is_fist: bool


@dataclass(frozen=True)
class TrackerOutput:
    """
    Full output from one video frame.

    Attributes:
        frame:      BGR image with any requested overlays drawn on it
        left:       HandState for the left hand, or None
        right:      HandState for the right hand, or None
    """
    frame: np.ndarray
    left: Optional[HandState]
    right: Optional[HandState]


class HandTracker:
    """
    Wraps cvzone HandDetector and emits HandState / TrackerOutput objects.

    Usage::

        tracker = HandTracker(cfg.vision)
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            output = tracker.process(frame)
            if output.right:
                print(output.right.palm_center)
    """

    def __init__(
        self,
        detection_confidence: float = 0.8,
        max_hands: int = 2,
        frame_width: int = 1280,
        frame_height: int = 720,
        draw_landmarks: bool = False,
    ) -> None:
        self._detector = HandDetector(
            detectionCon=detection_confidence,
            maxHands=max_hands,
        )
        self._fw = frame_width
        self._fh = frame_height
        self._draw = draw_landmarks
        logger.info(
            "HandTracker initialized (conf=%.2f, max=%d, res=%dx%d)",
            detection_confidence, max_hands, frame_width, frame_height,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> TrackerOutput:
        """
        Detect hands in one BGR frame and return a TrackerOutput.

        Args:
            frame: BGR image from cv2.VideoCapture.read()

        Returns:
            TrackerOutput with left/right HandState (or None if not detected).
        """
        hands, annotated = self._detector.findHands(frame, draw=self._draw)

        left: Optional[HandState] = None
        right: Optional[HandState] = None

        for hand in hands:
            state = self._build_state(hand)
            if state.hand_type == "Left":
                left = state
            else:
                right = state

        return TrackerOutput(frame=annotated, left=left, right=right)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _build_state(self, hand: dict) -> HandState:
        lm = hand["lmList"]                          # list of 21 [x, y, z]
        landmarks = np.array(lm, dtype=np.float32)   # (21, 3)

        features = self._normalize(landmarks)
        palm_center = (float(lm[MIDDLE_MCP][0]), float(lm[MIDDLE_MCP][1]))
        pinch_dist = float(np.linalg.norm(
            landmarks[THUMB_TIP, :2] - landmarks[INDEX_TIP, :2]
        ))

        fingers = self._detector.fingersUp(hand)     # [0/1 × 5]
        fingers_up = [bool(f) for f in fingers]
        is_fist = not any(fingers_up)

        return HandState(
            hand_type=hand["type"],
            landmarks=landmarks,
            features=features,
            palm_center=palm_center,
            pinch_distance=pinch_dist,
            fingers_up=fingers_up,
            is_fist=is_fist,
        )

    def _normalize(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Flatten 21 landmarks to a 42-element normalized vector.

        x_t = [x1/W, y1/H, x2/W, y2/H, ..., x21/W, y21/H]

        Paper Section IV-A: normalization makes the feature vector
        resolution-independent so the same model works on any camera.
        """
        xy = landmarks[:, :2].copy()                 # (21, 2)
        xy[:, 0] /= self._fw                         # normalize x to [0, 1]
        xy[:, 1] /= self._fh                         # normalize y to [0, 1]
        return xy.flatten().astype(np.float32)        # (42,)
