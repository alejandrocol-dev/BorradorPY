from typing import Tuple, Optional
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class KeepState:
    """Filtro temporal para suavizar cambios de estado y evitar parpadeos (debouncer)."""
    def __init__(self, keepfor: float = 0.2, fps: int = 30, initialState: bool = False):
        self.keepfor = max(1, int(keepfor * fps))
        self.i = self.keepfor
        self.state = initialState

    def update(self, state: bool) -> bool:
        if state == self.state:
            self.i = min(self.keepfor, self.i + 1)
            return self.state
        self.i -= 1
        if self.i < 0:
            self.i = self.keepfor
            self.state = state
        return self.state


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Pulgar
    (0, 5), (5, 6), (6, 7), (7, 8),        # Índice
    (5, 9), (9, 10), (10, 11), (11, 12),   # Medio
    (9, 13), (13, 14), (14, 15), (15, 16), # Anular
    (13, 17), (17, 18), (18, 19), (19, 20),# Meñique
    (0, 17)                                # Base de la palma
]


class HandStateClassifier:
    """Clasificador de gestos de mano con visualización de puntos anatómicos y mediciones."""
    def __init__(
        self, 
        modelPath: str = "hand_landmarker.task", 
        detectionConfidence: float = 0.5, 
        trackingConfidence: float = 0.5
    ) -> None:
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=detectionConfidence,
            min_tracking_confidence=trackingConfidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self.lastLandmarks = None
        self.lastExtendedFingers = 0
        self.lastFingerStates = {}

    def _getExtendedFingers(self, handLandmarks) -> int:
        wrist = handLandmarks[0]
        fingerPairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
        extendedFingers = 0
        self.lastFingerStates = {}
        for tip, pip in fingerPairs:
            isExtended = self._distance(handLandmarks[tip], wrist) > self._distance(handLandmarks[pip], wrist)
            self.lastFingerStates[tip] = isExtended
            if isExtended:
                extendedFingers += 1
        self.lastExtendedFingers = extendedFingers
        return extendedFingers

    def isHandClosed(self, frame: np.ndarray) -> Optional[bool]:
        """Detecta la mano en el frame y retorna True (cerrada/puño), False (abierta) o None si no se detecta."""
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
        result = self._detector.detect(mp_image)

        if not result.hand_landmarks:
            self.lastLandmarks = None
            return None

        self.lastLandmarks = result.hand_landmarks[0]
        extendedFingers = self._getExtendedFingers(self.lastLandmarks)
        if extendedFingers >= 3:
            return False
        return True

    def drawHand(self, frame: np.ndarray) -> np.ndarray:
        """Dibuja el esqueleto de la mano, articulaciones y vectores de medición sobre el frame."""
        if self.lastLandmarks is None:
            return frame

        h, w, _ = frame.shape
        points = []
        for lm in self.lastLandmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))

        wrist_pt = points[0]

        # 1. Dibujar conexiones óseas
        for start_idx, end_idx in HAND_CONNECTIONS:
            pt1 = points[start_idx]
            pt2 = points[end_idx]
            cv2.line(frame, pt1, pt2, (0, 240, 255), 2, cv2.LINE_AA)

        # 2. Dibujar vectores de distancia desde la muñeca a las puntas de los dedos
        for tip_idx in (8, 12, 16, 20):
            tip_pt = points[tip_idx]
            isExt = self.lastFingerStates.get(tip_idx, False)
            lineColor = (0, 255, 100) if isExt else (0, 70, 255)
            cv2.line(frame, wrist_pt, tip_pt, lineColor, 1, cv2.LINE_AA)

        # 3. Dibujar nodos de articulaciones
        for idx, pt in enumerate(points):
            if idx in (8, 12, 16, 20):
                isExt = self.lastFingerStates.get(idx, False)
                color = (0, 255, 0) if isExt else (0, 0, 255)
                cv2.circle(frame, pt, 7, color, -1)
                cv2.circle(frame, pt, 9, (255, 255, 255), 1)
            elif idx == 0:
                cv2.circle(frame, pt, 8, (255, 0, 255), -1)
                cv2.circle(frame, pt, 11, (255, 255, 255), 2)
            else:
                cv2.circle(frame, pt, 4, (255, 200, 0), -1)

        return frame

    @staticmethod
    def _distance(pointA, pointB) -> float:
        dx = pointA.x - pointB.x
        dy = pointA.y - pointB.y
        return (dx * dx + dy * dy) ** 0.5


class SilhouetteSegmenter:
    """Segmentador de silueta humana de alta precisión mediante MediaPipe ImageSegmenter."""
    def __init__(self, modelPath: str = "selfie_segmenter.tflite") -> None:
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.ImageSegmenterOptions(
            base_options=base_options,
            output_confidence_masks=True,
            running_mode=vision.RunningMode.IMAGE
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def getMask(self, frame: np.ndarray, blurEdges: bool = True) -> np.ndarray:
        """
        Retorna una máscara flotante (0.0 a 1.0) de la silueta de la persona
        con bordes suavizados para un alpha blending perfecto.
        """
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
        result = self._segmenter.segment(mp_image)

        if not result.confidence_masks:
            h, w = frame.shape[:2]
            return np.zeros((h, w, 1), dtype=np.float32)

        # confidence_masks[0] es la probabilidad de que cada pixel pertenezca a la persona
        raw_mask = result.confidence_masks[0].numpy_view()
        if raw_mask.ndim == 2:
            raw_mask = np.expand_dims(raw_mask, axis=-1)

        if blurEdges:
            # Suavizado gaussiano para bordes cinematográficos y sin efecto pixelado
            blurred = cv2.GaussianBlur(raw_mask, (7, 7), 0)
            if blurred.ndim == 2:
                blurred = np.expand_dims(blurred, axis=-1)
            return np.clip(blurred, 0.0, 1.0).astype(np.float32)

        return raw_mask


class PersonDetector:
    """Detector clásico de personas para modo caja delimitadora (Bounding Box)."""
    def __init__(
        self, 
        modelPath: str = "efficientdet_lite0.tflite", 
        scoreThreshold: float = 0.5
    ) -> None:
        base_options = python.BaseOptions(model_asset_path=modelPath)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=scoreThreshold,
            running_mode=vision.RunningMode.IMAGE,
            category_allowlist=["person"]
        )
        self._detector = vision.ObjectDetector.create_from_options(options)

    def detectBbox(self, frame: np.ndarray, padding: int = 10) -> Optional[Tuple[int, int, int, int]]:
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frameRGB)
        res = self._detector.detect(mp_image)
        if not res.detections:
            return None

        area = 0
        bbox = None
        for detection in res.detections:
            b = detection.bounding_box
            if b.width * b.height > area:
                area = b.width * b.height
                x = max(0, int(b.origin_x) - padding)
                y = max(0, int(b.origin_y) - padding)
                w = min(width - x, int(b.width) + padding * 2)
                h = height - y
                bbox = (x, y, w, h)

        return bbox

    def drawBox(
        self, 
        frame: np.ndarray, 
        bbox: Tuple[int, int, int, int], 
        color: Tuple[int, int, int] = (0, 255, 0), 
        thickness: int = 2
    ) -> np.ndarray:
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
        return frame