import cv2
import numpy as np
from typing import Optional
from utils import PersonDetector, HandStateClassifier, KeepState


class Borrame:
    """
    Motor principal de invisibilidad interactiva.
    Detecta la presencia humana y el gesto de la mano, aplicando
    el reemplazo directo y limpio del fondo sin artefactos ni retrasos.
    """
    def __init__(
        self, 
        bgImage: Optional[np.ndarray] = None, 
        registerFor: float = 0.2, 
        fps: int = 30
    ) -> None:
        self.personDetector = PersonDetector()
        self.handClassifier = HandStateClassifier()
        self.stateFilter = KeepState(keepfor=registerFor, fps=fps, initialState=False)
        self.bgImage: Optional[np.ndarray] = None

        if bgImage is not None:
            self.loadBackgroundImage(bgImage)

    def loadBackgroundImage(self, bgImage: np.ndarray) -> None:
        self.bgImage = bgImage.copy()

    def processFrame(
        self, 
        frame: np.ndarray, 
        drawBbox: bool = True,
        drawHand: bool = True
    ) -> np.ndarray:
        """
        Procesa el fotograma actual. Si el puño está cerrado, reemplaza la región
        del cuerpo por la imagen de fondo limpia de forma instantánea.
        """
        if self.bgImage is None:
            raise ValueError("No se ha configurado la imagen de fondo. Use loadBackgroundImage().")

        # Ajustar dimensiones del fondo si difieren del frame actual
        if self.bgImage.shape != frame.shape:
            self.bgImage = cv2.resize(self.bgImage, (frame.shape[1], frame.shape[0]))

        # 1. Detección del estado de la mano (puño cerrado o mano abierta)
        rawHandClosed = self.handClassifier.isHandClosed(frame)
        if rawHandClosed is None:
            rawHandClosed = False
        
        handClosed = self.stateFilter.update(rawHandClosed)

        outputFrame = frame.copy()
        bbox = self.personDetector.detectBbox(frame)

        # 2. Efecto de Invisibilidad: Si el puño está cerrado, reemplazar el área por el fondo
        if handClosed and bbox is not None:
            x, y, w, h = bbox
            outputFrame[y:y+h, x:x+w] = self.bgImage[y:y+h, x:x+w]

        # 3. Dibujar el contorno verde del cuadro si está activado
        if drawBbox and bbox is not None:
            outputFrame = self.personDetector.drawBox(outputFrame, bbox, color=(0, 255, 0), thickness=2)

        # 4. Dibujar el esqueleto de la mano si está activado
        if drawHand:
            outputFrame = self.handClassifier.drawHand(outputFrame)

        return outputFrame


