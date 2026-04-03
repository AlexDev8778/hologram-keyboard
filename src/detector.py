# src/detector.py
import mediapipe as mp
import time
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.config import PINCH_THRESHOLD, MIN_CONFIDENCE, CURSOR_SMOOTHING

class HandDetector:
    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=MIN_CONFIDENCE,
            min_hand_presence_confidence=MIN_CONFIDENCE,
            min_tracking_confidence=MIN_CONFIDENCE,
            running_mode=vision.RunningMode.IMAGE
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_process_time = 0
        self.history = {} # Guardar el ultimo punto suavizado por indice de mano

    def process_frame(self, frame_ext):
        """Procesa el frame y devuelve los landmarks normalizados."""
        # Rate-limiting simple para skip_frames: 
        # Forzar un poco de espera simulada o directamente no bloquear todo,
        # pero como usamos IMAGE mode, procesamos instantaneamente.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_ext)
        result = self.detector.detect(mp_image)
        return result

    def get_pointer_coordinates(self, hand_landmarks, img_width, img_height, hand_id="default"):
        """
        Extrae la coordenada del puntero (Punto medio entre Pulgar e Indice).
        Retorna (x_pixel, y_pixel).
        """
        if not hand_landmarks:
            return None
        
        index_tip = hand_landmarks[8]
        thumb_tip = hand_landmarks[4]
        
        # El puntero original saltaba al hacer pinch porque el indice baja hacia el pulgar.
        # Si usamos el centro entre indice y pulgar, el puntero se vuelve super estable.
        raw_x = int(((index_tip.x + thumb_tip.x) / 2) * img_width)
        raw_y = int(((index_tip.y + thumb_tip.y) / 2) * img_height)

        # Aplicar suavizado (Exponential Moving Average) para evitar temblor
        if hand_id not in self.history:
            self.history[hand_id] = (raw_x, raw_y)
        else:
            prev_x, prev_y = self.history[hand_id]
            
            # Anti-Teletransporte: Si la mano salto bruscamente de posicion (>150px),
            # cortamos el suavizado de raiz para no verla flotando de un lado a otro.
            dist = math.sqrt((raw_x - prev_x)**2 + (raw_y - prev_y)**2)
            if dist > 150:
                self.history[hand_id] = (raw_x, raw_y)
            else:
                smooth_x = int((raw_x * CURSOR_SMOOTHING) + (prev_x * (1.0 - CURSOR_SMOOTHING)))
                smooth_y = int((raw_y * CURSOR_SMOOTHING) + (prev_y * (1.0 - CURSOR_SMOOTHING)))
                self.history[hand_id] = (smooth_x, smooth_y)
            
        return self.history[hand_id]

    def is_pinching(self, hand_landmarks):
        """
        Determina si el pulgar (4) y el indice (8) estan haciendo pinch ("click").
        """
        if not hand_landmarks:
            return False
            
        thumb_tip = hand_landmarks[4]
        index_tip = hand_landmarks[8]
        
        # Calcular distancia euclidiana 3D
        dist = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2 + 
            (thumb_tip.z - index_tip.z)**2
        )
        
        return dist < PINCH_THRESHOLD
