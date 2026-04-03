# src/camera.py
import cv2
import threading
import time
from src.config import CAPTURE_WIDTH, CAPTURE_HEIGHT

class ThreadedCamera:
    """
    Captura frames en un hilo separado.
    Esto es critico para procesadores de gama baja (i3) para asegurar que 
    la lectura del driver USB no bloquee el renderizado ni a MediaPipe.
    """
    def __init__(self, src_index=0):
        self.cap = cv2.VideoCapture(src_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        
        self.ret = False
        self.frame = None
        self.running = True
        
        # Iniciar hilo inmediatamente si la camara abrio bien
        if self.cap.isOpened():
            # Leer primeros frames basura
            for _ in range(5):
                self.cap.read()
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
        else:
            print(f"Error: No se pudo abrir la camara en index={src_index}")

    def _update(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    # Invertir el frame de inmediato (efecto espejo natural)
                    self.frame = cv2.flip(frame, 1)
                    self.ret = ret
            time.sleep(0.01) # Pequeño sleep para no freir el procesador

    def get_frame(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
        self.cap.release()
