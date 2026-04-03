"""
HoloGesture — Script de Validacion de Hardware
================================================
Prueba si MediaPipe Hands corre aceptable en tu maquina.
Compatible con MediaPipe >= 0.10 (Tasks API).

Metricas que muestra:
  - FPS del video (verde=ok, amarillo=limite, rojo=lento)
  - Latencia de MediaPipe por frame
  - Distancia de pinch pulgar <> indice en tiempo real
  - Estado del gesto detectado

Controles:
  Q / ESC  -> Salir y ver diagnostico final
  D        -> Toggle modo debug (todos los landmarks)
  S        -> Guardar snapshot PNG
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
CAPTURE_WIDTH    = 640
CAPTURE_HEIGHT   = 480
CAMERA_INDEX     = None   # None = auto-detectar

# DroidCam: pone la IP que muestra la app del telefono (ej: "192.168.1.5")
# Dejalo en None para usar el driver virtual o auto-deteccion
DROIDCAM_IP      = None
DROIDCAM_PORT    = 4747
PINCH_THRESHOLD  = 0.05
MODEL_PATH       = "hand_landmarker.task"
MODEL_URL        = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

# Colores BGR
C_CYAN   = (255, 220,   0)
C_GREEN  = (  0, 220,   0)
C_YELLOW = (  0, 200, 255)
C_RED    = (  0,  50, 255)
C_WHITE  = (255, 255, 255)
C_PINCH  = (180, 255, 180)

# Conexiones de la mano (indices de landmarks)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # Pulgar
    (0,5),(5,6),(6,7),(7,8),       # Indice
    (0,9),(9,10),(10,11),(11,12),  # Medio
    (0,13),(13,14),(14,15),(15,16),# Anular
    (0,17),(17,18),(18,19),(19,20),# Menique
    (5,9),(9,13),(13,17),          # Palma
]


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def fps_color(fps):
    if fps >= 18: return C_GREEN
    if fps >= 12: return C_YELLOW
    return C_RED

def latency_color(ms):
    if ms < 60:  return C_GREEN
    if ms < 120: return C_YELLOW
    return C_RED

def draw_text_bg(frame, text, pos, font_scale=0.55, color=C_WHITE,
                 thickness=1, bg=(20,20,20), padding=5):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame, (x-padding, y-th-padding), (x+tw+padding, y+padding), bg, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_connections(frame, landmarks, w, h, color=(60,60,60)):
    for s, e in HAND_CONNECTIONS:
        sx, sy = int(landmarks[s].x * w), int(landmarks[s].y * h)
        ex, ey = int(landmarks[e].x * w), int(landmarks[e].y * h)
        cv2.line(frame, (sx,sy), (ex,ey), color, 1, cv2.LINE_AA)

def draw_dot(frame, lm, w, h, color, r=5):
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(frame, (cx, cy), r, color, -1)
    return cx, cy

def pinch_dist(lm):
    t, i = lm[4], lm[8]
    return np.sqrt((t.x-i.x)**2 + (t.y-i.y)**2 + (t.z-i.z)**2)

def download_model():
    if os.path.exists(MODEL_PATH):
        print(f"  Modelo encontrado: {MODEL_PATH}")
        return
    print(f"  Descargando modelo MediaPipe (~26MB)...")
    print(f"  URL: {MODEL_URL}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  Descarga completa.")
    except Exception as e:
        print(f"\nERROR descargando modelo: {e}")
        print("Descargalo manualmente desde:")
        print(f"  {MODEL_URL}")
        print(f"Y ponelo en: {os.path.abspath(MODEL_PATH)}")
        sys.exit(1)


# ──────────────────────────────────────────────
# FPS counter
# ──────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=20):
        self._times = []
        self._window = window

    def tick(self):
        self._times.append(time.perf_counter())
        if len(self._times) > self._window:
            self._times.pop(0)

    def fps(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


# ──────────────────────────────────────────────
# Diagnostico
# ──────────────────────────────────────────────
class Diagnostics:
    def __init__(self):
        self.latencies   = []
        self.fps_samples = []
        self.frames_run  = 0
        self.hands_seen  = 0
        self.pinches     = 0

    def record(self, lat_ms, fps):
        self.latencies.append(lat_ms)
        self.fps_samples.append(fps)
        self.frames_run += 1

    def report(self):
        if not self.latencies:
            print("No se procesaron frames.")
            return
        avg_lat = sum(self.latencies) / len(self.latencies)
        max_lat = max(self.latencies)
        avg_fps = sum(self.fps_samples) / len(self.fps_samples)

        print("\n" + "=" * 50)
        print("  DIAGNOSTICO FINAL - HoloGesture")
        print("=" * 50)
        print(f"  Frames procesados : {self.frames_run}")
        print(f"  FPS promedio      : {avg_fps:.1f}")
        print(f"  Latencia MediaPipe:")
        print(f"    Promedio        : {avg_lat:.1f} ms")
        print(f"    Maxima          : {max_lat:.1f} ms")
        print(f"  Manos detectadas  : {self.hands_seen} frames")
        print(f"  Pinches           : {self.pinches}")
        print()
        if avg_fps >= 18:
            print("  OK: Hardware suficiente para el proyecto.")
        elif avg_fps >= 12:
            print("  AVISO: Hardware en el limite.")
            print("  -> Se usara skip_frames=1 en produccion.")
        else:
            print("  LENTO: Considerar bajar resolucion a 320x240.")
        print("=" * 50 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    print("\nHoloGesture - Validacion de Hardware")
    print(f"  MediaPipe version: {mp.__version__}")
    print(f"  Resolucion: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"  Camara index: {CAMERA_INDEX}")
    print()

    # Descargar modelo si no existe
    download_model()

    # Configurar HandLandmarker (nueva API Tasks)
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = HandLandmarkerOptions(
        base_options=base_options,
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    detector = HandLandmarker.create_from_options(options)

    # ── Deteccion de camara ──────────────────────────────────────────────────
    # Estrategia:
    #   1. DroidCam HTTP stream (la mas confiable, sin driver)
    #   2. Virtual driver por indice (webcam fisica o DroidCam driver)

    cap = None

    # 1. Intentar DroidCam HTTP stream si hay una IP configurada
    if DROIDCAM_IP:
        url = f"http://{DROIDCAM_IP}:{DROIDCAM_PORT}/video"
        print(f"  Probando DroidCam HTTP: {url}")
        _test = cv2.VideoCapture(url)
        if _test.isOpened():
            ret_test, _ = _test.read()
            if ret_test:
                cap = _test
                print(f"  DroidCam HTTP OK: {url}")
            else:
                _test.release()
                print("  DroidCam HTTP: abre pero sin frame. Verifica que la app este activa.")
        else:
            print("  DroidCam HTTP: no responde. Verifica IP y que DroidCam este corriendo.")

    # 2. Fallback: buscar camara por indice saltando las 'pantallas verdes' de DroidCam
    if cap is None:
        print("  Buscando camara por indice (0, 1, 2, 3)...")
        for try_idx in range(4):
            if cap is not None:
                break
            for backend, bname in [(cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "AUTO")]:
                _test = cv2.VideoCapture(try_idx, backend)
                if _test.isOpened():
                    # Leer unos cuantos frames porque el 1ro suele ser negro (inicializacion)
                    for _ in range(5):
                        ret_test, frame_test = _test.read()
                    
                    if ret_test and frame_test is not None:
                        # Verificar si es la pantalla de espera verde de DroidCam
                        g_mean = frame_test[:,:,1].mean()
                        r_mean = frame_test[:,:,0].mean()
                        if g_mean > r_mean * 1.5 and frame_test.mean() > 5:
                            print(f"  Ignorando index={try_idx} (Devuelve pantalla verde de espera DroidCam)")
                            _test.release()
                            continue  # Sigue con la proxima camara
                            
                        # Es una camara valida con video
                        cap = _test
                        print(f"  Camara valida encontrada: index={try_idx} backend={bname}")
                        break
                    _test.release()

    if cap is None:
        print("\nERROR: No se encontro ninguna camara.")
        print("  Opciones:")
        print("  1. Pone la IP de DroidCam en DROIDCAM_IP al inicio del script")
        print("  2. Conecta una webcam USB")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    # Warmup: descartar primeros frames mientras el driver virtual se inicializa
    print("  Calentando camara (30 frames)...", end="", flush=True)
    for _ in range(30):
        cap.read()
    print(" OK")

    # Detectar si DroidCam envía la pantalla verde de desconectado
    ret_check, frame_check = cap.read()
    if ret_check and frame_check is not None:
        mean_val = frame_check.mean()
        g_mean = frame_check[:,:,1].mean()
        r_mean = frame_check[:,:,0].mean()
        print(f"  Frame check: shape={frame_check.shape} mean={mean_val:.1f} R={r_mean:.0f} G={g_mean:.0f}")
        if g_mean > r_mean * 1.5 and mean_val > 5:
            print("  AVISO: Frame verde detectado. Esto usualmente significa que DroidCam NO esta recibiendo video del telefono.")
        else:
            print("  Frame OK.")
    else:
        print("  AVISO: No se pudo leer frame de verificacion.")

    fps_counter = FPSCounter()
    diag        = Diagnostics()
    debug_mode  = False
    snapshot_n  = 0

    print("\nControles: Q/ESC=Salir  D=Debug  S=Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: No se pudo leer frame.")
            break

        frame = cv2.flip(frame, 1)

        h, w  = frame.shape[:2]

        # Convertir a MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame[:, :, ::-1].copy()  # BGR -> RGB
        )

        # Deteccion
        t0     = time.perf_counter()
        result = detector.detect(mp_image)
        mp_ms  = (time.perf_counter() - t0) * 1000

        fps_counter.tick()
        current_fps = fps_counter.fps()
        diag.record(mp_ms, current_fps)

        # Dibujar landmarks
        pinch_active   = False
        pinch_dist_val = 1.0
        n_hands        = len(result.hand_landmarks)

        if n_hands > 0:
            diag.hands_seen += 1

        for hand_idx, hand_lm in enumerate(result.hand_landmarks):
            lm = hand_lm  # lista de NormalizedLandmark

            if debug_mode:
                # Dibujar todos los puntos en modo debug
                for idx, point in enumerate(lm):
                    px, py = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (px, py), 3, C_CYAN, -1)
                    cv2.putText(frame, str(idx), (px+4, py-4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, C_WHITE, 1)
            
            # Conexiones
            draw_connections(frame, lm, w, h, color=(60, 60, 60))

            # Puntos clave
            draw_dot(frame, lm[0],  w, h, C_CYAN,   4)   # Muneca
            draw_dot(frame, lm[4],  w, h, C_YELLOW, 7)   # Pulgar tip
            draw_dot(frame, lm[8],  w, h, C_CYAN,   7)   # Indice tip
            draw_dot(frame, lm[12], w, h, C_CYAN,   5)
            draw_dot(frame, lm[16], w, h, C_CYAN,   5)
            draw_dot(frame, lm[20], w, h, C_CYAN,   5)

            # Linea pinch
            t4x, t4y = int(lm[4].x * w), int(lm[4].y * h)
            t8x, t8y = int(lm[8].x * w), int(lm[8].y * h)

            d = pinch_dist(lm)
            if hand_idx == 0:
                pinch_dist_val = d

            is_pinch   = d < PINCH_THRESHOLD
            line_color = C_PINCH if is_pinch else (100, 100, 100)
            cv2.line(frame, (t4x, t4y), (t8x, t8y), line_color, 2, cv2.LINE_AA)

            if is_pinch:
                pinch_active = True
                diag.pinches += 1
                mid_x = (t4x + t8x) // 2
                mid_y = (t4y + t8y) // 2
                cv2.circle(frame, (mid_x, mid_y), 14, C_PINCH, 2)
                cv2.circle(frame, (mid_x, mid_y),  4, C_PINCH, -1)

        # HUD
        draw_text_bg(frame, f"FPS: {current_fps:.1f}",
                     (10, 25), color=fps_color(current_fps))
        draw_text_bg(frame, f"MediaPipe: {mp_ms:.0f}ms",
                     (10, 55), color=latency_color(mp_ms))
        draw_text_bg(frame, f"Manos: {n_hands}/2",
                     (10, 85), color=C_CYAN if n_hands > 0 else (120,120,120))
        draw_text_bg(frame,
                     f"Pinch: {pinch_dist_val:.3f}  {'<< PINCH!' if pinch_active else ''}",
                     (10, 115), color=C_PINCH if pinch_active else C_WHITE)

        if debug_mode:
            draw_text_bg(frame, "DEBUG ON", (w-120, 25), color=C_YELLOW)

        # Barra FPS
        bar_w = int(min(current_fps / 30.0, 1.0) * 200)
        cv2.rectangle(frame, (10, h-14), (210, h-6), (40,40,40), -1)
        cv2.rectangle(frame, (10, h-14), (10+bar_w, h-6), fps_color(current_fps), -1)
        draw_text_bg(frame, "Q/ESC Salir | D Debug | S Snapshot",
                     (10, h-22), font_scale=0.4, color=(160,160,160))

        cv2.imshow("HoloGesture - Validacion", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('s'):
            fname = f"snapshot_{snapshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            snapshot_n += 1
            print(f"Snapshot: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    diag.report()


if __name__ == "__main__":
    main()
