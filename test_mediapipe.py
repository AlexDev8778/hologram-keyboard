"""
HoloGesture — Script de Validación de Hardware
================================================
Prueba si MediaPipe Hands corre acceptable en tu máquina.

Métricas que muestra:
  - FPS del video (verde = ok, amarillo = limite, rojo = lento)
  - Latencia de MediaPipe por frame (cuánto tarda en detectar)
  - Distancia de pinch en tiempo real (pulgar ↔ índice)
  - Estado del gesto detectado

Controles:
  Q / ESC  → Salir
  D        → Toggle modo debug (landmarks completos)
  S        → Guardar snapshot del diagnóstico en pantalla
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sys

# ──────────────────────────────────────────────
# Configuración para i3-2100
# ──────────────────────────────────────────────
CAPTURE_WIDTH     = 640
CAPTURE_HEIGHT    = 480
CAMERA_INDEX      = 0          # Cambiá a 1 si tu webcam no es la default
MODEL_COMPLEXITY  = 0          # 0 = Lite (recomendado para tu hardware)
MAX_HANDS         = 2
MIN_DETECTION_CF  = 0.7
MIN_TRACKING_CF   = 0.7
PINCH_THRESHOLD   = 0.05       # Distancia normalizada para pinch

# Colores (BGR)
C_CYAN   = (255, 220,   0)   # Cian holográfico
C_GREEN  = (  0, 220,   0)
C_YELLOW = (  0, 200, 255)
C_RED    = (  0,  50, 255)
C_WHITE  = (255, 255, 255)
C_BLACK  = (  0,   0,   0)
C_PINCH  = (200, 255, 200)


# ──────────────────────────────────────────────
# Helpers de UI
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
                 thickness=1, bg_color=(20, 20, 20), padding=5):
    """Texto con fondo sólido para legibilidad."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(frame,
                  (x - padding, y - th - padding),
                  (x + tw + padding, y + padding),
                  bg_color, -1)
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_landmark_dot(frame, lm, w, h, color, radius=5):
    cx, cy = int(lm.x * w), int(lm.y * h)
    cv2.circle(frame, (cx, cy), radius, color, -1)
    return cx, cy

def draw_connections(frame, landmarks, connections, w, h, color=(80, 80, 80), thickness=1):
    for start_idx, end_idx in connections:
        s = landmarks[start_idx]
        e = landmarks[end_idx]
        sx, sy = int(s.x * w), int(s.y * h)
        ex, ey = int(e.x * w), int(e.y * h)
        cv2.line(frame, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)

def pinch_distance(lm_list):
    """Distancia euclidiana 3D entre pulgar (4) e índice (8), normalizado."""
    t = lm_list[4]
    i = lm_list[8]
    return np.sqrt((t.x - i.x)**2 + (t.y - i.y)**2 + (t.z - i.z)**2)


# ──────────────────────────────────────────────
# FPS counter con ventana deslizante
# ──────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=20):
        self._times = []
        self._window = window

    def tick(self):
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)

    def fps(self):
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        if span <= 0:
            return 0.0
        return (len(self._times) - 1) / span


# ──────────────────────────────────────────────
# Diagnóstico acumulativo
# ──────────────────────────────────────────────
class Diagnostics:
    def __init__(self):
        self.latencies   = []
        self.fps_samples = []
        self.frames_run  = 0
        self.hands_seen  = 0
        self.pinches     = 0

    def record_latency(self, ms):
        self.latencies.append(ms)
        self.frames_run += 1

    def record_fps(self, fps):
        self.fps_samples.append(fps)

    def report(self):
        if not self.latencies:
            print("No se procesaron frames.")
            return
        avg_lat = sum(self.latencies) / len(self.latencies)
        max_lat = max(self.latencies)
        avg_fps = sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else 0

        print("\n" + "═" * 50)
        print("  DIAGNÓSTICO FINAL — HoloGesture Test")
        print("═" * 50)
        print(f"  Frames procesados:   {self.frames_run}")
        print(f"  FPS promedio:        {avg_fps:.1f}")
        print(f"  Latencia MediaPipe:")
        print(f"    Promedio:          {avg_lat:.1f} ms")
        print(f"    Máxima:            {max_lat:.1f} ms")
        print(f"  Manos detectadas:    {self.hands_seen} frames")
        print(f"  Pinches detectados:  {self.pinches}")
        print()
        if avg_fps >= 18:
            print("  ✅ RESULTADO: Hardware OK para el proyecto.")
        elif avg_fps >= 12:
            print("  ⚠️  RESULTADO: Hardware limite. Funcionará con optimizaciones.")
            print("     → Activar skip_frames=1 en el proyecto real.")
        else:
            print("  ❌ RESULTADO: Hardware muy lento para tiempo real fluido.")
            print("     → Considerar resolución 320x240 o skip_frames=2.")
        print("═" * 50 + "\n")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    mp_hands    = mp.solutions.hands
    mp_drawing  = mp.solutions.drawing_utils
    mp_styles   = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        model_complexity    = MODEL_COMPLEXITY,
        max_num_hands       = MAX_HANDS,
        min_detection_confidence = MIN_DETECTION_CF,
        min_tracking_confidence  = MIN_TRACKING_CF
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir la cámara (index={CAMERA_INDEX}).")
        print("  Probá cambiando CAMERA_INDEX a 1 o 2 en la config del script.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)

    fps_counter = FPSCounter()
    diag        = Diagnostics()
    debug_mode  = False
    snapshot_n  = 0

    print("\nHoloGesture — Validación de Hardware")
    print(f"  Resolución: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT}")
    print(f"  Modelo MediaPipe: complexity={MODEL_COMPLEXITY} (Lite)")
    print(f"  Cámara index: {CAMERA_INDEX}")
    print("\nControles: Q/ESC=Salir  D=Debug  S=Snapshot\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: No se pudo leer frame de la cámara.")
            break

        frame = cv2.flip(frame, 1)   # Espejo (más natural)
        h, w  = frame.shape[:2]
        frame_rgb = frame[:, :, ::-1]   # BGR→RGB sin copia

        # ── MediaPipe ────────────────────────────
        t0     = time.perf_counter()
        result = hands.process(frame_rgb)
        mp_ms  = (time.perf_counter() - t0) * 1000

        fps_counter.tick()
        current_fps = fps_counter.fps()

        diag.record_latency(mp_ms)
        diag.record_fps(current_fps)

        # ── Dibujar landmarks ────────────────────
        pinch_active = False
        pinch_dist_val = 1.0

        if result.multi_hand_landmarks:
            diag.hands_seen += 1

            for hand_idx, hand_lm in enumerate(result.multi_hand_landmarks):
                lm = hand_lm.landmark

                if debug_mode:
                    # Dibujo completo de MediaPipe
                    mp_drawing.draw_landmarks(
                        frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )
                else:
                    # Dibujo minimalista: solo conexiones y puntos clave
                    draw_connections(
                        frame, lm, mp_hands.HAND_CONNECTIONS, w, h,
                        color=(60, 60, 60), thickness=1
                    )
                    # Puntos clave destacados
                    draw_landmark_dot(frame, lm[0],  w, h, C_CYAN,   4)  # Muñeca
                    draw_landmark_dot(frame, lm[4],  w, h, C_YELLOW, 7)  # Pulgar tip
                    draw_landmark_dot(frame, lm[8],  w, h, C_CYAN,   7)  # Índice tip
                    draw_landmark_dot(frame, lm[12], w, h, C_CYAN,   5)  # Medio tip
                    draw_landmark_dot(frame, lm[16], w, h, C_CYAN,   5)  # Anular tip
                    draw_landmark_dot(frame, lm[20], w, h, C_CYAN,   5)  # Meñique tip

                # Línea pulgar ↔ índice (feedback de pinch)
                t4x, t4y = int(lm[4].x * w), int(lm[4].y * h)
                t8x, t8y = int(lm[8].x * w), int(lm[8].y * h)

                d = pinch_distance(lm)
                if hand_idx == 0:
                    pinch_dist_val = d

                is_pinch = d < PINCH_THRESHOLD
                line_color = C_PINCH if is_pinch else (100, 100, 100)
                cv2.line(frame, (t4x, t4y), (t8x, t8y), line_color, 2, cv2.LINE_AA)

                if is_pinch:
                    pinch_active = True
                    diag.pinches += 1
                    # Círculo de confirmación de pinch
                    mid_x = (t4x + t8x) // 2
                    mid_y = (t4y + t8y) // 2
                    cv2.circle(frame, (mid_x, mid_y), 14, C_PINCH, 2)
                    cv2.circle(frame, (mid_x, mid_y),  4, C_PINCH, -1)

        # ── HUD de diagnóstico ───────────────────
        # FPS
        draw_text_bg(frame, f"FPS: {current_fps:.1f}",
                     (10, 25), color=fps_color(current_fps))

        # Latencia MediaPipe
        draw_text_bg(frame, f"MP: {mp_ms:.0f}ms",
                     (10, 55), color=latency_color(mp_ms))

        # Modelo
        draw_text_bg(frame, f"Model: Lite (complexity={MODEL_COMPLEXITY})",
                     (10, 85), color=C_WHITE, font_scale=0.45)

        # Manos detectadas
        n_hands = len(result.multi_hand_landmarks) if result.multi_hand_landmarks else 0
        draw_text_bg(frame, f"Manos: {n_hands}/{MAX_HANDS}",
                     (10, 115), color=C_CYAN if n_hands > 0 else (120, 120, 120))

        # Distancia de pinch
        draw_text_bg(frame, f"Pinch dist: {pinch_dist_val:.3f}  {'← PINCH!' if pinch_active else ''}",
                     (10, 145), color=C_PINCH if pinch_active else C_WHITE)

        # Debug mode indicator
        if debug_mode:
            draw_text_bg(frame, "DEBUG ON", (w - 120, 25), color=C_YELLOW)

        # Barra de FPS tipo indicador
        bar_w  = int(min(current_fps / 30.0, 1.0) * 200)
        bar_x  = 10
        bar_y  = h - 15
        cv2.rectangle(frame, (bar_x, bar_y - 8), (bar_x + 200, bar_y), (40, 40, 40), -1)
        cv2.rectangle(frame, (bar_x, bar_y - 8), (bar_x + bar_w, bar_y), fps_color(current_fps), -1)
        draw_text_bg(frame, "FPS 0       15      30",
                     (bar_x, bar_y - 12), font_scale=0.38, color=(160, 160, 160),
                     bg_color=(20, 20, 20))

        # Instrucciones
        draw_text_bg(frame, "Q/ESC: Salir | D: Debug | S: Snapshot",
                     (10, h - 25), font_scale=0.42, color=(180, 180, 180))

        cv2.imshow("HoloGesture — Validacion Hardware", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q o ESC
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
        elif key == ord('s'):
            fname = f"snapshot_{snapshot_n:03d}.png"
            cv2.imwrite(fname, frame)
            snapshot_n += 1
            print(f"Snapshot guardado: {fname}")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    diag.report()


if __name__ == "__main__":
    main()
