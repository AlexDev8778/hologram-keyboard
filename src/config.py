# src/config.py
import cv2

# --- CONFIGURACION DE CAMARA ---
CAPTURE_WIDTH    = 640
CAPTURE_HEIGHT   = 480
TARGET_FPS       = 30

# --- CONFIGURACION DE RENDERIZADO ---
# Colores (B, G, R) para OpenCV
COLOR_TEXT       = (255, 255, 255)
COLOR_BG_NORMAL  = (30, 30, 30)
COLOR_BG_HOVER   = (60, 60, 60)
COLOR_BG_PRESS   = (100, 200, 100)
COLOR_GLOW       = (0, 255, 255)
COLOR_POINTER    = (0, 255, 0)
COLOR_POINTER_ACT= (0, 0, 255)

# Alpha blending para efecto holograma (0.0 = transparente, 1.0 = opaco)
ALPHA_KEYBOARD   = 0.65

# --- CONFIGURACION DE DETECCION ---
PINCH_THRESHOLD  = 0.05    # Distancia entre pulgar e indice para considerar "click"
COOLDOWN_MS      = 300     # Tiempo muerto luego de presionar una tecla para no spamearla
MIN_CONFIDENCE   = 0.70    # Balance perfecto: no saltan tantos falsos positivos y sigue rastreando comodo
CURSOR_SMOOTHING = 0.45    # Nivel de suavizado del cursor (1.0 = instantaneo, 0.1 = muy suave/lento)

# --- LAYOUT DEL TECLADO ---
KEYBOARD_LAYOUT = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", "Ñ"],
    ["Z", "X", "C", "V", "B", "N", "M", "BKSP"],
    ["SPACE"]
]

KEY_SIZE         = 50
KEY_SPACING      = 10
START_X          = 40
START_Y          = 150
