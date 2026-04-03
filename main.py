import cv2
import numpy as np
import time
from src.camera import ThreadedCamera
from src.detector import HandDetector
from src.keyboard import VirtualKeyboard
from src.config import (
    COLOR_POINTER, COLOR_POINTER_ACT, ALPHA_KEYBOARD
)

def apply_glow_overlay(bg_img, overlay_img, alpha):
    """
    Aplica el 'Holograma' combinando un overlay sobre un background.
    Usa el addWeighted simple para performance en CPU.
    """
    # Combinamos el frame original (bg) con el render del teclado (overlay)
    return cv2.addWeighted(bg_img, 1.0, overlay_img, alpha, 0)

def main():
    print("Iniciando Hologram Keyboard...")
    print("Cargando modelo de MediaPipe (puede tardar un momento)...")
    
    # 1. Inicializar Modulos
    camera = ThreadedCamera(src_index=0)  # Utiliza el index 0, asumimos DroidCam o Webcam
    detector = HandDetector()
    keyboard = VirtualKeyboard()

    print("Cargado. Presiona Q para salir.")
    
    # Variables FPS
    prev_time = time.time()
    skip_frames = 1 # Procesar MediaPipe 1 de cada 2 frames (Ideal CPU gama baja)
    frame_count = 0
    last_landmarks = []
    
    while True:
        ret, frame = camera.get_frame()
        if not ret or frame is None:
            time.sleep(0.01)
            continue
            
        frame_count += 1
        h, w = frame.shape[:2]

        # Evitar el frame verde de vacio de DroidCam
        if frame.mean() > 5 and frame[:,:,1].mean() > frame[:,:,0].mean() * 1.5:
            # Es verde, saltamos procesamiento
            cv2.putText(frame, "Esperando video...", (50,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
            cv2.imshow("Hologram Keyboard", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 2. Procesamiento de IA (con Rate Limiting)
        # Optimizacion clave para el i3-2100: No correr MediaPipe todos los frames
        if frame_count % (skip_frames + 1) == 0:
            result = detector.process_frame(frame)
            if hasattr(result, 'hand_landmarks'):
                last_landmarks = result.hand_landmarks
                last_handedness = result.handedness
            else:
                last_landmarks = []
                last_handedness = []

        # Extraer punteros y evaluar clicks
        pointers = []
        is_pinching_list = []
        
        for i, hand_lms in enumerate(last_landmarks):
            # Obtener identidad real de la mano ("Left" o "Right") en vez de usar 0 o 1 aleatorio
            if 'last_handedness' in locals() and len(last_handedness) > i:
                hand_id = last_handedness[i][0].category_name
            else:
                hand_id = str(i)
                
            pointer = detector.get_pointer_coordinates(hand_lms, w, h, hand_id)
            if pointer:
                pointers.append(pointer)
                is_pinching_list.append(detector.is_pinching(hand_lms))

        # 3. Lógica del Teclado Holográfico
        keyboard.process_interactions(pointers, is_pinching_list)

        # 4. Renderizado
        # A) Frame limpio para el overlay
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        # B) Dibujar el teclado en el overlay
        keyboard.draw(overlay)
        
        # C) Blend Holograma
        hologram_frame = apply_glow_overlay(frame, overlay, ALPHA_KEYBOARD)

        # D) Dibujar los punteros de los dedos
        for i, pointer in enumerate(pointers):
            px, py = pointer
            clicking = is_pinching_list[i] if i < len(is_pinching_list) else False
            
            # Color dinamico
            color = COLOR_POINTER_ACT if clicking else COLOR_POINTER
            radius = 15 if clicking else 10
            
            cv2.circle(hologram_frame, (px, py), radius, color, cv2.FILLED)
            # Efecto glow simulado en puntero
            cv2.circle(hologram_frame, (px, py), radius + 10, color, 1)

        # 5. UI de Diagnostico (FPS)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if curr_time > prev_time else 0
        prev_time = curr_time
        cv2.putText(hologram_frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

        # 6. Mostrar y Leer Input
        cv2.imshow("Hologram Keyboard", hologram_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Limpieza
    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
