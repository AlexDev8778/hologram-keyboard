# src/keyboard.py
import cv2
import time
from pynput.keyboard import Controller, Key
import copy
from src.config import (
    KEYBOARD_LAYOUT, KEY_SIZE, KEY_SPACING, START_X, START_Y,
    COLOR_BG_NORMAL, COLOR_BG_HOVER, COLOR_BG_PRESS,
    COLOR_TEXT, COOLDOWN_MS
)

keyboard_controller = Controller()

class KeyButton:
    def __init__(self, text, x, y, width, height):
        self.text = text
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.state = "NORMAL"  # NORMAL, HOVER, PRESS
        self.last_press_time = 0

    def is_hovered(self, px, py):
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)

    def draw(self, img):
        if self.state == "PRESS":
            bg_color = COLOR_BG_PRESS
            border_color = (255, 255, 255)
            font_color = (0, 0, 0) # Texto negro sobre fondo verde iluminado
            thickness = 3
        elif self.state == "HOVER":
            bg_color = COLOR_BG_HOVER
            border_color = (255, 255, 0) # Cyan BGR
            font_color = COLOR_TEXT
            thickness = 2
        else:
            bg_color = COLOR_BG_NORMAL
            border_color = (150, 150, 150)
            font_color = COLOR_TEXT
            thickness = 1

        # Dibujar rectangulo de fondo
        cv2.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), bg_color, cv2.FILLED)
        
        # Efecto Sci-Fi: Esquinas reforzadas (HUD Style)
        L = 12 # Largo de las esquinas
        # Arriba-Izq
        cv2.line(img, (self.x, self.y), (self.x + L, self.y), border_color, thickness)
        cv2.line(img, (self.x, self.y), (self.x, self.y + L), border_color, thickness)
        # Arriba-Der
        cv2.line(img, (self.x + self.width, self.y), (self.x + self.width - L, self.y), border_color, thickness)
        cv2.line(img, (self.x + self.width, self.y), (self.x + self.width, self.y + L), border_color, thickness)
        # Abajo-Izq
        cv2.line(img, (self.x, self.y + self.height), (self.x + L, self.y + self.height), border_color, thickness)
        cv2.line(img, (self.x, self.y + self.height), (self.x, self.y + self.height - L), border_color, thickness)
        # Abajo-Der
        cv2.line(img, (self.x + self.width, self.y + self.height), (self.x + self.width - L, self.y + self.height), border_color, thickness)
        cv2.line(img, (self.x + self.width, self.y + self.height), (self.x + self.width, self.y + self.height - L), border_color, thickness)

        # Borde ultra sutil intermedio si no esta presionado
        if self.state != "PRESS":
            cv2.rectangle(img, (self.x, self.y), (self.x + self.width, self.y + self.height), (80, 80, 80), 1)

        # Texto centrado
        font = cv2.FONT_HERSHEY_PLAIN
        font_scale = 1.8 if len(self.text) < 4 else 1.2
        text_size = cv2.getTextSize(self.text, font, font_scale, thickness if self.state != "NORMAL" else 1)[0]
        text_x = self.x + (self.width - text_size[0]) // 2
        text_y = self.y + (self.height + text_size[1]) // 2

        cv2.putText(img, self.text, (text_x, text_y), font, font_scale, font_color, thickness if self.state != "NORMAL" else 1)


class VirtualKeyboard:
    def __init__(self):
        self.keys = []
        self._build_layout()

    def _build_layout(self):
        curr_y = START_Y
        for row in KEYBOARD_LAYOUT:
            curr_x = START_X
            for char in row:
                if char == "SPACE":
                    w = KEY_SIZE * 5 + KEY_SPACING * 4
                    curr_x = START_X + KEY_SIZE * 3
                elif char == "BKSP":
                    w = KEY_SIZE * 2 + KEY_SPACING
                else:
                    w = KEY_SIZE
                
                self.keys.append(KeyButton(char, curr_x, curr_y, w, KEY_SIZE))
                curr_x += w + KEY_SPACING
            curr_y += KEY_SIZE + KEY_SPACING

    def process_interactions(self, pointers, is_pinching_list):
        """
        pointers: lista de tuplas (x, y) de los dedos indices de cada mano conectada.
        is_pinching_list: lista de booleanos indicando si la mano en cuestion esta haciendo click.
        """
        # Reseteamos estado a Normal
        for key in self.keys:
            if key.state != "PRESS": # Los PRESS los manejamos por tiempo
                key.state = "NORMAL"

        current_time = time.time() * 1000

        # Mantenemos las teclas presionadas el tiempo suficiente para que se note en pantalla
        for key in self.keys:
            if key.state == "PRESS" and (current_time - key.last_press_time) > COOLDOWN_MS:
                key.state = "NORMAL"

        if not pointers:
            return

        # Para cada mano detectada
        for i, pointer in enumerate(pointers):
            px, py = pointer
            clicking = is_pinching_list[i] if i < len(is_pinching_list) else False

            for key in self.keys:
                if key.is_hovered(px, py):
                    if clicking:
                        # Revisa el cooldown para no spamear
                        if current_time - key.last_press_time > COOLDOWN_MS:
                            key.state = "PRESS"
                            key.last_press_time = current_time
                            self._trigger_keypress(key.text)
                    else:
                        if key.state != "PRESS":
                            key.state = "HOVER"

    def _trigger_keypress(self, text):
        """Usa pynput para simular la tecla a nivel del Sistema Operativo."""
        try:
            if text == "SPACE":
                keyboard_controller.press(Key.space)
                keyboard_controller.release(Key.space)
            elif text == "BKSP":
                keyboard_controller.press(Key.backspace)
                keyboard_controller.release(Key.backspace)
            else:
                keyboard_controller.press(text.lower())
                keyboard_controller.release(text.lower())
            print(f"Key Typed: {text}")
        except Exception as e:
            print(f"Error typing key {text}: {e}")

    def draw(self, frame):
        for key in self.keys:
            key.draw(frame)
