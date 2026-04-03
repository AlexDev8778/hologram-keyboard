# HoloGesture Keyboard 

Un teclado flotante holográfico controlado por inteligencia artificial y visión artificial. 
Convierte cualquier cámara web (o tu teléfono) en una interfaz interactiva donde tus manos actúan como puntero para escribir directamente sobre el Sistema Operativo, sin necesidad de hardware adicional.

## Características

- **Zero-Hardware tracking**: Infiere posiciones 3D de tus nudillos usando solamente una cámara 2D estándar.
- **Inyección Nivel de SO**: No es sólo de muestra; simula presiones de teclas físicas reales gracias a `pynput`.
- **Efecto HUD Sci-Fi**: Diseño vectorial basado en Alpha Blending con *scanlines* e interfaces futuristas sin quemar recursos.
- **Micro-Arquitectura Threads**: Rendimiento masivamente optimizado para hardware de escritorio limitado (testado en CPU Intel i3-2100).
- **Smooth Cursor Dynamics**: Implementación matemática de filtro EMA (Exponential Moving Average) y "Anti-Teletransporte" multi-mano.

## Stack Tecnológico

- **Python 3.12**: Lenguaje principal de orquestación.
- **OpenCV**: Renderizado por hardware, alpha blending y captura de streams de video (MSMF / DroidCam).
- **MediaPipe (Tasks API v0.10+)**: Machine Learning avanzado para identificación predictiva de manos.
- **Pynput**: Comunicación a bajo nivel con el Kernel de teclado del sistema operativo anfitrión.

---

## Instalación y Setup (Windows)

### 1. Clonar el repositorio
```bash
git clone https://github.com/AlexDev8778/hologram-keyboard.git
cd hologram-keyboard
```

### 2. Configurar el Entorno Virtual (Importante)
Asegurate de usar **Python 3.12** (versiones más nuevas pueden tener conflictos con los bindings C de MediaPipe).
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Instalar las Dependencias
```bash
pip install opencv-python mediapipe numpy pynput
```

### 4. Modelo de MediaPipe
El proyecto requiere el modelo pre-entrenado oficial de IA (`hand_landmarker.task`).
Si no lo tenés, podés ejecutar el script de testeo para que lo descargue por vos automáticamente:
```bash
python test_mediapipe.py
```

---

## Cómo usar

Asegurate de tener una cámara conectada por USB (o la cámara virtual de DroidCam corriendo sin bloqueos en segundo plano).

Para arrancar el motor holográfico, simplemente ejecutá:
```bash
python main.py
```

### Uso y atajos:
- Mostrá una mano frente a la cámara. Aparecerá un puntero sobre la misma.
- Para hacer "Click" sobre una tecla, juntá el dedo **índice** con el **pulgar** (*gesto de Pinch*).
- Salí del programa tocando la tecla `Q` en la ventana o directamente `ESC`.

## Arquitectura del Proyecto

Siguiendo principios de diseño modular:
- `main.py`: Bucle orquestador. Llama al framerate, al IA rate-limit y ensambla las capas gráficas.
- `src/config.py`: Diccionario global. Cambiá colores hex, layout de teclado, márgenes y tolerancias de rastreo desde acá.
- `src/camera.py`: Worker de captura asíncrona. Pone en cuarentena los bloqueos del driver USB lejos de tu procesador central.
- `src/detector.py`: Wrapper de IA. Configura a MediaPipe para que decodifique si es una mano Izquierda o Derecha y extraiga los puntos medios de presión (Anti-Jitter).
- `src/keyboard.py`: El lienzo. Modela clases matriciales manejando su estado propio (Hover, Press, Sleep) para la interactividad visual.

## Known Issues y Trucos
- **Pantalla verde vacía**: Si usás *DroidCam* y tu cámara se bloquea en una pantalla plana de color verde, significa que tu aplicación de Windows atrapó el driver antes que la aplicación de tu teléfono enviara el video. Reiniciá el programa del teléfono, y luego volvé a iniciar el programa de Python.
- **Falsos positivos (rastrea la cara)**: Modificá `MIN_CONFIDENCE` en el `config.py`. En `0.85`, solo aceptará manos posicionadas robóticamente perfectas. En `0.70`, proporciona fluidez.

---
*Desarrollado y estructurado durante una sesión de pair-programming de arquitectura de software para llevar interfaces al próximo nivel.*
