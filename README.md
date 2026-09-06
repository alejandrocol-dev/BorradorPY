# Borrame

Sistema de visión por computadora en tiempo real desarrollado en Python con OpenCV y MediaPipe. La aplicación permite generar un efecto de invisibilidad interactiva al detectar el cierre del puño del usuario, reemplazando la región corporal por un fondo estático previamente capturado, a la vez que proporciona seguimiento articular y métricas espaciales de la mano.

---

## Características Principales

- **Segmentación y Reemplazo de Fondo**: Sustitución dinámica del área delimitada del usuario en tiempo real con latencia mínima.
- **Rastreo Articular de Mano (MediaPipe Hand Landmarker)**:
  - Detección de los 21 puntos clave anatómicos de la mano.
  - Cálculo continuo de la distancia euclidiana entre las puntas de los dedos y la articulación de la muñeca.
  - Clasificación de estados gestuales (mano abierta vs. puño cerrado).
  - Filtro temporal de histéresis (*debouncer*) para estabilizar la transición de estados y mitigar falsos positivos por oclusión.
- **Detección de Presencia Humana (Bounding Box)**: Localización continua de la persona dentro del encuadre mediante modelos basados en TensorFlow Lite.
- **Interfaz de Usuario (Tkinter)**: Panel de control con diseño oscuro integrado, asistente guiado para la captura del fondo y selectores de superposiciones visuales.
- **Panel de Información en Vivo (HUD)**: Superposición en el flujo de video con indicadores de estado, tasa de cuadros por segundo (FPS) y controles rápidos.

---

## Estructura del Repositorio

```text
├── main.py                     # Interfaz gráfica de usuario y ciclo principal
├── borrame.py                  # Módulo de procesamiento y lógica de invisibilidad
├── utils.py                    # Clasificadores de mano, detectores y utilidades auxiliares
├── requirements.txt            # Especificación de dependencias del proyecto
├── efficientdet_lite0.tflite   # Modelo de detección de objetos/personas
├── hand_landmarker.task        # Modelo de puntos clave de mano de MediaPipe
├── selfie_segmenter.tflite     # Modelo de segmentación de silueta
├── .gitignore                  # Patrones de exclusión para Git
└── README.md                   # Documentación técnica del proyecto
```

---

## Requisitos del Sistema

- **Python**: Versión 3.9 o superior.
- **Cámara Web**: Resolución mínima recomendada de 720p (1280x720).
- **Sistema Operativo**: Compatible con Windows, Linux y macOS.

---

## Instalación y Configuración

1. **Clonar el repositorio**:
   ```bash
   git clone https://github.com/alejandrocol-dev/BorradorPY.git
   cd BorradorPY
   ```

2. **Crear y activar un entorno virtual (recomendado)**:
   - En Windows:
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\activate
     ```
   - En Linux/macOS:
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Ejecución

```bash
python main.py
```

---

## Flujo de Operación

1. **Paso 1 - Captura de Fondo**:
   - Presionar el botón `1. Capturar Fondo`.
   - Despejar el encuadre frente a la cámara durante la cuenta regresiva de 5 segundos.
2. **Paso 2 - Inicio del Sistema**:
   - Presionar el botón `2. Iniciar Borrame`.
   - Ubicarse frente a la cámara dentro del campo visual.
3. **Control Gestual**:
   - **Mano abierta**: Visualización normal con superposición del esqueleto articular y vectores de medición.
   - **Puño cerrado**: Activación del efecto de invisibilidad sobre el usuario.

---

## Controles en Tiempo Real (Modo Cámara)

| Tecla | Función |
|:---:|---|
| `B` | Alternar la visibilidad del recuadro delimitador (Bounding Box). |
| `H` | Alternar la visibilidad del esqueleto articular de la mano. |
| `C` | Recapturar el fotograma de fondo en caliente. |
| `Q` / `Esc` | Finalizar el modo de captura y regresar a la interfaz principal. |

---

## Licencia

Este proyecto está distribuido bajo los términos de la Licencia MIT. Para mayor información, consulte el archivo [LICENSE](LICENSE).
