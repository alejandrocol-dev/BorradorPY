import tkinter as tk
from tkinter import ttk, messagebox
import time
import cv2
import numpy as np
from borrame import Borrame


BACKGROUND_FILE = "bg.jpg"
CAPTURE_COUNTDOWN_SECONDS = 5


class BorrameApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Borrame")
        self.root.geometry("600x310")
        self.root.minsize(540, 280)
        self.root.configure(bg="#12131c")

        self.backgroundImage: np.ndarray | None = None
        self.statusVar = tk.StringVar(
            value="Paso 1: Despeja el área frente a la cámara y captura el fondo vacío."
        )

        # Variables de visualización
        self.drawBboxVar = tk.BooleanVar(value=True)
        self.drawHandVar = tk.BooleanVar(value=True)

        self._setupStyles()
        self._buildUI()

    def _setupStyles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Configuración de paleta oscura
        style.configure("Dark.TFrame", background="#12131c")
        style.configure("Card.TFrame", background="#1c1e2b", relief="flat")
        
        style.configure(
            "Title.TLabel", 
            background="#12131c", 
            foreground="#00f2fe", 
            font=("Segoe UI", 24, "bold")
        )
        style.configure(
            "Status.TLabel", 
            background="#12131c", 
            foreground="#4fd1c5", 
            font=("Segoe UI", 9, "italic")
        )

        # Eliminar bordes punteados de foco en Checkbuttons
        style.layout("Dark.TCheckbutton", [
            ("Checkbutton.padding", {
                "sticky": "nswe",
                "children": [
                    ("Checkbutton.indicator", {"side": "left", "sticky": ""}),
                    ("Checkbutton.label", {"side": "left", "sticky": ""})
                ]
            })
        ])
        style.configure(
            "Dark.TCheckbutton", 
            background="#1c1e2b", 
            foreground="#edf2f7", 
            font=("Segoe UI", 10),
            focuscolor="",
            focusthickness=0
        )
        style.map(
            "Dark.TCheckbutton",
            background=[("active", "#1c1e2b")],
            foreground=[("active", "#00f2fe")]
        )

        # Eliminar bordes punteados de foco en Botones
        style.layout("Primary.TButton", [
            ("Button.padding", {
                "sticky": "nswe",
                "children": [
                    ("Button.label", {"sticky": "nswe"})
                ]
            })
        ])
        style.configure(
            "Primary.TButton", 
            background="#4f46e5", 
            foreground="#ffffff", 
            font=("Segoe UI", 11, "bold"),
            borderwidth=0,
            focuscolor="",
            focusthickness=0
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#4338ca"), ("disabled", "#2d3748")],
            foreground=[("disabled", "#718096")]
        )

        style.layout("Secondary.TButton", [
            ("Button.padding", {
                "sticky": "nswe",
                "children": [
                    ("Button.label", {"sticky": "nswe"})
                ]
            })
        ])
        style.configure(
            "Secondary.TButton", 
            background="#0d9488", 
            foreground="#ffffff", 
            font=("Segoe UI", 11, "bold"),
            borderwidth=0,
            focuscolor="",
            focusthickness=0
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#0f766e"), ("disabled", "#2d3748")],
            foreground=[("disabled", "#718096")]
        )

    def _buildUI(self) -> None:
        container = ttk.Frame(self.root, padding=25, style="Dark.TFrame")
        container.pack(fill="both", expand=True)

        # Encabezado limpio
        headerFrame = ttk.Frame(container, style="Dark.TFrame")
        headerFrame.pack(fill="x", pady=(0, 15))

        title = ttk.Label(headerFrame, text="Borrame", style="Title.TLabel")
        title.pack(anchor="w")

        # Panel de acciones y opciones
        card = ttk.Frame(container, padding=18, style="Card.TFrame")
        card.pack(fill="both", expand=True, pady=(0, 15))

        # Botones principales
        btnRow = ttk.Frame(card, style="Card.TFrame")
        btnRow.pack(fill="x", pady=(0, 15))

        self.captureBtn = ttk.Button(
            btnRow,
            text="📷  1. Capturar Fondo",
            style="Secondary.TButton",
            command=self.captureBackground,
            takefocus=False
        )
        self.captureBtn.pack(side="left", ipady=6, ipadx=12)

        self.startBtn = ttk.Button(
            btnRow,
            text="▶  2. Iniciar Borrame",
            style="Primary.TButton",
            command=self.startApp,
            state="disabled",
            takefocus=False
        )
        self.startBtn.pack(side="left", padx=(12, 0), ipady=6, ipadx=12)

        ttk.Separator(card, orient="horizontal").pack(fill="x", pady=(0, 14))

        # Opciones con checkboxes limpios sin líneas punteadas
        optsRow = ttk.Frame(card, style="Card.TFrame")
        optsRow.pack(fill="x")

        ttk.Checkbutton(
            optsRow,
            text="Dibujar contorno verde de persona",
            variable=self.drawBboxVar,
            style="Dark.TCheckbutton",
            takefocus=False
        ).pack(side="left", padx=(0, 25))

        ttk.Checkbutton(
            optsRow,
            text="Dibujar esqueleto de mano y mediciones IA",
            variable=self.drawHandVar,
            style="Dark.TCheckbutton",
            takefocus=False
        ).pack(side="left")

        # Barra de estado
        statusLabel = ttk.Label(container, textvariable=self.statusVar, style="Status.TLabel", wraplength=580)
        statusLabel.pack(anchor="w")

    def captureBackground(self) -> None:
        self.statusVar.set(
            f"Abriendo cámara. Sal del plano antes de que finalice la cuenta atrás de {CAPTURE_COUNTDOWN_SECONDS}s..."
        )
        self.root.update_idletasks()
        winName = "Borrame"

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error de Cámara", "No se pudo acceder a la webcam.")
            self.statusVar.set("Error: Cámara no disponible.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, 1024, 576)

        frame = None
        startTime = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                elapsed = time.time() - startTime
                remaining = max(0, int(CAPTURE_COUNTDOWN_SECONDS - elapsed + 0.999))

                preview = frame.copy()

                # Cartel visual de cuenta regresiva
                cv2.rectangle(preview, (25, 25), (520, 160), (20, 20, 30), -1)
                cv2.rectangle(preview, (25, 25), (520, 160), (0, 242, 254), 2)

                cv2.putText(preview, "CAPTURANDO FONDO VACIO", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 242, 254), 2, cv2.LINE_AA)
                cv2.putText(preview, "Por favor, sal del encuadre...", (40, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(preview, f"Foto en: {remaining} segundos", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 120), 2, cv2.LINE_AA)

                cv2.imshow(winName, preview)
                self.statusVar.set(f"Capturando fondo en {remaining}s. Mantén el plano despejado.")
                self.root.update_idletasks()

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    self.statusVar.set("Captura cancelada por el usuario.")
                    return
                if cv2.getWindowProperty(winName, cv2.WND_PROP_VISIBLE) < 1:
                    self.statusVar.set("Captura cancelada.")
                    return

                if elapsed >= CAPTURE_COUNTDOWN_SECONDS:
                    break

            if frame is None:
                messagebox.showerror("Error", "No se pudo obtener imagen de la cámara.")
                return
        finally:
            cap.release()
            cv2.destroyWindow(winName)

        self.backgroundImage = frame.copy()
        cv2.imwrite(BACKGROUND_FILE, self.backgroundImage)
        self.statusVar.set("✓ Fondo capturado exitosamente. ¡Listo para iniciar!")
        self.startBtn.config(state="normal")

    def startApp(self) -> None:
        if self.backgroundImage is None:
            messagebox.showinfo("Fondo requerido", "Primero captura una imagen de fondo.")
            return

        self.root.withdraw()
        try:
            self.runMainLoop()
        finally:
            self.root.deiconify()

    def runMainLoop(self) -> None:
        appWin = "Borrame"
        engine = Borrame(
            bgImage=self.backgroundImage,
            fps=30
        )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error de Cámara", "No se pudo abrir la cámara.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        cv2.namedWindow(appWin, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(appWin, 1024, 576)

        prevTime = time.time()
        fpsDisplay = 30

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Procesar fotograma con el motor Bórrame (reemplazo limpio y directo)
                outputFrame = engine.processFrame(
                    frame,
                    drawBbox=self.drawBboxVar.get(),
                    drawHand=self.drawHandVar.get()
                )

                # Calcular FPS
                now = time.time()
                fpsDisplay = int(1.0 / max(0.001, now - prevTime))
                prevTime = now

                # Barra superior informativa limpia
                hud_bg = outputFrame[:42, :].astype(np.float32)
                outputFrame[:42, :] = np.clip(hud_bg * 0.35 + np.array([20, 20, 30]) * 0.65, 0, 255).astype(np.uint8)

                bbox_str = f"Cuadro Verde [B]: {'ON' if self.drawBboxVar.get() else 'OFF'}"
                hand_str = f"Esqueleto Mano [H]: {'ON' if self.drawHandVar.get() else 'OFF'}"
                hud_text = f"BORRAME  |  {bbox_str}  |  {hand_str}  |  FPS: {fpsDisplay}  |  Recapturar [C]  |  Salir [Q]"
                cv2.putText(outputFrame, hud_text, (15, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 242, 254), 1, cv2.LINE_AA)

                cv2.imshow(appWin, outputFrame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")): # Salir
                    break
                elif key == ord("b"): # Alternar contorno verde
                    self.drawBboxVar.set(not self.drawBboxVar.get())
                elif key == ord("h"): # Alternar esqueleto de mano
                    self.drawHandVar.set(not self.drawHandVar.get())
                elif key == ord("c"): # Recapturar fondo en vivo
                    engine.loadBackgroundImage(frame)
                    cv2.imwrite(BACKGROUND_FILE, frame)

                if cv2.getWindowProperty(appWin, cv2.WND_PROP_VISIBLE) < 1:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    BorrameApp().run()