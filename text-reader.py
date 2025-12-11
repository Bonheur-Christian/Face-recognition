"""
GUI-based OCR tool with:
- Image load and camera capture
- ROI (click + drag) selection
- Run OCR button with overlay preview of detected text
- Live camera input with start/stop controls
- Text display of extracted content
"""

import cv2
import pytesseract
from pytesseract import Output
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from PIL import Image, ImageTk


class OCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Text Reader - OCR with ROI and Camera")

        # Canvas/display settings
        self.canvas_width = 900
        self.canvas_height = 600

        # State
        self.current_image = None  # OpenCV BGR image currently displayed/used
        self.video_capture = None
        self.running_camera = False
        self.pause_live_preview = False
        self.display_image_meta = None  # For coordinate transforms
        self.roi_start = None
        self.roi_box = None  # (x1, y1, x2, y2) in image coords
        self.photo_image = None  # Keep reference to avoid GC
        self.colors = {
            "bg": "#0f172a",
            "card": "#111827",
            "canvas_bg": "#0b1220",
            "accent": "#6366f1",
            "accent_active": "#818cf8",
            "fg": "#e5e7eb",
            "muted": "#9ca3af",
            "border": "#1f2937",
        }

        self._build_styles()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_styles(self) -> None:
        self.root.configure(bg=self.colors["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=self.colors["bg"], foreground=self.colors["fg"], font=("Segoe UI", 10))
        style.configure(
            "Card.TFrame",
            background=self.colors["card"],
            relief="flat",
        )
        style.configure(
            "Accent.TButton",
            font=("Segoe UI Semibold", 10),
            padding=(10, 6),
            background=self.colors["accent"],
            foreground="white",
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.colors["accent_active"])],
            relief=[("pressed", "sunken")],
        )
        style.configure(
            "Title.TLabel",
            font=("Segoe UI Semibold", 16),
            background=self.colors["card"],
            foreground=self.colors["fg"],
        )
        style.configure(
            "Subtitle.TLabel",
            font=("Segoe UI", 10),
            background=self.colors["card"],
            foreground=self.colors["muted"],
        )
        style.configure(
            "Status.TLabel",
            font=("Segoe UI", 10),
            padding=(10, 4),
            background=self.colors["card"],
            foreground=self.colors["accent"],
        )

    def _build_ui(self) -> None:
        self.status_var = tk.StringVar(value="Load an image or start the camera.")

        hero = ttk.Frame(self.root, style="Card.TFrame", padding=(12, 10))
        hero.pack(fill=tk.X, padx=10, pady=(8, 6))
        ttk.Label(hero, text="Text Reader", style="Title.TLabel").pack(anchor="w")
        ttk.Label(hero, text="OCR with ROI selection and live camera preview", style="Subtitle.TLabel").pack(
            anchor="w", pady=(2, 0)
        )
        ttk.Label(hero, textvariable=self.status_var, style="Status.TLabel").pack(anchor="e", pady=(6, 0))

        control_frame = ttk.Frame(self.root, style="Card.TFrame", padding=(10, 8))
        control_frame.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(control_frame, text="Load Image", style="Accent.TButton", command=self.load_image).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(
            control_frame, text="Start/Resume Camera", style="Accent.TButton", command=self.start_camera
        ).pack(side=tk.LEFT, padx=4)
        ttk.Button(control_frame, text="Stop Camera", style="Accent.TButton", command=self.stop_camera).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(control_frame, text="Run OCR", style="Accent.TButton", command=self.run_ocr).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(control_frame, text="Clear ROI", style="Accent.TButton", command=self.clear_roi).pack(
            side=tk.LEFT, padx=4
        )

        content = ttk.Frame(self.root, style="Card.TFrame", padding=10)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        canvas_card = ttk.Frame(content, style="Card.TFrame", padding=8)
        canvas_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(canvas_card, text="Live View + ROI Overlay", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 6))
        # Canvas for image/overlay + ROI drawing
        self.canvas = tk.Canvas(
            canvas_card,
            width=self.canvas_width,
            height=self.canvas_height,
            bg=self.colors["canvas_bg"],
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            cursor="crosshair",
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)
        ttk.Label(
            canvas_card,
            text="Tip: drag to mark ROI, then run OCR. Clear ROI to scan the whole frame.",
            style="Subtitle.TLabel",
            wraplength=780,
        ).pack(anchor="w", pady=(8, 0))

        text_card = ttk.Frame(content, style="Card.TFrame", padding=8)
        text_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        ttk.Label(text_card, text="Extracted Text", style="Subtitle.TLabel").pack(anchor="w", pady=(0, 6))
        self.text_output = scrolledtext.ScrolledText(text_card, height=24, wrap=tk.WORD)
        self.text_output.configure(
            background=self.colors["canvas_bg"],
            foreground=self.colors["fg"],
            insertbackground=self.colors["accent"],
            relief=tk.FLAT,
            borderwidth=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
        )
        self.text_output.pack(fill=tk.BOTH, expand=True)

    # ----------------- Utility functions -----------------
    def _cv_to_tk_image(self, cv_img: np.ndarray):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_h, img_w = pil_img.height, pil_img.width

        scale = min(self.canvas_width / img_w, self.canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        offset_x = (self.canvas_width - new_w) // 2
        offset_y = (self.canvas_height - new_h) // 2

        self.display_image_meta = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "img_w": img_w,
            "img_h": img_h,
        }
        self.photo_image = ImageTk.PhotoImage(resized)
        return self.photo_image, offset_x, offset_y

    def _canvas_to_image_coords(self, x: float, y: float):
        if not self.display_image_meta:
            return None
        m = self.display_image_meta
        img_x = int((x - m["offset_x"]) / m["scale"])
        img_y = int((y - m["offset_y"]) / m["scale"])
        if 0 <= img_x < m["img_w"] and 0 <= img_y < m["img_h"]:
            return img_x, img_y
        return None

    def _image_to_canvas_coords(self, x: float, y: float):
        m = self.display_image_meta
        canvas_x = x * m["scale"] + m["offset_x"]
        canvas_y = y * m["scale"] + m["offset_y"]
        return canvas_x, canvas_y

    def display_image(self, cv_img: np.ndarray) -> None:
        if cv_img is None:
            return
        tk_img, offset_x, offset_y = self._cv_to_tk_image(cv_img)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=tk_img, anchor=tk.NW, tags="img")
        self.draw_roi_box()

    def draw_roi_box(self) -> None:
        self.canvas.delete("roi")
        if not self.roi_box or not self.display_image_meta:
            return
        x1, y1, x2, y2 = self.roi_box
        c1 = self._image_to_canvas_coords(x1, y1)
        c2 = self._image_to_canvas_coords(x2, y2)
        self.canvas.create_rectangle(
            *c1, *c2, outline=self.colors["accent"], width=2, tags="roi"
        )

    # ----------------- Event handlers -----------------
    def on_mouse_press(self, event) -> None:
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            self.roi_start = coords
            self.roi_box = None
            self.draw_roi_box()

    def on_mouse_drag(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.draw_roi_box()

    def on_mouse_release(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.roi_start = None
        self.draw_roi_box()

    # ----------------- Camera handling -----------------
    def start_camera(self) -> None:
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                self.video_capture.release()
                self.video_capture = None
                messagebox.showerror("Camera Error", "Cannot open camera.")
                return
        self.running_camera = True
        self.pause_live_preview = False
        self.status_var.set("Camera running. Drag to set ROI or click Run OCR.")
        self.update_camera_frame()

    def stop_camera(self) -> None:
        self.running_camera = False
        self.pause_live_preview = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.status_var.set("Camera stopped.")

    def update_camera_frame(self) -> None:
        if not self.running_camera or self.pause_live_preview:
            return
        if self.video_capture is None:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.status_var.set("Failed to read from camera.")
            return
        self.current_image = frame
        self.display_image(frame)
        # Schedule next frame
        self.root.after(30, self.update_camera_frame)

    # ----------------- OCR handling -----------------
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing to boost OCR quality."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduce noise and improve contrast
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        # Adaptive threshold to handle uneven lighting
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        return binary

    def _prepare_roi(self):
        if self.current_image is None:
            return None, None
        h, w = self.current_image.shape[:2]
        if not self.roi_box:
            return self.current_image.copy(), (0, 0)
        x1, y1, x2, y2 = self.roi_box
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return self.current_image.copy(), (0, 0)
        roi_img = self.current_image[y1:y2, x1:x2].copy()
        return roi_img, (x1, y1)

    def run_ocr(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return

        # Freeze live preview to keep overlay visible
        if self.running_camera:
            self.pause_live_preview = True

        roi_img, (offset_x, offset_y) = self._prepare_roi()
        if roi_img is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return

        # Preprocess to improve OCR quality
        processed = self._preprocess_for_ocr(roi_img)

        # Run OCR
        ocr_config = "--psm 6"
        ocr_text = pytesseract.image_to_string(processed, config=ocr_config)
        data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=ocr_config)

        annotated = self.current_image.copy()
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
            if not text or conf < 10:
                continue
            x = data["left"][i] + offset_x
            y = data["top"][i] + offset_y
            w = data["width"][i]
            h = data["height"][i]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                text,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        self.text_output.delete("1.0", tk.END)
        stripped = ocr_text.strip()
        if stripped:
            self.text_output.insert(tk.END, stripped)
            self.status_var.set("OCR complete. Overlay shows detected text. Resume camera if needed.")
        else:
            self.text_output.insert(tk.END, "[No text detected]")
            self.status_var.set("No text detected. Try a larger ROI, better lighting, or move closer.")
        self.display_image(annotated)

    # ----------------- Misc helpers -----------------
    def clear_roi(self) -> None:
        self.roi_box = None
        self.draw_roi_box()
        self.status_var.set("ROI cleared.")

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Load Error", "Could not load the selected image.")
            return
        self.current_image = img
        self.pause_live_preview = False
        self.running_camera = False
        self.display_image(img)
        self.status_var.set("Image loaded. Select ROI or run OCR.")

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
