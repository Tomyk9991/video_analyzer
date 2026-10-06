"""Trigger 2 (Demo): Lampen-ROI per Pixel-Mittelwert.

Idee aus deiner Spec: Pixelbereich wählen, Mittelwert der Zielfarbe,
Schwelle -> AN/AUS. Kostet ~0.05ms, also quasi gratis vs. NN.
"""
import os
import cv2
import numpy as np


def parse_roi(s: str, w: int, h: int):
    """Format 'x,y,breite,hoehe' in Prozent (0-100) oder Pixel. Default: rechts oben 10x10%."""
    if not s:
        return int(w * 0.8), int(h * 0.05), int(w * 0.15), int(h * 0.15)
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI_FORMAT muss 'x,y,w,h' sein")
    # Heuristik: Werte <=100 -> Prozent
    if all(p <= 100 for p in parts):
        x = int(parts[0] / 100 * w); y = int(parts[1] / 100 * h)
        rw = int(parts[2] / 100 * w); rh = int(parts[3] / 100 * h)
        return x, y, rw, rh
    return int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])


class LampRoi:
    def __init__(self):
        # Ziel-Helligkeit: V-Kanal (HSV) Mittelwert; Schwelle -> AN
        self.thres = float(os.getenv("LAMP_THRES", "150"))
        self.roi_str = os.getenv("ROI", "")

    def check(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        x, y, rw, rh = parse_roi(self.roi_str, w, h)
        x, y = max(0, x), max(0, y)
        roi = frame_bgr[y:y + rh, x:x + rw]
        if roi.size == 0:
            return False, 0.0, (x, y, rw, rh)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_mean = float(hsv[:, :, 2].mean())
        is_on = v_mean >= self.thres
        return is_on, v_mean, (x, y, rw, rh)
