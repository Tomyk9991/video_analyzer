"""Person-Erkennung per MobileNet-SSD + OpenCV-DNN (CPU).

Bewusst KEIN YOLO/torch/onnxruntime im Default: für den PoC zählt ein
repräsentativer, aber übertragbarer Aufbau (kleines Docker-Image, ~400MB,
kein 2GB-torch). MobileNet-SSD erkennt "person" (VOC-Klasse 15) zuverlässig
genug für die Ja/Nein-Frage + Watt-Messung. Später tauschbar gegen YOLOv8n.
"""
import os
import urllib.request
import cv2
import numpy as np

PROTO_URL = os.getenv(
    "SSD_PROTO_URL",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
)
MODEL_URL = os.getenv(
    "SSD_MODEL_URL",
    "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/mobilenet_iter_73000.caffemodel",
)
PROTO_PATH = os.getenv("SSD_PROTO_PATH", "models/deploy.prototxt")
MODEL_PATH = os.getenv("SSD_MODEL_PATH", "models/mobilenet_iter_73000.caffemodel")
CONF_THRES = float(os.getenv("CONF_THRES", "0.5"))

PERSON_CLASS = 15  # VOC: background=0, ..., person=15


def _dl(url: str, path: str, min_bytes: int):
    if os.path.exists(path) and os.path.getsize(path) > min_bytes:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    print(f"[detector] lade {url} -> {path} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(path, "wb") as f:
        f.write(r.read())
    print("[detector] OK")
    return path


def ensure_model():
    _dl(PROTO_URL, PROTO_PATH, 10_000)
    _dl(MODEL_URL, MODEL_PATH, 1_000_000)
    return PROTO_PATH, MODEL_PATH


class PersonDetector:
    def __init__(self, proto: str = PROTO_PATH, model: str = MODEL_PATH):
        proto, model = ensure_model() if not (
            os.path.exists(proto) and os.path.exists(model)) else (proto, model)
        # Falls Pfade per ENV gesetzt aber noch nicht geladen:
        ensure_model()
        self.net = cv2.dnn.readNetFromCaffe(PROTO_PATH, MODEL_PATH)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print("[detector] MobileNet-SSD geladen (CPU, OpenCV-DNN)")

    def infer(self, frame_bgr):
        """Returns (person_found: bool, best_conf: float, boxes: list)."""
        h, w = frame_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(frame_bgr, 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        det = self.net.forward()  # (1,1,N,7): [img, class, conf, x1,y1,x2,y2]
        boxes = []
        best = 0.0
        for i in range(det.shape[2]):
            cls = int(det[0, 0, i, 1])
            conf = float(det[0, 0, i, 2])
            if cls == PERSON_CLASS and conf >= CONF_THRES:
                x1 = float(det[0, 0, i, 3]) * w
                y1 = float(det[0, 0, i, 4]) * h
                x2 = float(det[0, 0, i, 5]) * w
                y2 = float(det[0, 0, i, 6]) * h
                boxes.append(([x1, y1, x2, y2], conf))
                best = max(best, conf)
        return (len(boxes) > 0), best, boxes
