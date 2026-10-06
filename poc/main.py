"""PoC-Einstieg: zwei Modi – streamen (privater PC mit Webcam) oder
analysieren (HA-Maschine ohne Kamera, liest den Stream übers Netz).

 Modi:
   python -m poc.main --mode stream --source 0            # HIER: Webcam -> MJPEG :8000/stream.mjpg
   python -m poc.main --mode analyze --source http://PC-IP:8000/stream.mjpg  # DORT: Analyse + Watt-Log
   python -m poc.main --mode analyze --source video.mp4  # Datei / RTSP-URL geht auch
   python -m poc.main --bench 200                         # ohne Kamera: 200 Inferenzen auf Testbild

 ENV (siehe .env.example):
   VIDEO_SOURCE, CONF_THRES, DETECT_EVERY_N, ROI, LAMP_THRES, HTTP_PORT, CSV_LOG,
   STREAM_WIDTH, STREAM_FPS, STREAM_QUALITY
   (Home Assistant wird nur vom Plot-Tool gelesen, nie beschrieben –
   siehe tools/plot_watts.py mit HA_URL/HA_TOKEN/HA_WATT_ENTITY)
"""
import argparse
import csv
import os
import sys

try:
    # .env auch bei nativem Start (ohne Docker) berücksichtigen
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import threading
import time
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler

# FFmpeg-Timeout für Netz-Streams kurz halten (Default 30s blockiert jeden Retry)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;5000000")

import cv2
import numpy as np

from poc.detector import PersonDetector
from poc.power import PowerMonitor
from poc.roi_lamp import LampRoi

TEST_IMG_URL = "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg"
TEST_IMG_PATH = "models/test_bus.jpg"

latest_jpeg = None
state = {}


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/metrics.json":
            import json
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path in ("/", "/index.html"):
            html = """<html><body style="background:#111;color:#eee;font-family:sans-serif">
            <h2>Video Feed Analyzer PoC</h2>
            <img src="/stream.mjpg" style="max-width:95vw;border:2px solid #444"/>
            <p><a href="/metrics.json" style="color:#8cf">metrics.json</a> (für Home Assistant)</p>
            </body></html>""".encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()
            while True:
                try:
                    if latest_jpeg is None:
                        time.sleep(0.1)
                        continue
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(latest_jpeg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.08)
                except Exception:
                    break
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


def serve(port):
    # Threading-Server ist Pflicht: ein einzelner /stream.mjpg-Viewer (z.B. Browser)
    # darf /healthz, /metrics.json und weitere Viewer nicht blockieren.
    # (Single-threaded hat genau das getan: TCP-Connect ok, aber 0 Bytes Antwort.)
    from http.server import ThreadingHTTPServer
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    srv.daemon_threads = True
    print(f"[http] Preview: http://localhost:{port}  Metriken: http://localhost:{port}/metrics.json")
    srv.serve_forever()


def ensure_test_img():
    if os.path.exists(TEST_IMG_PATH) and os.path.getsize(TEST_IMG_PATH) > 1000:
        return TEST_IMG_PATH
    os.makedirs("models", exist_ok=True)
    print(f"[bench] lade Testbild {TEST_IMG_URL} ...")
    req = urllib.request.Request(TEST_IMG_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r, open(TEST_IMG_PATH, "wb") as f:
        f.write(r.read())
    return TEST_IMG_PATH


def open_source(src):
    # Zahl -> Webcam-Index, sonst Pfad/URL. Windows: mehrere Backends probieren,
    # weil DSHOW allein oft versagt (MSMF ist auf Win10/11 meist der funktionierende).
    try:
        idx = int(src)
    except ValueError:
        cap = cv2.VideoCapture(src)
        # Kein Puffer: immer das neueste Frame (wichtig bei niedrigen fps,
        # sonst verrechnet der Analyzer alte Frames mit wachsendem Verzug)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    if sys.platform == "win32":
        for backend in (cv2.CAP_ANY, cv2.CAP_MSMF, cv2.CAP_DSHOW):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                return cap
            cap.release()
        return cv2.VideoCapture(idx)  # letzter Versuch (isOpened()==False möglich)
    return cv2.VideoCapture(idx, cv2.CAP_ANY)


def check_tcp(host, port, timeout=3):
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


def open_with_retry(source):
    """Öffnet lokale Quellen direkt; Netz-Streams mit Diagnose + Endlos-Retry.
    Gibt (capture, is_url) zurück. Unterscheidet 'Host/Port dicht' (Streamer aus
    oder Firewall) von 'Port offen, aber kein Video' (falscher Pfad/Modus)."""
    from urllib.parse import urlparse
    if not str(source).startswith("http"):
        cap = open_source(source)
        if not cap.isOpened():
            print(f"[main] FEHLER: Quelle '{source}' lässt sich nicht öffnen.")
            print("  Kamera suchen: python -m poc.main --scan")
            print("  Oder ohne Kamera testen: python -m poc.main --bench 100")
            sys.exit(1)
        return cap, False
    u = urlparse(str(source))
    host, port = u.hostname, u.port or 80
    attempt = 0
    while True:
        attempt += 1
        if not check_tcp(host, port):
            print(f"[main] Versuch {attempt}: {host}:{port} nicht erreichbar. "
                  f"Streamer auf dem PC starten ('--mode stream') und Windows-Firewall "
                  f"prüfen (Python für privates Netzwerk erlauben). Retry in 5s ...")
            time.sleep(5)
            continue
        cap = open_source(source)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                return cap, True
            cap.release()
        print(f"[main] Versuch {attempt}: {host}:{port} antwortet, aber liefert kein "
              f"Video unter {u.path or '/'} – läuft dort '--mode stream'? Retry in 5s ...")
        time.sleep(5)


def scan_cameras(max_idx=4):
    """Probiert Indizes 0..max_idx (je mit Lesen eines Frames) und meldet Treffer."""
    print(f"[scan] suche Kameras auf Indizes 0..{max_idx} ...")
    hits = 0
    for i in range(max_idx + 1):
        cap = open_source(str(i))
        if not cap.isOpened():
            print(f"  Index {i}: keine Kamera")
            cap.release()
            continue
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            h, w = frame.shape[:2]
            print(f"  Index {i}: GEFUNDEN ({w}x{h}) -> --source {i} nutzen")
            hits += 1
        else:
            print(f"  Index {i}: öffnet, liefert aber keine Frames (evtl. von anderer App belegt)")
    if not hits:
        print("[scan] nichts gefunden. Häufigste Ursachen: Kamera von Teams/Browser "
              "belegt, oder Windows-Einstellungen -> Datenschutz -> Kamera -> "
              "Desktop-Apps-Zugriff verweigert.")


def run_bench(n, detector, power):
    img_path = ensure_test_img()
    img = cv2.imread(img_path)
    if img is None:
        print("[bench] Testbild konnte nicht gelesen werden")
        sys.exit(1)
    print(f"[bench] {n} Inferenzen auf {img_path} ({img.shape[1]}x{img.shape[0]}) ...")
    times = []
    for i in range(n):
        t0 = time.perf_counter()
        found, conf, _ = detector.infer(img)
        times.append((time.perf_counter() - t0) * 1000)
        if (i + 1) % 20 == 0:
            s = power.snapshot()
            print(f"  {i+1}/{n} Mensch={found} conf={conf:.2f} "
                  f"infer={np.mean(times[-20:]):.1f}ms CPU={s['cpu_percent']}% "
                  f"{s['cpu_watts']}W{' (ECHT/RAPL)' if s['watt_real'] else ' (SCHÄTZUNG)'}")
    times = np.array(times)
    s = power.snapshot()
    print("\n===== BENCH-ERGEBNIS =====")
    print(f"Inferenz:  Mittel {times.mean():.1f}ms  Median {np.median(times):.1f}ms  p95 {np.percentile(times,95):.1f}ms")
    print(f"FPS-Äquivalent (Dauerlast, jedes Frame): {1000.0/times.mean():.1f} fps")
    print(f"CPU: {s['cpu_percent']}%  RAM: {s['mem_mb']}MB  Temp: {s['cpu_temp']}C")
    print(f"Leistung: {s['cpu_watts']}W {'(ECHT via RAPL)' if s['watt_real'] else '(SCHÄTZUNG via TDP – auf Zielmaschine läuft echte Messung!)'}")
    print(f"Energie/Jahr Dauerlast: {s['cpu_watts']*24*365/1000:.1f} kWh  (~{s['cpu_watts']*24*365/1000*0.35:.2f} EUR bei 0,35 EUR/kWh)")
    print("==========================")
    print("\nEinordnung: <5W Dauer = unkritisch für NUC/Mini-PC. "
          "5-12W = ok wenn nur bei Bedarf (alle N Frames). "
          ">15W Dauer = kritisch, dann Takt/Modell/Intervall anpassen.")


def run_stream(args):
    """Leichtgewichtig-Modus für den PC MIT Webcam: nur capturen + MJPEG serven.
    Keine Modelle, keine Analyse – damit die Watt-Messung auf der HA-Maschine
    nicht verfälscht wird und der Start schnell geht."""
    width = int(os.getenv("STREAM_WIDTH", "640"))
    fps = float(os.getenv("STREAM_FPS", "10"))
    quality = int(os.getenv("STREAM_QUALITY", "70"))
    threading.Thread(target=serve, args=(args.port,), daemon=True).start()
    cap = open_source(args.source)
    if not cap.isOpened():
        print(f"[stream] FEHLER: Kamera '{args.source}' lässt sich nicht öffnen.")
        print("  1) Andere Indizes suchen: python -m poc.main --scan")
        print("  2) Kamera evtl. von Teams/Browser/Discord belegt -> dort schließen.")
        print("  3) Windows: Einstellungen -> Datenschutz & Sicherheit -> Kamera ->")
        print("     Kamerazugriff + 'Desktop-Apps auf Kamera zugreifen lassen' AN.")
        sys.exit(1)
    print(f"[stream] {args.source} -> http://<diese-IP>:{args.port}/stream.mjpg "
          f"({width}px, {fps}fps, q{quality}) | Stop: Strg+C")
    period = 1.0 / max(fps, 0.5)
    global latest_jpeg
    try:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("[stream] Frame weg – warte 1s ...")
                time.sleep(1)
                continue
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)))
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            latest_jpeg = buf.tobytes()
            state.update({"mode": "stream",
                          "frame_ts": datetime.now().isoformat(timespec="seconds")})
            dt = period - (time.perf_counter() - t0)
            if dt > 0:
                time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        print("[stream] beendet.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["analyze", "stream"],
                    default=os.getenv("MODE", "analyze"),
                    help="stream=HIER (Webcam senden), analyze=DORT (Stream empfangen + auswerten)")
    ap.add_argument("--source", default=os.getenv("VIDEO_SOURCE", "0"))
    ap.add_argument("--bench", type=int, default=0, help="N Inferenzen ohne Kamera (Watt-Test)")
    ap.add_argument("--no-serve", action="store_true")
    ap.add_argument("--port", type=int, default=int(os.getenv("HTTP_PORT", "8000")))
    ap.add_argument("--scan", action="store_true",
                    help="Kamera-Indizes 0..4 durchprobieren und funktionierende melden")
    args = ap.parse_args()

    if args.scan:
        scan_cameras()
        return

    if args.mode == "stream":
        run_stream(args)
        return

    detect_every = int(os.getenv("DETECT_EVERY_N", "5"))
    csv_path = os.getenv("CSV_LOG", "poc_log.csv")

    power = PowerMonitor()
    power.start()
    detector = PersonDetector()
    lamp = LampRoi()

    if args.bench:
        run_bench(args.bench, detector, power)
        power.stop()
        return

    if not args.no_serve:
        threading.Thread(target=serve, args=(args.port,), daemon=True).start()

    cap, is_url = open_with_retry(args.source)
    if is_url:
        print(f"[main] Netz-Quelle (Stream vom privaten PC). Bei Abbrüchen wird "
              f"automatisch neu verbunden.")
    print(f"[main] Quelle OK: {args.source} | Inferenz jedes {detect_every}. Frame | "
          f"CSV: {csv_path} | Stop: Strg+C")
    csvf = open(csv_path, "w", newline="")
    w = csv.writer(csvf)
    w.writerow(["ts_iso", "person", "conf", "infer_ms", "lamp_on", "lamp_v",
                "cpu_w", "cpu_percent", "mem_mb", "watt_real"])

    global latest_jpeg
    frame_i = 0
    found, conf, boxes = False, 0.0, []
    infer_ms = 0.0
    t_last_log = time.time()
    watts = []
    t_start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if is_url:
                    # Stream kurz weg (Netz/Neustart): neu verbinden statt abbrechen
                    print("[main] Stream abgerissen – verbinde neu in 3s ...")
                    time.sleep(3)
                    cap.release()
                    cap = open_source(args.source)
                    continue
                print("[main] Frame lesen fehlgeschlagen (Kamera weg / Videoende).")
                break
            frame_i += 1

            if frame_i % detect_every == 0 or frame_i == 1:
                t0 = time.perf_counter()
                found, conf, boxes = detector.infer(frame)
                infer_ms = (time.perf_counter() - t0) * 1000

            lamp_on, lamp_v, (rx, ry, rw, rh) = lamp.check(frame)

            # Overlay
            for (b, c) in boxes:
                x1, y1, x2, y2 = map(int, b)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {c:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                          (0, 255, 255) if lamp_on else (80, 80, 80), 2)
            cv2.putText(frame, f"LAMPE {'AN' if lamp_on else 'AUS'} {lamp_v:.0f}",
                        (rx, max(0, ry - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255) if lamp_on else (200, 200, 200), 2)
            s = power.snapshot()
            cv2.putText(frame, f"Mensch:{'JA' if found else 'nein'} {conf:.2f} | "
                               f"{infer_ms:.0f}ms | {s['cpu_watts']}W",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            latest_jpeg = buf.tobytes()

            state.update({"person": found, "conf": round(conf, 3), "infer_ms": round(infer_ms, 1),
                          "lamp_on": lamp_on, "lamp_v": round(lamp_v, 1), **s,
                          "frame": frame_i})

            w.writerow([datetime.now().isoformat(timespec="seconds"),
                        found, round(conf, 3), round(infer_ms, 1),
                        lamp_on, round(lamp_v, 1), s["cpu_watts"], s["cpu_percent"],
                        s["mem_mb"], s["watt_real"]])

            if time.time() - t_last_log >= 2.0:
                t_last_log = time.time()
                watts.append(s["cpu_watts"])
                tag = "ECHT/RAPL" if s["watt_real"] else "SCHÄTZUNG"
                print(f"[{time.strftime('%H:%M:%S')}] Mensch={'JA' if found else 'nein'} ({conf:.2f}) | "
                      f"Infer {infer_ms:.0f}ms | Lampe={'AN' if lamp_on else 'AUS'} | "
                      f"CPU {s['cpu_percent']}% | {s['cpu_watts']}W ({tag}) | RAM {s['mem_mb']}MB")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        csvf.close()
        power.stop()
        if watts:
            avg = sum(watts) / len(watts)
            print(f"\n[main] Laufzeit {time.time()-t_start:.0f}s, Ø {avg:.1f}W -> "
                  f"{avg*24*365/1000:.1f} kWh/Jahr bei Dauerlast.")
        print(f"[main] Log gespeichert: {csv_path}")


if __name__ == "__main__":
    main()
