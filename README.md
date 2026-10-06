# Video Feed Analyzer – PoC (Mensch-Erkennung + Stromcheck)

PoC für deine Home-Assistant-Integrations-Idee: **lohnt sich 24/7-Bildanalyse
strommäßig überhaupt?** Antwort gibt dieser Prototyp durch echte Messung
auf der Zielmaschine.

**Setup (dein Fall):** Die Webcam hängt am privaten PC, die Analyse läuft auf
der HA-Maschine. Darum zwei Modi – Stream hier, Auswertung dort:

```
privater PC (MIT Webcam)          HA-Maschine (OHNE Kamera)
┌──────────────────────┐         ┌───────────────────────────┐
│ --mode stream        │  MJPEG  │ --mode analyze            │
│ Webcam -> :8000/     │ ──────▶ │ Stream -> Person/Lampe    │
│ stream.mjpg          │  Netz   │ -> Watt-Log (poc_log.csv) │
└──────────────────────┘         └───────────────────────────┘

   Watt-Entität lesen (nur lesen!)         poc_log.csv lesen
           │                                         │
           └──────────────┬──────────────────────────┘
                          ▼
               tools/plot_watts.py -> graph.png
```

Der PoC schreibt **nichts** nach Home Assistant – er liest nur den Verlauf
deiner Watt-Entität für den Graphen.

**Was der PoC tut:**
1. **Trigger 1 (NN):** Person-Erkennung mit MobileNet-SSD (OpenCV-DNN, CPU) – jedes N-te Frame.
2. **Trigger 2 (billig):** Lampen-ROI per Pixel-Mittelwert (HSV-V) → AN/AUS (~0,05 ms, vernachlässigbar).
3. **Strommessung:** RAPL (echte Watt, Linux) + PoC-CSV **plus** deine HA-Watt-Entität → gemeinsamer Graph.

> Warum **Python statt Rust** für den PoC? Die Inferenz-Kosten stecken zu >95 % im neuronalen Netz, nicht in der Sprache. Python + OpenCV-DNN ist in Minuten per Docker übertragbar und die Watt-Zahl ist repräsentativ. Rust wäre 10× Bauaufwand bei gleichem Messergebnis; für die spätere Produkt-Integration kann man immer noch wechseln.

---

## 1. HIER: Stream starten (privater PC mit Webcam)

Windows am einfachsten **nativ** (kein Docker-USB-Gefrickel):

```powershell
pip install -r requirements.txt
python -m poc.main --mode stream --source 0
# Test im Browser: http://localhost:8000/stream.mjpg
```

Linux alternativ per Docker: `docker compose --profile stream up streamer`.
Bandbreite drosseln über `.env`: `STREAM_WIDTH=640`, `STREAM_FPS=10`, `STREAM_QUALITY=70`
(ca. 1–3 Mbit/s – reicht für den PoC; MJPEG ist Absicht: null Latenz-Drama, `cv2` frisst es direkt).

**Erreichbarkeit von der HA-Maschine:** im Heimnetz einfach `http://<PC-IP>:8000/stream.mjpg`.
Von außerhalb **keine Portfreigabe** – nimm Tailscale/WireGuard auf beiden Maschinen, dann dieselbe URL mit VPN-IP.

## 2. DORT: Analyse auf der HA-Maschine (keine Kamera nötig)

```bash
git clone <dein-repo> && cd video_feed_analyzer
cp .env.example .env   # STREAM_URL + HA_TOKEN eintragen (unten)
docker compose up --build analyzer
```

- `STREAM_URL=http://<PC-IP>:8000/stream.mjpg` in `.env` (deine PC-IP im Heim-/VPN-Netz).
- Preview: `http://ha-maschine:8000`, Metriken: `.../metrics.json`, Log: `poc_log.csv` (ISO-Zeitstempel, passend zu HA-Verlauf).
- Reißt der Stream ab (PC schläft/neustartet), verbindet der Analyzer alle 3 s neu – kein Neustart nötig.
- Ohne laufenden Stream testen: `docker compose run --rm analyzer python -m poc.main --bench 200`.

## 3. Graph: Watt-Entität (lesen) + PoC-Log

**Token (einmalig):** HA-Profil (unten links) → Sicherheit → Langlebige Zugriffstoken → erstellen → als `HA_TOKEN` in `.env` **dort, wo der Plot läuft** (dein PC). Nie committen (`.env` ist git-ignoriert). Der Token wird nur lesend benutzt (`/api/history`), es wird nichts in HA angelegt oder verändert.

**Graph (sobald du mir URL + Namen deiner Watt-Entität schickst):**

```bash
pip install matplotlib requests   # hier auf deinem PC
# poc_log.csv von der HA-Maschine hierher kopieren, dann:
python tools/plot_watts.py --fetch --since "2026-10-06T12:00:00+02:00" --out watts.json
python tools/plot_watts.py --csv poc_log.csv --ha-json watts.json --out graph.png
```

`graph.png`: rote Treppe = deine Steckdosen-/System-Watt-Entität, orange Punkte = PoC-intern, grüne Fläche = „Person erkannt". Daraus lesen wir ab, ob die Analyse-Last im Rauschen untergeht oder trägt – und ob du die Idee verwirfst oder weiterbaust.

## 4. Strom-Test-Protokoll

5–10 Min laufen lassen, dann in `graph.png` / `poc_log.csv` auf die Watt achten:
- **< 5 W extra → unkritisch**, Idee weiterverfolgen.
- **5–12 W → ok, wenn** `DETECT_EVERY_N` hoch (z. B. 10) oder nur bei Bewegung inferiert wird.
- **> 15 W Dauer → kritisch** (~130 kWh/Jahr ≈ 45 €). Dann: größeres Intervall, NUC-Power-Profil, Coral-TPU.

**Größte Stellschrauben:** `DETECT_EVERY_N` (5 → 15 drittelt den Verbrauch), `STREAM_FPS` (weniger Frames = weniger Inferenzen), Power-Profil des Hosts.

---

## 5. Konfiguration (`.env`)

| Var | Beispiel | Wirkung |
|---|---|---|
| `STREAM_URL` | `http://192.168.178.10:8000/stream.mjpg` | **DORT:** wo der Stream herkommt |
| `VIDEO_SOURCE` | `0` / `/dev/video0` | **HIER:** welche Kamera sendet |
| `STREAM_WIDTH/FPS/QUALITY` | `640`/`10`/`70` | Bandbreite vs. Erkennungsqualität |
| `DETECT_EVERY_N` | `5` | nur jedes N-te Frame durchs Netz (Stromhebel!) |
| `CONF_THRES` | `0.5` | Person-Schwelle |
| `ROI` / `LAMP_THRES` | `80,5,15,15` / `150` | Lampen-Fenster (%), AN-Schwelle |
| `HA_URL` / `HA_TOKEN` | `http://homeassistant.local:8123` | nur für Plot-Fetch (`--fetch`), Analyzer braucht sie nicht |
| `HA_WATT_ENTITY` | `sensor.xyz_leistung` | deine Watt-Entität für den Graphen |
| `TDP_WATTS`/`IDLE_WATTS` | `15`/`3` | nur Fallback-Schätzung ohne RAPL |

## 6. Repo-Struktur

```
docker-compose.yml    # analyzer (HA) + streamer --profile stream (PC)
Dockerfile            # python:3.12-slim + opencv (~300 MB, kein torch!)
requirements.txt      # opencv-headless, numpy, psutil, requests
poc/main.py           # --mode stream | analyze, Reconnect, CSV, MJPEG, --bench
poc/detector.py       # MobileNet-SSD (OpenCV-DNN), nur Klasse "person"
poc/power.py          # RAPL-Watt (Linux) + Fallback-Schätzung
poc/roi_lamp.py       # Lampen-ROI-Trigger
poc/ha.py             # HA-Verlauf lesen (nur lesen, kein Schreiben)
tools/plot_watts.py   # graph.png aus poc_log.csv + HA-Watt-Entität
models/               # SSD-Modell + Testbild (Autodownload, git-ignoriert)
src/main.rs           # altes leeres Rust-Gerüst, vom PoC nicht benutzt
```

## 7. Nächste Schritte → echte HA-Integration

- Zonen: ROI-Masken pro Zone + `DETECT_EVERY_N` dynamisch (nur aktive Zone inferieren).
- `metrics.json`/HA-Sensoren per HA-Automation in Zonen-Trigger + Waschmaschinen-Benachrichtigung gießen.
- HA-Add-on: `config.yaml` um den Analyzer-Container legen, Kamera via `camera`-Entity statt MJPEG-URL.
