"""Strom-/Lastmessung für den PoC.

Strategie (Linux Zielmaschine):
1. Intel/AMD RAPL via /sys/class/powercap/intel-rapl*/energy_uj -> echte CPU-Watt.
2. Fallback: psutil CPU-% + TDP-Schätzung (klar als Schätzung markiert).

Diese Datei bewusst ohne Fremdabhängigkeiten außer psutil.
"""
import glob
import os
import threading
import time
import psutil


def find_rapl_files():
    # package domain: intel-rapl:0/energy_uj (Gesamt-CPU)
    files = sorted(glob.glob("/sys/class/powercap/intel-rapl*/energy_uj"))
    if not files:
        files = sorted(glob.glob("/sys/class/powercap/*/energy_uj"))
    return files


def read_uj(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


class PowerMonitor(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.rapl_files = find_rapl_files()
        self.has_rapl = len(self.rapl_files) > 0
        self.cpu_watts = 0.0      # RAPL-gemessen (0 wenn n/a)
        self.cpu_percent = 0.0
        self.mem_mb = 0.0
        self.cpu_temp = 0.0
        self.is_estimate = not self.has_rapl
        self.tdp = float(os.getenv("TDP_WATTS", "15"))
        self.idle_w = float(os.getenv("IDLE_WATTS", "3"))
        self._stop = threading.Event()
        self._last_energy = None
        self._last_t = None
        if self.has_rapl:
            print(f"[power] RAPL gefunden: {self.rapl_files} -> echte Watt-Messung")
        else:
            print(f"[power] kein RAPL (typisch Windows/VM/Mac). "
                  f"Fallback: CPU-% + TDP-Schätzung (TDP={self.tdp}W). "
                  f"Auf der Linux-Zielmaschine kommt echte Watt-Messung.")

    def _read_temp(self):
        for p in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
            try:
                with open(p) as f:
                    v = int(f.read().strip())
                    return v / 1000.0 if v > 1000 else float(v)
            except Exception:
                continue
        return 0.0

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self._stop.is_set():
            t = time.time()
            # RAPL-Delta
            if self.has_rapl:
                total_uj = 0
                ok = True
                # Nur Top-Level-Package nehmen (erstes File), sonst doppelt gezählt
                uj = read_uj(self.rapl_files[0])
                if uj is None:
                    ok = False
                else:
                    total_uj = uj
                if ok:
                    if self._last_energy is not None:
                        dt = t - self._last_t
                        dJ = (total_uj - self._last_energy) / 1e6
                        if dt > 0 and dJ >= 0:
                            self.cpu_watts = dJ / dt
                    self._last_energy = total_uj
                    self._last_t = t
            else:
                # Schätzung
                self.cpu_watts = self.idle_w + (self.tdp - self.idle_w) * (self.cpu_percent / 100.0)

            self.cpu_percent = psutil.cpu_percent(interval=None)
            self.mem_mb = psutil.Process().memory_info().rss / 1024 / 1024
            self.cpu_temp = self._read_temp()
            if not self.has_rapl:
                # Schätzung nach CPU-% Update neu berechnen
                self.cpu_watts = self.idle_w + (self.tdp - self.idle_w) * (self.cpu_percent / 100.0)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()

    def snapshot(self):
        return {
            "cpu_watts": round(self.cpu_watts, 2),
            "cpu_percent": round(self.cpu_percent, 1),
            "mem_mb": round(self.mem_mb, 1),
            "cpu_temp": round(self.cpu_temp, 1),
            "watt_real": self.has_rapl,
        }
