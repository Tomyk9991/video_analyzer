"""Watt-Graph: PoC-Log (Inferenzen/Person) + HA-Watt-Entität in einem Bild.

Verwendung (hier auf deinem PC, braucht nur die beiden Dateien):
  # 1) Verlauf deiner Watt-Entität aus HA ziehen (auf der HA-Maschine oder hier mit VPN):
  python tools/plot_watts.py --fetch --since "2026-10-06T12:00:00+02:00" --out watts.json
  # 2) Plot bauen (poc_log.csv von der HA-Maschine hierher kopieren):
  python tools/plot_watts.py --csv poc_log.csv --ha-json watts.json --out graph.png

Setup: pip install matplotlib requests
ENV für --fetch: HA_URL, HA_TOKEN, HA_WATT_ENTITY
"""
import argparse
import csv
import json
import os
import sys
from datetime import datetime


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="poc_log.csv")
    ap.add_argument("--ha-json", default="watts.json")
    ap.add_argument("--fetch", action="store_true",
                    help="Watt-Verlauf live aus HA holen und als --ha-json speichern")
    ap.add_argument("--since", default="",
                    help="Startzeit ISO, z.B. 2026-10-06T12:00:00+02:00")
    ap.add_argument("--out", default="graph.png")
    return ap.parse_args()


def fetch_watts(since, path):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from poc.ha import get_history
    entity = os.getenv("HA_WATT_ENTITY", "")
    if not entity:
        print("HA_WATT_ENTITY nicht gesetzt (.env) – "
              "z.B. sensor.strom_ha_host_leistung")
        sys.exit(1)
    data = get_history(entity, since)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"[plot] {len(data)} Messpunkte von {entity} -> {path}")


def load_csv(path):
    ts, person, watts = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                ts.append(datetime.fromisoformat(row["ts_iso"]))
                person.append(1 if row["person"] == "True" else 0)
                watts.append(float(row["cpu_w"]))
            except (ValueError, KeyError):
                continue
    return ts, person, watts


def main():
    args = parse_args()
    if args.fetch:
        if not args.since:
            print("--since fehlt, z.B. --since \"2026-10-06T12:00:00+02:00\"")
            sys.exit(1)
        fetch_watts(args.since, args.ha_json)
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Bitte: pip install matplotlib requests")
        sys.exit(1)

    ts, person, poc_w = load_csv(args.csv)
    if not ts:
        print(f"Keine Daten in {args.csv} (PoC lief noch nicht / falscher Pfad).")
        sys.exit(1)
    with open(args.ha_json) as f:
        ha = json.load(f)
    ha_ts = [datetime.fromisoformat(p["last_changed"]) for p in ha]
    ha_w = [float(p["state"]) for p in ha]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    if ha_ts:
        ax1.step(ha_ts, ha_w, where="post", label="HA Watt-Entität (Steckdose/RAPL)",
                 color="tab:red")
    if poc_w:
        ax1.plot(ts, poc_w, ".", markersize=2, alpha=0.5,
                 label="PoC-intern (RAPL/Schätzung)", color="tab:orange")
    ax1.set_ylabel("Watt")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.fill_between(ts, 0, person, step="mid", alpha=0.2, color="tab:green",
                     label="Person erkannt")
    ax2.set_ylabel("Person (0/1)")
    ax2.set_ylim(-0.1, 1.5)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.suptitle("Video-PoC: Strom vs. Erkennung – Entscheidungshilfe")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print(f"[plot] {args.out} geschrieben ({len(ts)} PoC-Zeilen, {len(ha_ts)} HA-Punkte).")


if __name__ == "__main__":
    main()
