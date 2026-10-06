"""Home-Assistant-Anbindung für den PoC: NUR LESEN, nie schreiben.

Der PoC legt keine Sensoren in HA an und verändert dort nichts.
Dieses Modul holt ausschließlich den Verlauf deiner Watt-Entität
für den Strom-Graphen (tools/plot_watts.py --fetch).

Braucht einen langlebigen Token (nur lesend verwendet):
  HA-Profil (unten links) -> Sicherheit -> Langlebige Zugriffstoken -> erstellen.
ENV: HA_URL (z.B. http://homeassistant.local:8123), HA_TOKEN, HA_WATT_ENTITY
(z.B. sensor.strom_ha_host_leistung – schickst du mir, sobald der PoC läuft).
"""
import os

import requests

TIMEOUT = 10


def cfg():
    url = os.getenv("HA_URL", "").rstrip("/")
    token = os.getenv("HA_TOKEN", "")
    if not url or not token:
        raise RuntimeError("HA_URL/HA_TOKEN nicht gesetzt (.env).")
    return url, {"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"}


def get_history(entity_id: str, since_iso: str):
    """Holt den Verlauf einer HA-Entität (z.B. deiner Watt-Entität) als JSON-Liste.

    since_iso: '2026-10-06T12:00:00+02:00' (HA versteht ISO mit Zeitzone).
    Rückgabe: [{'state': '8.4', 'last_changed': '...'}, ...]
    """
    url, headers = cfg()
    r = requests.get(f"{url}/api/history/period/{since_iso}",
                     params={"filter_entity_id": entity_id,
                             "minimal_response": "", "no_attributes": ""},
                     headers=headers, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data[0] if data else []
