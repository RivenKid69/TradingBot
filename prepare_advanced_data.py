# -*- coding: utf-8 -*-
"""
prepare_advanced_data.py
---------------------------------------------------------------
Fetches the Fear & Greed index and writes a normalized CSV at:
    data/fear_greed.csv
Run as:
    python prepare_advanced_data.py
"""
import os, time, csv, json
import requests

OUT_DIR = os.path.join("data")
OUT_PATH = os.path.join(OUT_DIR, "fear_greed.csv")
API = "https://api.alternative.me/fng/"

def fetch_fng(limit: int = 0) -> list:
    # limit=0 means "all available"
    params = {"limit": limit, "format": "json"}
    r = requests.get(API, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()["data"]
    rows = []
    for d in data:
        # API returns time in UNIX seconds (string), value as string
        ts = int(d["timestamp"])
        val = int(d["value"])
        # Floor to hour
        ts = (ts // 3600) * 3600
        rows.append((ts, val))
    # Dedup & sort
    rows = sorted({(ts, val) for ts, val in rows})
    return rows

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = fetch_fng(limit=0)
    # Load existing CSV (if any) and detect last timestamp
    existing_rows = []
    last_ts = None
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r", newline="") as rf:
            rdr = csv.reader(rf)
            header = next(rdr, None)
            for r in rdr:
                if not r:
                    continue

                try:
                    ts = int(r[0])
                except (TypeError, ValueError):
                    continue

                if len(r) < 2:
                    # Skip rows without a value column to avoid indexing errors
                    continue

                raw_val = r[1]
                try:
                    val = int(raw_val)
                except (TypeError, ValueError):
                    try:
                        val = float(raw_val)
                    except (TypeError, ValueError):
                        continue

                existing_rows.append([ts, val])
            if existing_rows:
                last_ts = max(x[0] for x in existing_rows)

    # Keep only strictly new daily points
    new_rows = [r for r in rows if last_ts is None or int(r[0]) > int(last_ts)]
    if not new_rows:
        print(f"✓ Fear & Greed up-to-date at {OUT_PATH} ({len(existing_rows)} rows).")
        return

    # Append incrementally with atomic replace
    all_rows = existing_rows + [[int(t), int(v)] for t, v in new_rows]
    tmp = OUT_PATH + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp","fear_greed_value"])
        w.writerows(all_rows)
    os.replace(tmp, OUT_PATH)
    print(f"✓ Fear & Greed updated at {OUT_PATH} (+{len(new_rows)} new rows, total {len(all_rows)}).")

if __name__ == "__main__":
    main()
