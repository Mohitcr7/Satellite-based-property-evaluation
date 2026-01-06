import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAPBOX_TOKEN = "sk.eyJ1IjoibW9oaXRjcjc3IiwiYSI6ImNtanZ3a3I2MjNkOWkzZnF5eW82aDBkdHcifQ.zsdkqLn4iBXiSX3uLgnbLg"
OUT_DIR = "data/images_rgb_test"
ZOOM = 17
SIZE = 640

REQUEST_DELAY = 0.08     # ~12 requests / second (safe)
MAX_RETRIES = 3
TIMEOUT = 10

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("test_cords.csv")

# --------------------------------------------------
# Reuse HTTP session (IMPORTANT)
# --------------------------------------------------
session = requests.Session()

def download_image(prop_id, lat, lon):
    url = (
        f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/"
        f"{lon},{lat},{ZOOM}/{SIZE}x{SIZE}"
    )
    params = {"access_token": MAPBOX_TOKEN}

    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)

            if r.status_code == 200:
                with open(f"{OUT_DIR}/{prop_id}.png", "wb") as f:
                    f.write(r.content)
                return True

            elif r.status_code == 429:
                # Rate limited → back off
                time.sleep(2 + attempt)
            else:
                time.sleep(1)

        except requests.exceptions.RequestException:
            time.sleep(2 + attempt)

    return False

# --------------------------------------------------
# Download loop
# --------------------------------------------------
for _, row in tqdm(df.iterrows(), total=len(df)):
    prop_id = row["id"]
    lat = row["lat"]
    lon = row["long"]

    out_path = f"{OUT_DIR}/{prop_id}.png"
    if os.path.exists(out_path):
        continue  # skip already downloaded

    download_image(prop_id, lat, lon)
    time.sleep(REQUEST_DELAY)