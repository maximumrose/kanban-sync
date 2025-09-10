import os, requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
loaded = load_dotenv(ENV_PATH)

def need(name):
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing {name}. Check {ENV_PATH}")
    return v

TRELLO_KEY = need("TRELLO_KEY")
TRELLO_TOKEN = need("TRELLO_TOKEN")
TRELLO_BOARD_ID = need("TRELLO_BOARD_ID")  # can be the board short link from the URL

url = f"https://api.trello.com/1/boards/{TRELLO_BOARD_ID}/lists"
params = {"key": TRELLO_KEY, "token": TRELLO_TOKEN, "fields": "name"}

r = requests.get(url, params=params)
r.raise_for_status()
for lst in r.json():
    print(lst["id"], lst["name"])