import os, requests
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

K = os.getenv("TRELLO_KEY")
T = os.getenv("TRELLO_TOKEN")
B = os.getenv("TRELLO_BOARD_ID")
BASE = "https://api.trello.com/1"

def _req(method, path, **params):
    params.update(key=K, token=T)
    r = requests.request(method, f"{BASE}{path}", params=params)
    r.raise_for_status()
    return r.json()

def get_lists():
    return _req("GET", f"/boards/{B}/lists", fields="name,id", filter="open")

def create_card(list_id, name):
    return _req("POST", "/cards", idList=list_id, name=name)

def move_card(card_id, list_id):
    return _req("PUT", f"/cards/{card_id}", idList=list_id)

def archive_card(card_id):
    return _req("PUT", f"/cards/{card_id}/closed", value="true")

def rename_card(card_id, name):
    return _req("PUT", f"/cards/{card_id}", name=name)
