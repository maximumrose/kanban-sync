import json
from pathlib import Path
from trello_api import get_lists

ROOT = Path(__file__).resolve().parents[1]
cfg = json.loads((ROOT / "config" / "board_layout.json").read_text())
cols = cfg["columns"]

lists = get_lists()
print("Trello lists:")
for i, lst in enumerate(lists):
    print(f"{i}: {lst['name']} ({lst['id']})")

mapping = {}
for col in cols:
    idx = int(input(f"Map column '{col['name']}' to Trello list index: ").strip())
    mapping[col["name"]] = lists[idx]["id"]

out = {"column_to_list": mapping}
(ROOT / "config" / "trello_mapping.json").write_text(json.dumps(out, indent=2))
print("Saved → config/trello_mapping.json")
