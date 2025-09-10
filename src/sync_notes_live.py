# Simpler scan-based sticky -> Trello sync (no motion prediction)
# - Board is scanned at fixed intervals (default 300ms)
# - Identity via appearance: HSV histogram + perceptual aHash
# - NEW stickies promoted only after a few consistent scans (debounce)
# - MOVE when the same sticky is stably seen in a new column
# - ARCHIVE only after 60s missing (time-based)
# - Optional OCR for initial title + post-create rename attempts

import json, sys, time, math
from pathlib import Path
import numpy as np
import cv2
import difflib

from trello_api import create_card, move_card, archive_card, rename_card
from ocr_utils import ocr_text

# ---------------- Config load ----------------
ROOT = Path(__file__).resolve().parents[1]
cfg_path = ROOT / "config" / "board_layout.json"
map_path = ROOT / "config" / "trello_mapping.json"

def die(msg): sys.exit("Config error: " + msg)
if not cfg_path.exists(): die("Missing board_layout.json. Run calibrate_board.py")
if not map_path.exists(): die("Missing trello_mapping.json. Run map_columns_to_trello.py")
try: CFG = json.loads(cfg_path.read_text())
except Exception as e: die(f"board_layout.json unreadable: {e}")
try: MAP = json.loads(map_path.read_text())
except Exception as e: die(f"trello_mapping.json unreadable: {e}")
for k in ["warp","columns"]:
    if k not in CFG: die(f"board_layout.json missing key: {k}")
if not all(x in CFG["warp"] for x in ("M","W","H")): die("warp must contain M, W, H")
if "column_to_list" not in MAP: die("trello_mapping.json missing key: column_to_list")

M = np.array(CFG["warp"]["M"], dtype=np.float32)
W, H = int(CFG["warp"]["W"]), int(CFG["warp"]["H"])
COLUMNS = CFG["columns"]
COL_TO_LIST = MAP["column_to_list"]
for c in COLUMNS:
    if c["name"] not in COL_TO_LIST:
        die(f"List mapping missing for column '{c['name']}'. Run map_columns_to_trello.py")

def col_for_x(x):
    for c in COLUMNS:
        if c["x0"] <= x < c["x1"]:
            return c["name"]
    return None

# ---------------- Detection params ----------------
COLOR_RANGES = [
    ((15,  80, 100), (35, 255, 255)),  # yellow
    ((5,   80, 100), (15, 255, 255)),  # orange
    ((35,  60,  70), (85, 255, 255)),  # green
    ((90,  50,  70), (130,255, 255)),  # blue
    ((140, 50,  70), (179,255, 255)),  # magenta/pink
    ((0,   80, 100), (5,  255, 255)),  # red low band
]
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.001, 0.20
ASPECT_MIN, ASPECT_MAX = 0.6, 1.8

# Throttle & debounce (power-friendly)
SCAN_INTERVAL_MS     = 300   # scan every 0.3s (adjust 200-400ms to taste)
CREATE_SEEN_SCANS    = 2     # scans required before creating Trello card
MOVE_STABLE_SCANS    = 2     # scans in same new column before moving
CANDIDATE_FORGET_SEC = 10.0  # drop candidate if not seen within this window
MISSING_TTL_SEC      = 60.0  # archive only after 60s missing

# Appearance similarity thresholds
HIST_BINS_H, HIST_BINS_S = 24, 24
HIST_INTERSECT_MIN   = 0.22      # gate for considering a match (0..1)
AHASH_MIN_SIM        = 0.60      # 1 - Hamming/64
SIM_WEIGHT_HIST      = 0.65
SIM_WEIGHT_AHASH     = 0.35
RECOVER_SIM_THRESH   = 0.55      # to match MISSING (reacquire)
ACTIVE_SIM_THRESH    = 0.80      # to match ACTIVE (same sticky)
CAND_SIM_THRESH      = 0.70      # to persist a candidate across scans

# OCR/rename cadence
OCR_FOR_CREATE       = True      # run a quick OCR when creating card
OCR_RENAME_EVERY_N   = 3         # rename attempt every N scans
RENAME_MIN_CONF      = 60.0
IMPROVE_DELTA        = 8.0
RENAME_WINDOW_SCANS  = 30

ALPHA_HIST           = 0.3       # EMA update for hist when matched

# ---------------- Helpers ----------------
def detect_notes(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, 1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, 2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out, area_tot = [], W*H
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_RATIO*area_tot or area > MAX_AREA_RATIO*area_tot: continue
        x,y,w,h = cv2.boundingRect(c)
        ar = w / max(1.0, h)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX): continue
        eps = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 4 or not cv2.isContourConvex(approx): continue
        cx, cy = x + w//2, y + h//2
        out.append((x,y,w,h,cx,cy))
    return out, mask

def clamp_box(box):
    x,y,w,h = box
    x = int(max(0, min(x, W-1))); y = int(max(0, min(y, H-1)))
    w = int(max(1, min(w, W-x)));  h = int(max(1, min(h, H-y)))
    return (x,y,w,h)

def roi_from_box(img, box, m=6):
    x,y,w,h = box
    return img[max(0,y-m):min(H,y+h+m), max(0,x-m):min(W,x+w+m)]

def hs_hist(img_bgr, box):
    roi = roi_from_box(img_bgr, box, m=0)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[HIST_BINS_H,HIST_BINS_S],[0,180,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist.astype(np.float32)

def ahash64(img_bgr, box):
    roi = cv2.cvtColor(roi_from_box(img_bgr, box), cv2.COLOR_BGR2GRAY)
    small = cv2.resize(roi, (8,8), interpolation=cv2.INTER_AREA)
    avg = float(small.mean())
    bits = (small > avg).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(bool(b))
    return val  # 64-bit int

def ahash_sim(a, b):
    # similarity in [0,1]; 1 is identical
    hd = bin((a ^ b) & ((1<<64)-1)).count("1")
    return 1.0 - hd/64.0

def hist_intersection(h1, h2):
    if h1 is None or h2 is None: return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT))

def text_sim(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def sim_score(hsim, hashsim):
    return SIM_WEIGHT_HIST*hsim + SIM_WEIGHT_AHASH*hashsim

def center_x(box):
    return box[0] + box[2]/2.0

def column_for_box(box):
    return col_for_x(center_x(box))

# ---------------- Registry ----------------
class Entry:
    # state ∈ {"CANDIDATE","ACTIVE","MISSING"}
    def __init__(self, sid, box, hist, ahash, now_ts):
        self.id = sid
        self.box = clamp_box(box)
        self.hist = hist
        self.ahash = ahash

        self.state = "CANDIDATE"
        self.seen_scans = 1
        self.last_seen = now_ts
        self.cand_last_seen = now_ts
        self.missing_since = None

        self.measured_col = column_for_box(self.box)
        self.col_stable_scans = 1
        self.committed_col = None

        # Trello
        self.card_id = None
        self.card_name = None

        # OCR cache
        self.best_text = ""
        self.best_conf = 0.0
        self.rename_attempts = 0

    def update_seen(self, box, hist, ahash, now_ts):
        self.box = clamp_box(box)
        # EMA hist
        if self.hist is None: self.hist = hist
        else: self.hist = (1-ALPHA_HIST)*self.hist + ALPHA_HIST*hist
        self.ahash = ahash
        self.last_seen = now_ts
        if self.state == "CANDIDATE":
            self.cand_last_seen = now_ts
            self.seen_scans += 1

        # Column stability
        col = column_for_box(self.box)
        if col == self.measured_col:
            self.col_stable_scans += 1
        else:
            self.measured_col = col
            self.col_stable_scans = 1

# Entries by id
entries = {}
global next_id
next_id = 1

def best_match(hist, ah, pool_ids):
    """Return (entry_id, score, hsim, hashsim) or (None, 0, 0, 0)."""
    best_id, best_score, best_h, best_a = None, 0.0, 0.0, 0.0
    for eid in pool_ids:
        e = entries[eid]
        hsim = hist_intersection(hist, e.hist)
        if hsim < HIST_INTERSECT_MIN: 
            continue
        asim = ahash_sim(ah, e.ahash)
        if asim < AHASH_MIN_SIM:
            continue
        s = sim_score(hsim, asim)
        if s > best_score:
            best_id, best_score, best_h, best_a = eid, s, hsim, asim
    return best_id, best_score, best_h, best_a

# ---------------- Main loop (scan-based) ----------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    die("Camera not found / permission issue.")

# (Optional) downshift camera load a bit
# cap.set(cv2.CAP_PROP_FPS, 15)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH,  960)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

print("Controls: q=quit, m=toggle mask")
show_mask = False
scan_idx = 0
last_scan = 0.0

while True:
    # Throttle scanning for power
    now = time.monotonic()
    sleep_s = max(0.0, (SCAN_INTERVAL_MS/1000.0) - (now - last_scan))
    if sleep_s > 0: time.sleep(sleep_s)
    last_scan = time.monotonic()
    scan_idx += 1

    ok, frame = cap.read()
    if not ok: break

    warped = cv2.warpPerspective(frame, M, (W, H))
    dets, mask = detect_notes(warped)

    # Precompute descriptors for all detections
    obs = []  # list of dicts: {box,hist,ahash,col}
    for (x,y,w,h,_,_) in dets:
        box = (int(x),int(y),int(w),int(h))
        hist = hs_hist(warped, box)
        ah = ahash64(warped, box)
        obs.append({"box": box, "hist": hist, "ahash": ah, "col": column_for_box(box)})

    # Mark all entries as unseen this scan
    seen_ids = set()

    # Pools
    active_ids   = [eid for eid,e in entries.items() if e.state == "ACTIVE"]
    missing_ids  = [eid for eid,e in entries.items() if e.state == "MISSING"]
    cand_ids     = [eid for eid,e in entries.items() if e.state == "CANDIDATE"]

    # Greedy matching for each observation:
    for ob in obs:
        box, hist, ah, col = ob["box"], ob["hist"], ob["ahash"], ob["col"]

        # 1) Prefer recovering a MISSING entry
        mid, mscore, _, _ = best_match(hist, ah, missing_ids)
        if mid is not None and mscore >= RECOVER_SIM_THRESH and mid not in seen_ids:
            e = entries[mid]
            e.state = "ACTIVE"
            e.missing_since = None
            e.update_seen(box, hist, ah, last_scan)
            seen_ids.add(mid)
            continue

        # 2) Try binding to an ACTIVE entry (same sticky)
        aid, ascore, _, _ = best_match(hist, ah, active_ids)
        if aid is not None and ascore >= ACTIVE_SIM_THRESH and aid not in seen_ids:
            e = entries[aid]
            e.update_seen(box, hist, ah, last_scan)
            seen_ids.add(aid)
            continue

        # 3) Try to continue a CANDIDATE
        cid, cscore, _, _ = best_match(hist, ah, cand_ids)
        if cid is not None and cscore >= CAND_SIM_THRESH and cid not in seen_ids:
            e = entries[cid]
            e.update_seen(box, hist, ah, last_scan)
            seen_ids.add(cid)
            continue

        # 4) Otherwise, start a new candidate
        eid = next_id; next_id += 1
        e = Entry(eid, box, hist, ah, last_scan)
        entries[eid] = e
        seen_ids.add(eid)
        cand_ids.append(eid)

    # Handle entries not seen this scan
    now_ts = last_scan
    for eid, e in list(entries.items()):
        if eid in seen_ids:
            continue
        if e.state == "ACTIVE":
            e.state = "MISSING"
            if e.missing_since is None:
                e.missing_since = now_ts
        elif e.state == "CANDIDATE":
            # forget if stale
            if (now_ts - e.cand_last_seen) > CANDIDATE_FORGET_SEC:
                del entries[eid]

    # Promote candidates, move actives, rename, archive
    for eid, e in list(entries.items()):
        x,y,w,h = e.box
        col = e.measured_col

        # --- promote CANDIDATE to ACTIVE (create Trello) ---
        if e.state == "CANDIDATE":
            if e.seen_scans >= CREATE_SEEN_SCANS and e.col_stable_scans >= MOVE_STABLE_SCANS and col:
                list_id = COL_TO_LIST.get(col)
                if list_id:
                    title = f"Note {int(time.time())}"
                    if OCR_FOR_CREATE:
                        roi = roi_from_box(warped, e.box, m=8)
                        t, c = ocr_text(roi)
                        if t and len(t) >= 3:
                            e.best_text, e.best_conf = t, c
                            title = t
                    try:
                        card = create_card(list_id, title)
                        e.card_id = card["id"]
                        e.card_name = title
                        e.committed_col = col
                        e.state = "ACTIVE"
                        e.rename_attempts = 0
                        print(f"[trello] create '{title}' in {col} -> {card.get('shortUrl','')}")
                    except Exception as ex:
                        print("[trello] create error:", ex)

        # --- ACTIVE: move if stable in a *new* column ---
        if e.state == "ACTIVE" and e.card_id:
            if e.committed_col and col and col != e.committed_col and e.col_stable_scans >= MOVE_STABLE_SCANS:
                list_id = COL_TO_LIST.get(col)
                if list_id:
                    try:
                        move_card(e.card_id, list_id)
                        e.committed_col = col
                        print(f"[trello] move card {e.card_id} -> {col}")
                    except Exception as ex:
                        print("[trello] move error:", ex)

        # --- Optional OCR rename for a while after creation ---
        if e.state == "ACTIVE" and e.card_id and (scan_idx % OCR_RENAME_EVERY_N == 0) and e.rename_attempts < RENAME_WINDOW_SCANS:
            roi = roi_from_box(warped, e.box, m=8)
            t, c = ocr_text(roi)
            if t and c > (e.best_conf + 0.5):
                e.best_text, e.best_conf = t, c
            if e.best_text and len(e.best_text) >= 3:
                better = (e.best_conf >= RENAME_MIN_CONF) or \
                         ((e.best_conf - (0.0 if (e.card_name or '').startswith("Note ") else 50.0)) >= IMPROVE_DELTA)
                if better and e.best_text != (e.card_name or ""):
                    try:
                        rename_card(e.card_id, e.best_text)
                        e.card_name = e.best_text
                        print(f"[trello] rename card {e.card_id} -> '{e.best_text}' (conf {e.best_conf:.1f})")
                    except Exception:
                        pass
            e.rename_attempts += 1

        # --- MISSING: archive only after TTL ---
        if e.state == "MISSING" and e.missing_since is not None:
            if (now_ts - e.missing_since) >= MISSING_TTL_SEC:
                if e.card_id:
                    try:
                        archive_card(e.card_id)
                        print(f"[trello] archived {e.card_id} after {int(MISSING_TTL_SEC)}s missing (id {eid})")
                    except Exception as ex:
                        print("[trello] archive error:", ex)
                del entries[eid]

    # ---- Draw overlay ----
    vis = warped.copy()
    for eid, e in entries.items():
        x,y,w,h = e.box
        state = e.state if e.state != "MISSING" else f"MISSING {int(now_ts - (e.missing_since or now_ts))}s"
        label = f"id {eid} {state} -> {e.measured_col or '?'}"
        cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(vis, label, (x, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    cv2.imshow("warped (simple scan + sync)", vis)
    if show_mask: cv2.imshow("mask", mask)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            try: cv2.destroyWindow("mask")
            except cv2.error: pass

cap.release()
cv2.destroyAllWindows()
