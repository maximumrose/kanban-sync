import json, time, sys
from pathlib import Path
import numpy as np
import cv2

from trello_api import create_card, move_card, archive_card, rename_card
from ocr_utils import ocr_text

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
COL_NAMES = [c["name"] for c in COLUMNS]
missing = [n for n in COL_NAMES if n not in COL_TO_LIST]
if missing: die(f"mapping missing columns: {missing}. Re-run map_columns_to_trello.py")

def col_for_x(x):
    for c in COLUMNS:
        if c["x0"] <= x < c["x1"]:
            return c["name"]
    return None

# ---- detection params ----
COLOR_RANGES = [((15,80,100),(35,255,255)), ((5,80,100),(15,255,255)),
                ((35,60,70),(85,255,255)), ((90,50,70),(130,255,255)),
                ((140,50,70),(179,255,255)), ((0,80,100),(5,255,255))]
MIN_AREA_RATIO, MAX_AREA_RATIO = 0.001, 0.20
ASPECT_MIN, ASPECT_MAX = 0.6, 1.8
MAX_ASSIGN_DIST, MAX_MISSES = 70, 10

# ---- tracking / gating ----
tracks, next_id = {}, 1          # tid -> dict(x,y,box,misses)
track_to_card = {}               # tid -> trello card id
committed_col = {}               # tid -> last Trello-synced column
measured_col = {}                # tid -> measured column
stable = {}                      # tid -> frames in same measured column

STABLE_FRAMES = 5                # debounce
STILL_FRAMES  = 8                # must be still N frames
SPEED_PX      = 2.0
still = {}                       # tid -> still frames
prev_pos = {}                    # tid -> (x,y)

# ---- OCR improvement window / renaming ----
OCR_EVERY_N = 3                  # run OCR every N frames per track
OCR_RENAME_WINDOW = 30           # frames after create to keep improving name
RENAME_MIN_CONF = 60.0           # rename when OCR >= this
IMPROVE_DELTA = 8.0              # or when improves by this many points
ocr_best = {}                    # tid -> (text, conf)
ocr_attempts = {}                # tid -> frames since create
card_name = {}                   # tid -> last card name we set

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

def assign_tracks(dets):
    global next_id
    track_ids = list(tracks.keys())
    T, D = len(track_ids), len(dets)
    used_tracks, used_dets, assigned = set(), set(), {}

    if T and D:
        ct = np.array([[tracks[i]['x'], tracks[i]['y']] for i in track_ids], np.float32)
        cd = np.array([[d[4], d[5]] for d in dets], np.float32)
        d2 = np.sqrt(((ct[:,None,:]-cd[None,:,:])**2).sum(axis=2))
        while True:
            i_t, i_d = np.unravel_index(np.argmin(d2, axis=None), d2.shape)
            dist = d2[i_t, i_d]
            if np.isinf(dist) or dist > MAX_ASSIGN_DIST: break
            assigned[track_ids[i_t]] = i_d
            used_tracks.add(i_t); used_dets.add(i_d)
            d2[i_t,:] = np.inf; d2[:,i_d] = np.inf
            if len(used_tracks)==T or len(used_dets)==D: break

    for idx_t, tid in enumerate(track_ids):
        if idx_t in used_tracks:
            x,y,w,h,cx,cy = dets[assigned[tid]]
            tracks[tid].update(x=cx, y=cy, box=(x,y,w,h), misses=0)
        else:
            tracks[tid]['misses'] += 1

    for i, d in enumerate(dets):
        if i in used_dets: continue
        x,y,w,h,cx,cy = d
        tid = next_id; next_id += 1
        tracks[tid] = dict(x=cx, y=cy, box=(x,y,w,h), misses=0)
        measured_col[tid] = None; stable[tid] = 0; still[tid] = 0
        prev_pos[tid] = (cx, cy)
        ocr_best[tid] = ("", 0.0); ocr_attempts[tid] = 0

    gone = [tid for tid,t in tracks.items() if t['misses'] > MAX_MISSES]
    for tid in gone:
        if tid in track_to_card:
            try:
                archive_card(track_to_card[tid])
                print(f"[trello] archived {track_to_card[tid]} (track {tid})")
            except Exception as e:
                print("[trello] archive error:", e)
            track_to_card.pop(tid, None)
            card_name.pop(tid, None)
        for d in (tracks, measured_col, committed_col, stable, still, prev_pos, ocr_best, ocr_attempts):
            d.pop(tid, None)
        print(f"[event] remove track {tid}")

cap = cv2.VideoCapture(0)
if not cap.isOpened(): raise SystemExit("Camera not found / permission issue.")
print("Controls: q=quit, m=toggle mask view")
show_mask, frame_i = False, 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_i += 1

    warped = cv2.warpPerspective(frame, M, (W, H))
    dets, mask = detect_notes(warped)
    assign_tracks(dets)

    for tid, t in list(tracks.items()):
        x,y,w,h = t['box']; cx, cy = t['x'], t['y']
        current_col = col_for_x(cx)

        # debounce on measured column
        prev_meas = measured_col.get(tid)
        if prev_meas == current_col:
            stable[tid] = stable.get(tid, 0) + 1
        else:
            measured_col[tid] = current_col
            stable[tid] = 1

        # stillness
        px, py = prev_pos.get(tid, (cx, cy))
        speed = ((cx - px)**2 + (cy - py)**2) ** 0.5
        prev_pos[tid] = (cx, cy)
        still[tid] = still.get(tid, 0) + 1 if speed <= SPEED_PX else 0

        # draw
        cv2.rectangle(warped, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(warped, f"id {tid} -> {current_col or '?'}  st:{stable.get(tid,0)} still:{still.get(tid,0)} spd:{speed:.1f}",
                    (x, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

        ready = stable.get(tid,0) >= STABLE_FRAMES and still.get(tid,0) >= STILL_FRAMES

        # do OCR periodically and cache best
        if frame_i % OCR_EVERY_N == 0:
            mx = max(0, x-6); my = max(0, y-6)
            roi = warped[my:my+h+12, mx:mx+w+12]
            txt, conf = ocr_text(roi)
            best_txt, best_conf = ocr_best.get(tid, ("", 0.0))
            if (txt and len(txt) >= 3) and (conf > best_conf + 0.5):
                ocr_best[tid] = (txt, conf)

        # create
        if tid not in track_to_card and current_col and ready:
            list_id = COL_TO_LIST.get(current_col)
            if list_id:
                best_txt, best_conf = ocr_best.get(tid, ("", 0.0))
                title = best_txt if (best_txt and len(best_txt) >= 3) else f"Note {int(time.time())}"
                try:
                    card = create_card(list_id, title)
                    track_to_card[tid] = card["id"]
                    committed_col[tid] = current_col
                    card_name[tid] = title
                    ocr_attempts[tid] = 0
                    print(f"[trello] create '{title}' in {current_col} -> {card.get('shortUrl','')}")
                except Exception as e:
                    print("[trello] create error:", e)

        # move
        elif tid in track_to_card and current_col and ready:
            last = committed_col.get(tid)
            if last != current_col:
                list_id = COL_TO_LIST.get(current_col)
                if list_id:
                    try:
                        move_card(track_to_card[tid], list_id)
                        committed_col[tid] = current_col
                        print(f"[trello] move card {track_to_card[tid]} -> {current_col}")
                    except Exception as e:
                        print("[trello] move error:", e)

        # try to rename within the improvement window
        if tid in track_to_card and ocr_attempts.get(tid, 0) < OCR_RENAME_WINDOW:
            ocr_attempts[tid] = ocr_attempts.get(tid, 0) + 1
            best_txt, best_conf = ocr_best.get(tid, ("", 0.0))
            curr_name = card_name.get(tid, "")
            # only rename if we have a clearly better name
            if best_txt and len(best_txt) >= 3:
                better_conf = (best_conf >= RENAME_MIN_CONF) or \
                              (best_conf - (0.0 if curr_name.startswith("Note ") else 50.0)) >= IMPROVE_DELTA
                if better_conf and best_txt != curr_name:
                    try:
                        rename_card(track_to_card[tid], best_txt)
                        card_name[tid] = best_txt
                        print(f"[trello] rename card {track_to_card[tid]} -> '{best_txt}' (conf {best_conf:.1f})")
                    except Exception as e:
                        # non-fatal
                        pass

    cv2.imshow("warped (notes + sync)", warped)
    if show_mask: cv2.imshow("mask", mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'): break
    elif k == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            try: cv2.destroyWindow("mask")
            except cv2.error: pass

cap.release(); cv2.destroyAllWindows()
