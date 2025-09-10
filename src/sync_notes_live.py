# Sticky-note detection -> Trello sync with identity + grace archiving
# - 3 columns as calibrated
# - Create only after stable+still AND uniqueness check passes
# - Move when a known sticky changes columns (stable+still)
# - Recover from "missing" (no new card)
# - Archive only after 60 seconds missing
# - OCR titles + auto-rename window retained

import json, time, sys, math, difflib
from pathlib import Path
import numpy as np
import cv2

from trello_api import create_card, move_card, archive_card, rename_card
from ocr_utils import ocr_text

# ---------- Load config ----------
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

# ---------- Detection params ----------
# HSV ranges (H:0..179, S/V:0..255)
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

# ---------- Tracker / Association thresholds ----------
ASSIGN_DIST_MAX    = 180.0  # px (distance gate after prediction)
HIST_INTERSECT_MIN = 0.30   # appearance gate 0..1 for matching
ALPHA_SIZE         = 0.2    # EMA for size smoothing
ALPHA_HIST         = 0.3    # EMA for histogram update

# ---------- Event gating ----------
STABLE_FRAMES = 6           # same column N frames
STILL_FRAMES  = 8           # low motion N frames
SPEED_PX      = 2.0         # px/frame considered still
BIRTH_DELAY   = 12          # frames before a candidate may "create"
MISSING_TTL_SEC = 60.0      # archive only after this many seconds missing

# ---------- OCR / rename ----------
OCR_EVERY_N       = 3
OCR_RENAME_WINDOW = 30
RENAME_MIN_CONF   = 60.0
IMPROVE_DELTA     = 8.0

# ---------- Helpers ----------
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

def hs_hist(img_bgr, box):
    x,y,w,h = box
    x = max(0, x); y = max(0, y)
    w = max(1, min(w, img_bgr.shape[1]-x))
    h = max(1, min(h, img_bgr.shape[0]-y))
    roi = img_bgr[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[30,32],[0,180,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist

def hist_intersection(h1, h2):
    if h1 is None or h2 is None: return 0.0
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_INTERSECT))  # 0..1 with norm

def clamp_box(box):
    x,y,w,h = box
    x = int(max(0, min(x, W-1)))
    y = int(max(0, min(y, H-1)))
    w = int(max(1, min(w, W-x)))
    h = int(max(1, min(h, H-y)))
    return (x,y,w,h)

def box_center(box):
    x,y,w,h = box
    return (x + w/2.0, y + h/2.0)

def diag_len(box):
    return math.hypot(box[2], box[3])

def spatial_affinity(dist, ref_diag, scale=1.5):
    # map distance to [0..1]: 1 at 0 distance, 0 at >= scale*diag
    thr = max(1.0, scale * max(8.0, ref_diag))
    s = max(0.0, 1.0 - float(dist)/thr)
    return s

def text_similarity(a, b):
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------- Track model (with FSM state) ----------
class Track:
    # state ∈ {"CANDIDATE","ACTIVE","MISSING"}
    def __init__(self, tid, box, img_for_hist):
        self.id   = tid
        self.box  = clamp_box(box)
        cx, cy    = box_center(self.box)

        # Kalman (x,y,vx,vy) -> (x,y)
        self.kf = cv2.KalmanFilter(4,2)
        dt = 1.0
        self.kf.transitionMatrix = np.array([[1,0,dt,0],
                                             [0,1,0,dt],
                                             [0,0,1,0],
                                             [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0],
                                              [0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.array([[1,0,0,0],
                                            [0,1,0,0],
                                            [0,0,5,0],
                                            [0,0,0,5]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.6
        self.kf.statePost = np.array([[cx],[cy],[0],[0]], np.float32)
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.state = "CANDIDATE"
        self.birth_frames = 0
        self.stable = 0
        self.still = 0
        self.measured_col = None
        self.prev_pos = (cx, cy)

        self.hist = hs_hist(img_for_hist, self.box)
        self.best_text = ""
        self.best_conf = 0.0

        self.card_id = None
        self.card_name = None
        self.committed_col = None

        now = time.monotonic()
        self.last_seen = now
        self.missing_since = None
        self.ocr_attempts = 0

    def predict(self):
        p = self.kf.predict()
        cx, cy = float(p[0]), float(p[1])
        x = int(cx - self.box[2]/2)
        y = int(cy - self.box[3]/2)
        self.box = clamp_box((x,y,self.box[2],self.box[3]))
        return cx, cy

    def correct(self, det_box, img_for_hist):
        x,y,w,h = det_box
        cx, cy = x + w/2.0, y + h/2.0
        self.kf.correct(np.array([[np.float32(cx)], [np.float32(cy)]]))
        # smooth size
        bx,by,bw,bh = self.box
        bw = int((1-ALPHA_SIZE)*bw + ALPHA_SIZE*w)
        bh = int((1-ALPHA_SIZE)*bh + ALPHA_SIZE*h)
        self.box = clamp_box((int(cx - bw/2), int(cy - bh/2), bw, bh))
        # update appearance
        new_hist = hs_hist(img_for_hist, (int(x),int(y),int(w),int(h)))
        if new_hist is not None:
            if self.hist is None:
                self.hist = new_hist
            else:
                self.hist = (1-ALPHA_HIST)*self.hist + ALPHA_HIST*new_hist
        # seen now
        self.last_seen = time.monotonic()
        self.missing_since = None

# ---------- Registry ----------
tracks = {}      # tid -> Track
next_id = 1

# ---------- Matching / Uniqueness ----------
def associate_existing(detections, frame_img):
    """Primary association for already-known tracks (ACTIVE or CANDIDATE).
       Returns: assigned map {tid: det_index}, sets of unassigned tracks and dets."""
    # Predict all tracks
    for tr in tracks.values():
        tr.predict()

    # Candidate pairs with cost using distance + hist
    cand = []
    det_boxes = [(int(x),int(y),int(w),int(h)) for (x,y,w,h,_,_) in detections]
    det_centers = [(cx,cy) for (*_,cx,cy) in detections]

    tids = list(tracks.keys())
    for tid in tids:
        tr = tracks[tid]
        if tr.state not in ("ACTIVE","CANDIDATE"):
            continue
        tcx, tcy = box_center(tr.box)
        ref_diag = diag_len(tr.box)
        for j, (dbox, (cx,cy)) in enumerate(zip(det_boxes, det_centers)):
            dist = math.hypot(cx - tcx, cy - tcy)
            if dist > ASSIGN_DIST_MAX:  # distance gate
                continue
            hsim = hist_intersection(tr.hist, hs_hist(frame_img, dbox))
            if hsim < HIST_INTERSECT_MIN:  # appearance gate
                continue
            # cost: lower is better
            dist_norm = dist / max(8.0, ref_diag)
            cost = 0.7*dist_norm + 0.3*(1.0 - hsim)
            cand.append((cost, tid, j, dbox))

    cand.sort(key=lambda x: x[0])
    assigned_tr, assigned_det = set(), set()
    assigned = {}
    for cost, tid, j, dbox in cand:
        if tid in assigned_tr or j in assigned_det:
            continue
        tracks[tid].correct(dbox, frame_img)
        assigned[tid] = j
        assigned_tr.add(tid); assigned_det.add(j)

    # Unassigned sets
    unassigned_tracks = {tid for tid in tracks.keys() if tid not in assigned and tracks[tid].state in ("ACTIVE","CANDIDATE")}
    unassigned_dets = {j for j in range(len(detections)) if j not in assigned_det}
    return assigned, unassigned_tracks, unassigned_dets

def uniqueness_check_for_new_det(j, detections, frame_img):
    """Try to bind unmatched detection j to MISSING first (recover), then ACTIVE duplicate guard."""
    x,y,w,h,cx,cy = detections[j]
    dbox = (int(x),int(y),int(w),int(h))
    dcenter = (cx, cy)
    dhist = hs_hist(frame_img, dbox)

    # 1) try MISSING (recover)
    best_t, best_score = None, -1.0
    now = time.monotonic()
    for tid, tr in tracks.items():
        if tr.state != "MISSING":
            continue
        # prefer recent miss
        if tr.missing_since is None or now - tr.missing_since > MISSING_TTL_SEC:
            continue
        tcx, tcy = box_center(tr.box)  # predicted position already applied
        dist = math.hypot(dcenter[0]-tcx, dcenter[1]-tcy)
        sp = spatial_affinity(dist, diag_len(tr.box), scale=1.5)  # 0..1
        ap = hist_intersection(tr.hist, dhist)                    # 0..1
        if ap < 0.25:
            continue
        score = 0.6*ap + 0.4*sp
        if score > best_score:
            best_score, best_t = score, tid

    if best_t is not None and best_score >= 0.55:
        # Recover this missing track
        tracks[best_t].correct(dbox, frame_img)
        tracks[best_t].state = "ACTIVE"
        return ("RECOVER", best_t)

    # 2) ACTIVE duplicate guard (reject double-detections)
    best_t, best_score = None, -1.0
    for tid, tr in tracks.items():
        if tr.state != "ACTIVE":
            continue
        tcx, tcy = box_center(tr.box)
        dist = math.hypot(dcenter[0]-tcx, dcenter[1]-tcy)
        sp = spatial_affinity(dist, diag_len(tr.box), scale=0.8)
        ap = hist_intersection(tr.hist, dhist)
        score = 0.6*ap + 0.4*sp
        if score > best_score:
            best_score, best_t = score, tid

    if best_t is not None and best_score >= 0.70:
        # Treat as duplicate sighting of an ACTIVE track; ignore creation
        return ("DUPLICATE_ACTIVE", best_t)

    # 3) No match → new candidate
    return ("NEW", None)

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened(): die("Camera not found / permission issue.")
print("Controls: q=quit, m=toggle mask view")
show_mask, frame_i = False, 0

while True:
    ok, frame = cap.read()
    if not ok: break
    frame_i += 1

    warped = cv2.warpPerspective(frame, M, (W, H))
    detections, mask = detect_notes(warped)

    # 1) Primary association (ACTIVE/CANDIDATE)
    assigned, un_tracks, un_dets = associate_existing(detections, warped)

    # 2) Mark unassigned tracks as MISSING (start timer, do not archive yet)
    now = time.monotonic()
    for tid in list(un_tracks):
        tr = tracks[tid]
        if tr.state != "MISSING":
            tr.state = "MISSING"
            if tr.missing_since is None:
                tr.missing_since = now

    # 3) Uniqueness check for each unassigned detection
    for j in list(un_dets):
        verdict, ref_tid = uniqueness_check_for_new_det(j, detections, warped)
        if verdict == "RECOVER":
            # Already corrected inside uniqueness; track set ACTIVE
            pass
        elif verdict == "DUPLICATE_ACTIVE":
            # Ignore this detection
            pass
        else:
            # NEW → create a CANDIDATE track
            x,y,w,h,_,_ = detections[j]
            tid = next_id; next_id += 1
            tracks[tid] = Track(tid, (int(x),int(y),int(w),int(h)), warped)
            # Leave as CANDIDATE; promotion after gating below

    # 4) Per-track FSM + Trello sync
    for tid, tr in list(tracks.items()):
        # Update birth frames
        tr.birth_frames += 1

        # Column + gating counters
        x,y,w,h = tr.box
        cx, cy = box_center(tr.box)
        current_col = col_for_x(cx)

        # Debounce on measured column
        if tr.measured_col == current_col:
            tr.stable += 1
        else:
            tr.measured_col = current_col
            tr.stable = 1

        # Stillness
        px, py = tr.prev_pos
        speed = math.hypot(cx - px, cy - py)
        tr.prev_pos = (cx, cy)
        tr.still = tr.still + 1 if speed <= SPEED_PX else 0

        # State transitions
        if tr.state == "CANDIDATE":
            # Promote to ACTIVE only after birth delay + stable + still
            if tr.birth_frames >= BIRTH_DELAY and tr.stable >= STABLE_FRAMES and tr.still >= STILL_FRAMES and current_col:
                tr.state = "ACTIVE"
                tr.committed_col = current_col
                # Create Trello card now
                list_id = COL_TO_LIST.get(current_col)
                if list_id:
                    # try OCR title quickly (best so far)
                    mx = max(0, x-6); my = max(0, y-6)
                    roi = warped[my:my+h+12, mx:mx+w+12]
                    txt, conf = ocr_text(roi)
                    if txt and conf > tr.best_conf:
                        tr.best_text, tr.best_conf = txt, conf
                    title = tr.best_text if (tr.best_text and len(tr.best_text) >= 3) else f"Note {int(time.time())}"
                    try:
                        card = create_card(list_id, title)
                        tr.card_id = card["id"]
                        tr.card_name = title
                        tr.ocr_attempts = 0
                        print(f"[trello] create '{title}' in {current_col} -> {card.get('shortUrl','')}")
                    except Exception as e:
                        print("[trello] create error:", e)

        elif tr.state == "ACTIVE":
            # Move if column changed and gates satisfied
            if current_col and tr.committed_col and current_col != tr.committed_col and tr.stable >= STABLE_FRAMES and tr.still >= STILL_FRAMES:
                list_id = COL_TO_LIST.get(current_col)
                if list_id and tr.card_id:
                    try:
                        move_card(tr.card_id, list_id)
                        tr.committed_col = current_col
                        print(f"[trello] move card {tr.card_id} -> {current_col}")
                    except Exception as e:
                        print("[trello] move error:", e)

        elif tr.state == "MISSING":
            # Nothing to do here; archiving handled by TTL below
            pass

        # OCR periodic + rename window
        if tr.card_id and tr.ocr_attempts < OCR_RENAME_WINDOW and frame_i % OCR_EVERY_N == 0:
            mx = max(0, x-6); my = max(0, y-6)
            roi = warped[my:my+h+12, mx:mx+w+12]
            txt, conf = ocr_text(roi)
            if txt and conf > tr.best_conf + 0.5:
                tr.best_text, tr.best_conf = txt, conf
            # rename if clearly better
            if tr.best_text and len(tr.best_text) >= 3:
                better_conf = (tr.best_conf >= RENAME_MIN_CONF) or \
                              ((tr.best_conf - (0.0 if (tr.card_name or '').startswith("Note ") else 50.0)) >= IMPROVE_DELTA)
                if better_conf and tr.best_text != (tr.card_name or ""):
                    try:
                        rename_card(tr.card_id, tr.best_text)
                        tr.card_name = tr.best_text
                        print(f"[trello] rename card {tr.card_id} -> '{tr.best_text}' (conf {tr.best_conf:.1f})")
                    except Exception:
                        pass
            tr.ocr_attempts += 1

        # Draw overlay with state and timers
        state_tag = tr.state
        if tr.state == "MISSING" and tr.missing_since is not None:
            state_tag = f"MISSING {int(time.monotonic()-tr.missing_since)}s"
        cv2.rectangle(warped, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(warped, f"id {tid} {state_tag} -> {current_col or '?'} st:{tr.stable} still:{tr.still} spd:{speed:.1f}",
                    (x, max(15,y-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,0), 2)

    # 5) Time-based archiving for long-missing tracks
    now = time.monotonic()
    for tid in list(tracks.keys()):
        tr = tracks[tid]
        if tr.state == "MISSING" and tr.missing_since is not None:
            if (now - tr.missing_since) >= MISSING_TTL_SEC:
                if tr.card_id:
                    try:
                        archive_card(tr.card_id)
                        print(f"[trello] archived {tr.card_id} after {MISSING_TTL_SEC:.0f}s missing (track {tid})")
                    except Exception as e:
                        print("[trello] archive error:", e)
                del tracks[tid]

    # 6) Windows
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
