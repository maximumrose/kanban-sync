import json, time
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT / "config" / "board_layout.json").read_text())
M = np.array(CFG["warp"]["M"], dtype=np.float32)
W, H = int(CFG["warp"]["W"]), int(CFG["warp"]["H"])
COLUMNS = CFG["columns"]

def col_for_x(x):
    for c in COLUMNS:
        if c["x0"] <= x < c["x1"]:
            return c["name"]
    return None

# --- Color ranges (HSV) for common sticky colors. Tweak if needed. ---
# (H:0-179, S:0-255, V:0-255)
COLOR_RANGES = [
    # yellow
    ((15,  80, 100), (35, 255, 255)),
    # orange
    ((5,   80, 100), (15, 255, 255)),
    # green
    ((35,  60,  70), (85, 255, 255)),
    # blue
    ((90,  50,  70), (130,255, 255)),
    # pink/magenta
    ((140, 50,  70), (179,255, 255)),
    # red wraps: low and high bands
    ((0,   80, 100), (5,  255, 255)),
]

MIN_AREA_RATIO = 0.001   # ~0.1% of board
MAX_AREA_RATIO = 0.20    # up to 20% of board (ignore huge)
ASPECT_MIN, ASPECT_MAX = 0.6, 1.8  # sticky-ish
MAX_ASSIGN_DIST = 70     # px; tune for your camera
MAX_MISSES = 10          # frames to keep lost tracks

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera not found / permission issue.")

tracks = {}   # id -> dict(x,y,box,col,misses)
next_id = 1
prev_cols = {}  # id -> column

def detect_notes(img_bgr):
    """Return list of (x,y,w,h,cx,cy) for note candidates in warped board image."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES:
        mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

    # Clean up
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find blobs
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    area_tot = W * H
    for c in contours:
        area = cv2.contourArea(c)
        if area < MIN_AREA_RATIO*area_tot or area > MAX_AREA_RATIO*area_tot:
            continue
        x,y,w,h = cv2.boundingRect(c)
        ar = w / max(1.0, h)
        if not (ASPECT_MIN <= ar <= ASPECT_MAX):  # rough sticky shape
            continue
        # polygonal approximation for rectangular-ish shapes (optional)
        eps = 0.04 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        if len(approx) < 4 or not cv2.isContourConvex(approx):
            continue
        cx, cy = x + w//2, y + h//2
        out.append((x,y,w,h,cx,cy))
    return out, mask

def assign_tracks(dets):
    """Greedy nearest-centroid assignment of detections to existing tracks."""
    global next_id
    # Build cost (distance) matrix
    track_ids = list(tracks.keys())
    T = len(track_ids); D = len(dets)

    # Early outs
    assigned = {}
    used_tracks, used_dets = set(), set()

    if T and D:
        centers_t = np.array([[tracks[i]['x'], tracks[i]['y']] for i in track_ids], dtype=np.float32)
        centers_d = np.array([[d[4], d[5]] for d in dets], dtype=np.float32)
        # pairwise distances
        d2 = np.sqrt(((centers_t[:,None,:]-centers_d[None,:,:])**2).sum(axis=2))
        # Greedy: pick smallest distances iteratively
        while True:
            idx = np.unravel_index(np.argmin(d2, axis=None), d2.shape)
            i_t, i_d = int(idx[0]), int(idx[1])
            dist = d2[i_t, i_d]
            if np.isinf(dist) or dist > MAX_ASSIGN_DIST: break
            tid = track_ids[i_t]
            assigned[tid] = i_d
            used_tracks.add(i_t); used_dets.add(i_d)
            d2[i_t, :] = np.inf
            d2[:, i_d] = np.inf
            if len(used_tracks) == T or len(used_dets) == D: break

    # Update assigned tracks
    for idx_t, tid in enumerate(track_ids):
        if idx_t in used_tracks:
            i_d = assigned[tid]
            x,y,w,h,cx,cy = dets[i_d]
            tracks[tid].update(x=cx, y=cy, box=(x,y,w,h), misses=0)
        else:
            tracks[tid]['misses'] += 1

    # Create tracks for unassigned detections
    for i, d in enumerate(dets):
        if i in used_dets: continue
        x,y,w,h,cx,cy = d
        global next_id
        tid = next_id; next_id += 1
        tracks[tid] = dict(x=cx, y=cy, box=(x,y,w,h), misses=0, col=None)

    # Remove stale tracks
    gone = [tid for tid,t in tracks.items() if t['misses'] > MAX_MISSES]
    for tid in gone:
        print(f"[event] remove track {tid}")
        tracks.pop(tid, None)
        prev_cols.pop(tid, None)

print("Controls: q=quit, m=toggle mask view")
show_mask = False

while True:
    ok, frame = cap.read()
    if not ok: break

    warped = cv2.warpPerspective(frame, M, (W, H))
    dets, mask = detect_notes(warped)

    assign_tracks(dets)

    # Events: add/move based on column changes
    for tid, t in tracks.items():
        col = col_for_x(t['x'])
        if tid not in prev_cols:
            prev_cols[tid] = col
            if col:
                print(f"[event] add track {tid} -> {col}")
        elif prev_cols[tid] != col:
            print(f"[event] move track {tid} -> {col}")
            prev_cols[tid] = col
        # draw
        x,y,w,h = t['box']
        cv2.rectangle(warped, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.circle(warped, (int(t['x']), int(t['y'])), 4, (0,255,0), -1)
        cv2.putText(warped, f"id {tid} -> {col or '?'}", (x, max(15,y-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("warped (notes)", warped)
    if show_mask:
        cv2.imshow("mask", mask)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
    elif k == ord('m'):
        show_mask = not show_mask
        if not show_mask:
            try:
                cv2.destroyWindow("mask")
            except cv2.error:
                pass

cap.release(); cv2.destroyAllWindows()
