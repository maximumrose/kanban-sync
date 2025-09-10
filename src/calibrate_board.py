import cv2, json, os, glob
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPDIR = ROOT / "captures"
OUTDIR = ROOT / "config"; OUTDIR.mkdir(exist_ok=True)

def latest_image():
    imgs = sorted(glob.glob(str(CAPDIR / "snap_*.jpg")))
    if not imgs: raise SystemExit(f"No images in {CAPDIR}. Run cam_preview and save some.")
    return imgs[-1]

def click_points(img, n, msg):
    pts = []
    vis = img.copy()
    def on_mouse(e,x,y,flags,ud):
        nonlocal pts, vis
        if e == cv2.EVENT_LBUTTONDOWN:
            pts.append((x,y))
        elif e == cv2.EVENT_RBUTTONDOWN and pts:  # undo
            pts.pop()
        vis = img.copy()
        for i,(px,py) in enumerate(pts):
            cv2.circle(vis,(px,py),5,(0,255,0),-1)
            cv2.putText(vis,str(i+1),(px+6,py-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.putText(vis,f"{msg}  (LMB=add, RMB=undo, Enter=confirm)",(10,30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    win = "calibrate"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win,on_mouse)
    while True:
        cv2.imshow(win, vis)
        k = cv2.waitKey(20) & 0xFF
        if k in (13,10):  # Enter
            if len(pts) != n: continue
            cv2.destroyWindow(win); return np.array(pts, dtype=np.float32)
        elif k == 27:  # Esc
            cv2.destroyWindow(win); raise SystemExit("Cancelled.")

def warp_to_rect(img, corners, W=1200):
    # corners order: TL, TR, BR, BL
    TL, TR, BR, BL = corners
    # estimate height preserving aspect
    widthA = np.linalg.norm(BR-BL); widthB = np.linalg.norm(TR-TL)
    heightA = np.linalg.norm(TR-BR); heightB = np.linalg.norm(TL-BL)
    H = int(max(heightA, heightB) * (W / max(widthA, widthB)))
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(img, M, (W, H))
    return warped, M, W, H

def main():
    path = latest_image()
    img = cv2.imread(path)
    if img is None: raise SystemExit("Failed to read image.")
    # 1) Click 4 corners: TL, TR, BR, BL
    corners = click_points(img, 4, "Click board corners: TL, TR, BR, BL")
    warped, M, W, H = warp_to_rect(img, corners, W=1200)

    # 2) Choose columns: enter count in console
    try:
        ncols = int(input("Number of columns (lists): ").strip())
        assert ncols >= 1
    except Exception:
        raise SystemExit("Invalid number of columns.")

    # 3) Click vertical boundaries inside warped view (left→right), Ncols-1 clicks
    msg = f"Click {ncols-1} vertical boundaries (left→right). Edges 0 and {W} are implied."
    boundaries = []
    if ncols > 1:
        bpts = click_points(warped, ncols-1, msg)
        boundaries = sorted(int(x) for x,_ in bpts)
    xs = [0] + boundaries + [W]

    # build columns as [x0,x1]
    columns = [{"name": f"Col {i+1}", "x0": int(xs[i]), "x1": int(xs[i+1])} for i in range(ncols)]

    # 4) Save config
    cfg = {
        "warp": {"M": M.tolist(), "W": W, "H": H},
        "corners_order": ["TL","TR","BR","BL"],
        "columns": columns,
        "source_image": os.path.basename(path)
    }
    (OUTDIR / "board_layout.json").write_text(json.dumps(cfg, indent=2))

    # 5) Save preview with overlay
    overlay = warped.copy()
    for i,c in enumerate(columns):
        cv2.rectangle(overlay, (c["x0"], 0), (c["x1"], H-1), (0,255,0), 2)
        cx = (c["x0"]+c["x1"])//2
        cv2.putText(overlay, c["name"], (max(5,cx-60), 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    prev_path = str(OUTDIR / "calibration_preview.jpg")
    cv2.imwrite(prev_path, overlay)
    print(f"Saved config → {OUTDIR/'board_layout.json'}")
    print(f"Saved preview → {prev_path}")

if __name__ == "__main__":
    main()
