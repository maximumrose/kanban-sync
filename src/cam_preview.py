import cv2, time
from pathlib import Path

CAM_INDEX = 0  # try 1 or 2 if needed
out_dir = Path("captures"); out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened(): raise SystemExit("Camera not found. Try CAM_INDEX=1 or Windows camera permissions.")

print("Controls: [s]=save frame, [q]=quit")
while True:
    ok, frame = cap.read()
    if not ok: break
    cv2.putText(frame, "s: save  q: quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.imshow("kanban camera", frame)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"snap_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        print("Saved", path)
    elif k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
