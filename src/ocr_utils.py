from pathlib import Path
import cv2, pytesseract, numpy as np, re, os

# Point to the Windows exe if needed
TESS = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESS):
    pytesseract.pytesseract.tesseract_cmd = TESS

def preprocess_for_ocr(roi_bgr):
    # grayscale
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    # illumination/background correction
    bg = cv2.medianBlur(gray, 21)
    norm = cv2.divide(gray, bg, scale=255)
    # contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    norm = clahe.apply(norm)
    # binarize (black text on white)
    _, binary = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure background is white; text black
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    # thicken strokes slightly
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8), 1)
    return binary

def ocr_text(roi_bgr):
    img = preprocess_for_ocr(roi_bgr)
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6 -l eng"
    )
    words, confs = [], []
    for txt, conf in zip(data["text"], data["conf"]):
        try:
            c = float(conf)
        except:
            continue
        txt = (txt or "").strip()
        if c >= 35 and txt:
            words.append(txt)
            confs.append(c)
    if not words:
        return None, 0.0
    text = " ".join(words)
    text = re.sub(r"[^A-Za-z0-9 .,#:/()\-]+", "", text).strip()
    text = text[:120] if text else None
    avg_conf = float(np.mean(confs)) if confs else 0.0
    return (text if text else None), avg_conf
