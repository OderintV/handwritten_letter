"""Turn a photo/scan of handwriting into EMNIST-style 28x28 white-on-black letters."""
import cv2, numpy as np


def _odd(n):
    n = int(n)
    return n if n % 2 else n + 1


def binarize(gray):
    """Gray image -> binary (ink = 255). Handles uneven lighting from phone photos."""
    h, w = gray.shape
    s = 1000 / max(h, w)
    if s < 1:
        gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    # estimate paper background (remove dark ink with a big closing on a small copy)
    small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    k = _odd(max(5, 0.2 * max(small.shape)))
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 5), (w, h))
    norm = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    _, bw = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() > 127:                       # light ink on dark background
        bw = 255 - bw
    return cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def find_letter_boxes(bw, single=False):
    """Bounding boxes (x1,y1,x2,y2), left-to-right. Dots of i/j are merged with their stem."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    boxes = [[x, y, x + w, y + h] for x, y, w, h, a in stats[1:] if a >= max(30, 0.0005 * bw.size)]
    if not boxes:
        return []
    if single:
        return [[min(b[0] for b in boxes), min(b[1] for b in boxes),
                 max(b[2] for b in boxes), max(b[3] for b in boxes)]]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                ov = min(a[2], b[2]) - max(a[0], b[0])
                if ov > 0.3 * min(a[2] - a[0], b[2] - b[0]):
                    boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes.pop(j); changed = True; break
            if changed:
                break
    big = max(max(b[2] - b[0], b[3] - b[1]) for b in boxes)
    boxes = [b for b in boxes if max(b[2] - b[0], b[3] - b[1]) >= 0.2 * big]
    return sorted(boxes, key=lambda b: b[0])


def to_emnist(crop):
    """Binary letter crop -> 28x28 uint8, styled like EMNIST (square, ~2px border, thick blurred strokes)."""
    h, w = crop.shape
    s = max(h, w)
    sq = np.zeros((s, s), np.uint8)
    sq[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = crop
    dt = cv2.distanceTransform(sq, cv2.DIST_L2, 3)
    if (dt > 0).any():
        stroke = 2 * np.percentile(dt[dt > 0], 90)
        k = int(round(0.10 * s - stroke)) + 1          # thicken thin pen strokes toward EMNIST look
        if k >= 2:
            sq = cv2.dilate(sq, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    p = max(1, round(s / 12))
    sq = cv2.copyMakeBorder(sq, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(sq, (28, 28), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(out, (3, 3), 0)


def letters_from_image(path, single=False):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    bw = binarize(img)
    boxes = find_letter_boxes(bw, single)
    return [to_emnist(bw[y1:y2, x1:x2]) for x1, y1, x2, y2 in boxes], bw, boxes
