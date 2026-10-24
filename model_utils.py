"""Shared MNIST preprocessing, inference and multi-digit segmentation.
No TensorFlow dependency here: import works in unit tests without a model.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

IMAGE_SIZE = 28


def grayscale(image: Image.Image) -> np.ndarray:
    """Composite transparency over white and return uint8 grayscale."""
    rgba = np.asarray(image.convert('RGBA'), dtype=np.float32)
    alpha = rgba[..., 3:4] / 255.0
    rgb = rgba[..., :3] * alpha + 255.0 * (1.0 - alpha)
    return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)


def foreground(gray: np.ndarray, polarity: str = 'Auto', offset: int = 0):
    """Return uint8 grayscale digit intensity and a binary positioning mask.

    Uses original grayscale intensity, not binary pixels, for prediction.
    """
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError('Expected a nonempty grayscale image')
    gray = np.asarray(gray, np.uint8)
    if polarity == 'Auto':
        edge = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
        dark_ink = bool(np.median(edge) >= 127)
    elif polarity == 'Dark ink on light background':
        dark_ink = True
    else:
        dark_ink = False
    ink = (255 - gray) if dark_ink else gray.copy()
    # Estimate a uniform background, preserving antialiased strokes.
    edge_ink = np.concatenate((ink[0], ink[-1], ink[:, 0], ink[:, -1]))
    baseline = float(np.median(edge_ink))
    ink = np.clip((ink.astype(np.float32) - baseline) * (255.0 / max(1.0, 255.0 - baseline)), 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = max(12, min(245, int(otsu) + offset))
    mask = (ink > cutoff).astype(np.uint8)
    return ink, mask


def extract_boxes(mask: np.ndarray, min_area: int = 5, gap: int = 2):
    """Read left-to-right separate digits using ink occupied columns.

    Connected/touching digits may need manually supplied cuts.
    """
    active = np.any(mask > 0, axis=0)
    begins = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False]) + 1
    groups = []
    for a, b in zip(begins, ends):
        if groups and a - groups[-1][1] <= gap:
            groups[-1] = (groups[-1][0], int(b))
        else:
            groups.append((int(a), int(b)))
    return boxes_from_bands(mask, groups, min_area)


def boxes_from_bands(mask, bands, min_area=5):
    boxes = []
    height, width = mask.shape
    for left, right in bands:
        left, right = max(0, int(left)), min(width, int(right))
        if left >= right:
            continue
        ys, xs = np.nonzero(mask[:, left:right])
        if len(xs) < min_area:
            continue
        boxes.append((left + int(xs.min()), int(ys.min()), left + int(xs.max()) + 1, int(ys.max()) + 1))
    return boxes


def boxes_from_cuts(mask, cuts: str, min_area=5):
    width = mask.shape[1]
    try:
        positions = sorted(set(int(part.strip()) for part in cuts.split(',') if part.strip()))
    except ValueError as exc:
        raise ValueError('Cut positions must be comma-separated whole numbers, e.g. 48,95') from exc
    if any(n <= 0 or n >= width for n in positions):
        raise ValueError(f'Cut positions must be between 1 and {width - 1}')
    return boxes_from_bands(mask, list(zip([0] + positions, positions + [width])), min_area)


def normalise_digit(ink: np.ndarray, box, canvas: int = IMAGE_SIZE):
    """MNIST-style: preserve grayscale, crop, resize longest dimension to 20, centre mass on 28x28."""
    x1, y1, x2, y2 = box
    crop = np.asarray(ink[y1:y2, x1:x2], dtype=np.uint8)
    if crop.size == 0 or crop.max() == 0:
        raise ValueError('Empty digit crop')
    # Remove near-zero noise but retain continuous greyscale at stroke edges.
    ys, xs = np.nonzero(crop > max(6, int(crop.max() * .06)))
    if len(xs) == 0:
        raise ValueError('No digit pixels detected')
    crop = crop[ys.min():ys.max()+1, xs.min():xs.max()+1]
    h, w = crop.shape
    scale = 20.0 / max(w, h)
    out_w, out_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas_arr = np.zeros((canvas, canvas), dtype=np.uint8)
    ox, oy = (canvas - out_w) // 2, (canvas - out_h) // 2
    canvas_arr[oy:oy+out_h, ox:ox+out_w] = resized
    moments = cv2.moments(canvas_arr)
    if moments['m00'] > 0:
        cx = moments['m10']/moments['m00']
        cy = moments['m01']/moments['m00']
        transform = np.float32([[1, 0, (canvas-1)/2 - cx], [0, 1, (canvas-1)/2 - cy]])
        canvas_arr = cv2.warpAffine(canvas_arr, transform, (canvas, canvas), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return canvas_arr


def to_model_batch(images):
    """New CNN expects [N,28,28,1] float32 in [0,1]."""
    a = np.asarray(images, dtype=np.float32)
    if a.ndim == 2:
        a = a[None]
    if a.ndim != 3 or a.shape[1:] != (28, 28):
        raise ValueError('Expected one or more 28x28 digit images')
    return a[..., None] / 255.0


def predict_digits(model, images):
    raw = np.asarray(model.predict(to_model_batch(images), verbose=0))
    if raw.ndim != 2 or raw.shape[1] != 10 or not np.isfinite(raw).all():
        raise ValueError('Model must produce 10 finite class scores per digit')
    if np.any(raw < 0) or np.any(raw > 1) or not np.allclose(raw.sum(axis=1), 1, atol=1e-2):
        x = raw - raw.max(axis=1, keepdims=True)
        exp = np.exp(x)
        raw = exp / exp.sum(axis=1, keepdims=True)
    return raw
