import base64
import io
import os
import time

import cv2
import numpy as np
import runpod
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_REPO = os.environ.get('MODEL_REPO', 'ZhengPeng7/BiRefNet').strip() or 'ZhengPeng7/BiRefNet'
MODEL_SIZE = int(os.environ.get('MODEL_SIZE', '1024') or 1024)
MODEL_SIZE = max(512, min(1536, MODEL_SIZE))
MAX_INPUT_BYTES = 6 * 1024 * 1024
MAX_OUTPUT_EDGE = 1800

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

print(f'[AI] loading background model={MODEL_REPO} device={DEVICE} dtype={DTYPE}')
MODEL = AutoModelForImageSegmentation.from_pretrained(
    MODEL_REPO,
    trust_remote_code=True,
)
MODEL.to(DEVICE)
MODEL.eval()
if DEVICE.type == 'cuda':
    MODEL.half()

PREPROCESS = transforms.Compose([
    transforms.Resize((MODEL_SIZE, MODEL_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
TO_PIL = transforms.ToPILImage()
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def _decode_image(encoded: str, max_output_edge: int = MAX_OUTPUT_EDGE) -> Image.Image:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError('missing image_base64')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError('invalid image_base64') from exc
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError('image must be between 1 byte and 6 MB')

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError('unsupported or broken image') from exc

    image = ImageOps.exif_transpose(image).convert('RGB')
    max_output_edge = max(512, min(MAX_OUTPUT_EDGE, int(max_output_edge or MAX_OUTPUT_EDGE)))
    if max(image.size) > max_output_edge:
        ratio = max_output_edge / max(image.size)
        image = image.resize(
            (max(1, round(image.width * ratio)), max(1, round(image.height * ratio))),
            Image.Resampling.LANCZOS,
        )
    return image


def _foreground_mask(image: Image.Image) -> Image.Image:
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == 'cuda':
        tensor = tensor.half()
    with torch.inference_mode():
        pred = MODEL(tensor)[-1].sigmoid().float().cpu()[0].squeeze(0)
    return TO_PIL(pred).resize(image.size, Image.Resampling.LANCZOS)


def _encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format='PNG', optimize=True, compress_level=6)
    return buf.getvalue()


def _remove_background(image: Image.Image) -> bytes:
    mask = _foreground_mask(image)
    out = image.convert('RGBA')
    out.putalpha(mask)
    return _encode_png(out)


def _find_face(image: Image.Image):
    rgb = np.asarray(image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    min_side = max(36, int(min(image.size) * 0.08))
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(min_side, min_side))
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: int(f[2]) * int(f[3]))


def _alpha_bbox(alpha: np.ndarray):
    ys, xs = np.where(alpha > 28)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _soft_head_gate(width, height, cx, cy, rx, ry):
    yy, xx = np.mgrid[0:height, 0:width]
    dx = np.abs((xx - cx) / max(1.0, rx))
    dy = np.abs((yy - cy) / max(1.0, ry))
    # Superellipse keeps hair/ears better than a strict oval while trimming shoulders.
    d = np.power(dx, 4.0) + np.power(dy, 4.0)
    gate = np.clip((1.10 - d) / 0.14, 0.0, 1.0)
    gate_img = Image.fromarray(np.uint8(gate * 255), 'L').filter(ImageFilter.GaussianBlur(radius=max(1.2, min(width, height) * 0.006)))
    return np.asarray(gate_img, dtype=np.float32) / 255.0


def _head_cutout(image: Image.Image) -> bytes:
    mask = _foreground_mask(image)
    alpha = np.asarray(mask, dtype=np.float32)
    face = _find_face(image)
    w, h = image.size

    if face is not None:
        x, y, fw, fh = [float(v) for v in face]
        cx = x + fw * 0.5
        cy = y + fh * 0.36
        rx = fw * 1.12
        ry = fh * 1.18
        left = max(0, int(cx - fw * 1.18))
        right = min(w, int(cx + fw * 1.18))
        top = max(0, int(y - fh * 0.78))
        bottom = min(h, int(y + fh * 1.58))
        mode = 'face-detected'
    else:
        bbox = _alpha_bbox(alpha)
        if not bbox:
            raise ValueError('no foreground detected')
        bx0, by0, bx1, by1 = bbox
        bw, bh = bx1 - bx0, by1 - by0
        cx = (bx0 + bx1) * 0.5
        cy = by0 + bh * 0.28
        rx = max(32.0, bw * 0.55)
        ry = max(32.0, bh * 0.34)
        left = max(0, int(cx - bw * 0.58))
        right = min(w, int(cx + bw * 0.58))
        top = max(0, int(by0 - bh * 0.03))
        bottom = min(h, int(by0 + bh * 0.62))
        mode = 'foreground-fallback'

    gate = _soft_head_gate(w, h, cx, cy, rx, ry)
    combined = np.uint8(np.clip((alpha / 255.0) * gate, 0.0, 1.0) * 255)
    out = image.convert('RGBA')
    out.putalpha(Image.fromarray(combined, 'L'))
    out = out.crop((left, top, right, bottom))

    crop_alpha = np.asarray(out.getchannel('A'))
    bbox2 = _alpha_bbox(crop_alpha)
    if bbox2:
        x0, y0, x1, y1 = bbox2
        pad = max(10, int(max(x1 - x0, y1 - y0) * 0.055))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(out.width, x1 + pad); y1 = min(out.height, y1 + pad)
        out = out.crop((x0, y0, x1, y1))

    print(f'[AI] head cutout mode={mode} size={image.size} out={out.size}', flush=True)
    return _encode_png(out)


def handler(job):
    started = time.perf_counter()
    try:
        payload = job.get('input') or {}
        task = str(payload.get('task') or payload.get('action') or 'remove_background').strip().lower()
        if task in ('outpaint', 'expand', 'expand_image'):
            return {
                'status': 'error',
                'code': 'OUTPAINT_DISABLED',
                'error': 'outpaint is disabled',
                'model': MODEL_REPO,
            }

        image = _decode_image(payload.get('image_base64', ''), payload.get('max_output_edge', MAX_OUTPUT_EDGE))
        if task in ('head_cutout', 'head', 'portrait_head'):
            output = _head_cutout(image)
            result_task = 'head_cutout'
        else:
            output = _remove_background(image)
            result_task = 'remove_background'
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'status': 'success',
            'task': result_task,
            'model': MODEL_REPO,
            'width': image.width,
            'height': image.height,
            'execution_ms': elapsed_ms,
            'image_base64': base64.b64encode(output).decode('ascii'),
        }
    except Exception as exc:
        print('[AI] job failed:', repr(exc))
        return {
            'status': 'error',
            'error': str(exc),
            'model': MODEL_REPO,
        }


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
