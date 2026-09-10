import base64
import io
import os
import time

import runpod
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_REPO = os.environ.get('MODEL_REPO', 'ZhengPeng7/BiRefNet').strip() or 'ZhengPeng7/BiRefNet'
MODEL_SIZE = int(os.environ.get('MODEL_SIZE', '1024') or 1024)
MODEL_SIZE = max(512, min(1536, MODEL_SIZE))
MAX_INPUT_BYTES = 6 * 1024 * 1024
MAX_OUTPUT_EDGE = 1800

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

print(f'[AI] loading model={MODEL_REPO} device={DEVICE} dtype={DTYPE}')
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


def _remove_background(image: Image.Image) -> bytes:
    tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    if DEVICE.type == 'cuda':
        tensor = tensor.half()

    with torch.inference_mode():
        pred = MODEL(tensor)[-1].sigmoid().float().cpu()[0].squeeze(0)

    mask = TO_PIL(pred).resize(image.size, Image.Resampling.LANCZOS)
    out = image.convert('RGBA')
    out.putalpha(mask)

    buf = io.BytesIO()
    out.save(buf, format='PNG', optimize=True, compress_level=6)
    return buf.getvalue()


def handler(job):
    started = time.perf_counter()
    try:
        payload = job.get('input') or {}
        image = _decode_image(payload.get('image_base64', ''), payload.get('max_output_edge', MAX_OUTPUT_EDGE))
        output = _remove_background(image)
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'status': 'success',
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
