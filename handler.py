import base64
import io
import os
import time

import runpod
import torch
from PIL import Image, ImageOps, ImageFilter
from diffusers import StableDiffusionInpaintPipeline

MODEL_REPO = os.environ.get('OUTPAINT_MODEL_REPO', 'runwayml/stable-diffusion-inpainting').strip() or 'runwayml/stable-diffusion-inpainting'
MAX_INPUT_BYTES = 8 * 1024 * 1024
MAX_EDGE = max(512, min(1024, int(os.environ.get('OUTPAINT_MAX_EDGE', '768') or 768)))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DTYPE = torch.float16 if DEVICE.type == 'cuda' else torch.float32

print(f'[OUTPAINT] loading model={MODEL_REPO} device={DEVICE} dtype={DTYPE}')
_kwargs = {'torch_dtype': DTYPE}
if DEVICE.type == 'cuda':
    _kwargs['variant'] = 'fp16'
try:
    PIPE = StableDiffusionInpaintPipeline.from_pretrained(MODEL_REPO, **_kwargs)
except Exception:
    _kwargs.pop('variant', None)
    PIPE = StableDiffusionInpaintPipeline.from_pretrained(MODEL_REPO, **_kwargs)
PIPE = PIPE.to(DEVICE)
try:
    PIPE.enable_attention_slicing()
except Exception:
    pass
print('[OUTPAINT] model ready')


def _decode(encoded: str) -> Image.Image:
    if not isinstance(encoded, str) or not encoded:
        raise ValueError('missing image_base64')
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise ValueError('invalid image_base64') from exc
    if not raw or len(raw) > MAX_INPUT_BYTES:
        raise ValueError('image must be between 1 byte and 8 MB')
    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise ValueError('unsupported or broken image') from exc
    return ImageOps.exif_transpose(image).convert('RGB')


def _round8(value):
    return max(64, int(round(value / 8.0) * 8))


def _geometry(image, direction, ratio):
    ow, oh = image.size
    if direction in ('left', 'right'):
        nw, nh = int(ow * ratio), oh
    elif direction in ('top', 'up', 'bottom', 'down'):
        nw, nh = ow, int(oh * ratio)
    else:
        nw, nh = int(ow * ratio), int(oh * ratio)

    scale = min(1.0, MAX_EDGE / max(nw, nh))
    if scale < 1:
        image = image.resize(
            (max(64, int(ow * scale)), max(64, int(oh * scale))),
            Image.Resampling.LANCZOS,
        )
        ow, oh = image.size
        if direction in ('left', 'right'):
            nw, nh = int(ow * ratio), oh
        elif direction in ('top', 'up', 'bottom', 'down'):
            nw, nh = ow, int(oh * ratio)
        else:
            nw, nh = int(ow * ratio), int(oh * ratio)

    nw, nh = _round8(nw), _round8(nh)
    if direction == 'left':
        x, y = nw - ow, (nh - oh) // 2
    elif direction == 'right':
        x, y = 0, (nh - oh) // 2
    elif direction in ('top', 'up'):
        x, y = (nw - ow) // 2, nh - oh
    elif direction in ('bottom', 'down'):
        x, y = (nw - ow) // 2, 0
    else:
        x, y = (nw - ow) // 2, (nh - oh) // 2
    return image, nw, nh, x, y


def _outpaint(image: Image.Image, payload: dict):
    direction = str(payload.get('direction') or 'all').lower()
    if direction not in ('all', 'left', 'right', 'top', 'up', 'bottom', 'down'):
        direction = 'all'
    try:
        ratio = float(payload.get('expand_ratio') or 1.4)
    except Exception:
        ratio = 1.4
    ratio = max(1.10, min(1.65, ratio))

    prompt = str(payload.get('prompt') or '').strip()
    if not prompt:
        prompt = 'seamlessly extend the existing photo background, realistic natural continuation, preserve the original subject, consistent lighting and perspective, no text'
    negative = str(payload.get('negative_prompt') or '').strip() or 'duplicate subject, extra person, extra animal, duplicated face, deformed, text, watermark, logo, frame, border'

    image, nw, nh, x, y = _geometry(image, direction, ratio)
    ow, oh = image.size

    # A blurred cover gives the model compatible colours at the new borders.
    seed_bg = image.resize((nw, nh), Image.Resampling.LANCZOS)
    seed_bg = seed_bg.filter(ImageFilter.GaussianBlur(radius=max(10, min(nw, nh) // 32)))
    canvas = seed_bg.copy()
    canvas.paste(image, (x, y))

    mask = Image.new('L', (nw, nh), 255)
    seam = max(8, min(24, min(ow, oh) // 36))
    keep = (x + seam, y + seam, x + ow - seam, y + oh - seam)
    if keep[2] > keep[0] and keep[3] > keep[1]:
        mask.paste(0, keep)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(2, seam // 3)))

    try:
        steps = int(payload.get('steps') or 18)
    except Exception:
        steps = 18
    steps = max(14, min(24, steps))
    try:
        guidance = float(payload.get('guidance_scale') or 6.5)
    except Exception:
        guidance = 6.5
    guidance = max(4.0, min(9.0, guidance))

    generator = None
    if payload.get('seed') is not None:
        try:
            generator = torch.Generator(device=DEVICE.type).manual_seed(int(payload['seed']))
        except Exception:
            generator = None

    with torch.inference_mode():
        result = PIPE(
            prompt=prompt,
            negative_prompt=negative,
            image=canvas,
            mask_image=mask,
            width=nw,
            height=nh,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
        ).images[0]

    # Never regenerate the customer's original centre pixels.
    result.paste(image, (x, y))
    out = io.BytesIO()
    result.save(out, format='PNG', optimize=True, compress_level=5)
    return out.getvalue(), result.width, result.height


def handler(job):
    started = time.perf_counter()
    try:
        payload = job.get('input') or {}
        task = str(payload.get('task') or payload.get('action') or 'outpaint').strip().lower()
        if task not in ('outpaint', 'expand', 'expand_image'):
            return {
                'status': 'error',
                'code': 'UNSUPPORTED_TASK',
                'error': 'this endpoint only supports outpaint',
                'model': MODEL_REPO,
            }
        image = _decode(payload.get('image_base64', ''))
        output, width, height = _outpaint(image, payload)
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
        return {
            'status': 'success',
            'task': 'outpaint',
            'model': MODEL_REPO,
            'width': width,
            'height': height,
            'execution_ms': elapsed_ms,
            'image_base64': base64.b64encode(output).decode('ascii'),
        }
    except Exception as exc:
        print('[OUTPAINT] job failed:', repr(exc))
        return {
            'status': 'error',
            'error': str(exc),
            'model': MODEL_REPO,
        }


if __name__ == '__main__':
    runpod.serverless.start({'handler': handler})
