import base64
import io
import os
import time
import threading

import runpod
import torch
from PIL import Image, ImageOps, ImageFilter
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

MODEL_REPO = os.environ.get('MODEL_REPO', 'ZhengPeng7/BiRefNet').strip() or 'ZhengPeng7/BiRefNet'
MODEL_SIZE = int(os.environ.get('MODEL_SIZE', '1024') or 1024)
MODEL_SIZE = max(512, min(1536, MODEL_SIZE))
OUTPAINT_MODEL_REPO = os.environ.get('OUTPAINT_MODEL_REPO', 'runwayml/stable-diffusion-inpainting').strip() or 'runwayml/stable-diffusion-inpainting'
OUTPAINT_MAX_EDGE = max(512, min(1280, int(os.environ.get('OUTPAINT_MAX_EDGE', '1024') or 1024)))
MAX_INPUT_BYTES = 8 * 1024 * 1024
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

_OUTPAINT_PIPE = None
_OUTPAINT_LOCK = threading.Lock()


def _decode_image(encoded: str, max_output_edge: int = MAX_OUTPUT_EDGE) -> Image.Image:
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


def _load_outpaint_pipe():
    global _OUTPAINT_PIPE
    if _OUTPAINT_PIPE is not None:
        return _OUTPAINT_PIPE
    with _OUTPAINT_LOCK:
        if _OUTPAINT_PIPE is not None:
            return _OUTPAINT_PIPE
        print(f'[AI] lazy-loading outpaint model={OUTPAINT_MODEL_REPO}')
        from diffusers import StableDiffusionInpaintPipeline
        kwargs = {'torch_dtype': DTYPE}
        if DEVICE.type == 'cuda':
            kwargs['variant'] = 'fp16'
        try:
            pipe = StableDiffusionInpaintPipeline.from_pretrained(OUTPAINT_MODEL_REPO, **kwargs)
        except Exception:
            kwargs.pop('variant', None)
            pipe = StableDiffusionInpaintPipeline.from_pretrained(OUTPAINT_MODEL_REPO, **kwargs)
        pipe = pipe.to(DEVICE)
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass
        _OUTPAINT_PIPE = pipe
        print('[AI] outpaint model ready')
        return _OUTPAINT_PIPE


def _round8(v):
    return max(64, int(round(v / 8.0) * 8))


def _outpaint(image: Image.Image, payload: dict) -> tuple[bytes, int, int]:
    direction = str(payload.get('direction') or 'all').lower()
    ratio = float(payload.get('expand_ratio') or 1.35)
    ratio = max(1.10, min(1.80, ratio))
    prompt = str(payload.get('prompt') or '').strip()
    if not prompt:
        prompt = 'seamlessly extend the existing photo background, natural continuation, preserve the original subject exactly, consistent lighting, realistic photo, no text'
    negative = str(payload.get('negative_prompt') or '').strip() or 'extra people, duplicate subject, duplicate face, deformed, text, watermark, logo, frame, border'

    ow, oh = image.size
    if direction in ('left', 'right'):
        nw, nh = int(ow * ratio), oh
    elif direction in ('top', 'up', 'bottom', 'down'):
        nw, nh = ow, int(oh * ratio)
    else:
        nw, nh = int(ow * ratio), int(oh * ratio)

    scale = min(1.0, OUTPAINT_MAX_EDGE / max(nw, nh))
    if scale < 1:
        ow2, oh2 = max(64, int(ow * scale)), max(64, int(oh * scale))
        image = image.resize((ow2, oh2), Image.Resampling.LANCZOS)
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

    # Build a soft canvas from edge colours so the inpaint model gets a stable starting point.
    bg = image.resize((nw, nh), Image.Resampling.LANCZOS).filter(ImageFilter.GaussianBlur(radius=max(12, min(nw, nh) // 28)))
    canvas = bg.copy()
    canvas.paste(image, (x, y))

    mask = Image.new('L', (nw, nh), 255)
    # Keep the original photo nearly untouched; feather only a narrow seam around the old boundary.
    seam = max(8, min(28, min(ow, oh) // 35))
    inner = (x + seam, y + seam, x + ow - seam, y + oh - seam)
    if inner[2] > inner[0] and inner[3] > inner[1]:
        mask.paste(0, inner)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(3, seam // 3)))

    pipe = _load_outpaint_pipe()
    generator = None
    seed = payload.get('seed')
    if seed is not None:
        try:
            generator = torch.Generator(device=DEVICE.type).manual_seed(int(seed))
        except Exception:
            generator = None

    steps = max(18, min(36, int(payload.get('steps') or 26)))
    guidance = max(3.5, min(10.0, float(payload.get('guidance_scale') or 7.0)))
    with torch.inference_mode():
        result = pipe(
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

    # Restore the original pixels in the centre so faces/textures do not get regenerated.
    result.paste(image, (x, y))
    buf = io.BytesIO()
    result.save(buf, format='PNG', optimize=True, compress_level=5)
    return buf.getvalue(), result.width, result.height


def handler(job):
    started = time.perf_counter()
    try:
        payload = job.get('input') or {}
        task = str(payload.get('task') or payload.get('action') or 'remove_background').strip().lower()
        image = _decode_image(payload.get('image_base64', ''), payload.get('max_output_edge', MAX_OUTPUT_EDGE))

        if task in ('outpaint', 'expand', 'expand_image'):
            output, out_w, out_h = _outpaint(image, payload)
            model_name = OUTPAINT_MODEL_REPO
        else:
            output = _remove_background(image)
            out_w, out_h = image.width, image.height
            model_name = MODEL_REPO

        elapsed_ms = round((time.perf_counter() - started) * 1000)
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

        return {
            'status': 'success',
            'task': task,
            'model': model_name,
            'width': out_w,
            'height': out_h,
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
