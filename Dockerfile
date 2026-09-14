FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    OUTPAINT_MODEL_REPO=runwayml/stable-diffusion-inpainting \
    OUTPAINT_MAX_EDGE=768

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Dedicated generative worker: bake only the inpainting model into this image.
# This branch is intentionally separate from BiRefNet so normal background
# removal never waits for a multi-GB diffusion image to start.
RUN python - <<'PY'
from diffusers import StableDiffusionInpaintPipeline
import torch
repo='runwayml/stable-diffusion-inpainting'
try:
    StableDiffusionInpaintPipeline.from_pretrained(repo, torch_dtype=torch.float16, variant='fp16')
except Exception:
    StableDiffusionInpaintPipeline.from_pretrained(repo, torch_dtype=torch.float16)
print('Outpaint model cached in image')
PY

CMD ["python", "-u", "/app/handler.py"]
