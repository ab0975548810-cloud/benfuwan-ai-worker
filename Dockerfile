FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    MODEL_REPO=ZhengPeng7/BiRefNet \
    MODEL_SIZE=1024 \
    OUTPAINT_MODEL_REPO=runwayml/stable-diffusion-inpainting \
    OUTPAINT_MAX_EDGE=1024

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Bake both AI models into the image.  This makes the build heavier once, but
# prevents every scale-to-zero worker from downloading multi-GB weights again.
RUN python - <<'PY'
from transformers import AutoModelForImageSegmentation
from diffusers import StableDiffusionInpaintPipeline
import torch
seg='ZhengPeng7/BiRefNet'
out='runwayml/stable-diffusion-inpainting'
AutoModelForImageSegmentation.from_pretrained(seg, trust_remote_code=True)
try:
    StableDiffusionInpaintPipeline.from_pretrained(out, torch_dtype=torch.float16, variant='fp16')
except Exception:
    StableDiffusionInpaintPipeline.from_pretrained(out, torch_dtype=torch.float16)
print('BiRefNet + outpaint model cached in image')
PY

CMD ["python", "-u", "/app/handler.py"]
