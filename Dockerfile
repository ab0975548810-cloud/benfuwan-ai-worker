FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/runpod-volume/huggingface-cache \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    OUTPAINT_MODEL_REPO=runwayml/stable-diffusion-inpainting \
    OUTPAINT_MAX_EDGE=768

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# The model is intentionally NOT baked into this image.
# Runpod's Cached model setting provides it at /runpod-volume/huggingface-cache/hub,
# which keeps the container image much smaller and makes worker startup faster.
CMD ["python", "-u", "/app/handler.py"]
