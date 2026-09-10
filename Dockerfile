FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    MODEL_REPO=ZhengPeng7/BiRefNet \
    MODEL_SIZE=1024

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY handler.py /app/handler.py

# Bake the model weights + trusted model code into the image so new workers
# do not download them on every cold start.
RUN python - <<'PY'
from transformers import AutoModelForImageSegmentation
repo='ZhengPeng7/BiRefNet'
AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
print('BiRefNet cached in image')
PY

CMD ["python", "-u", "/app/handler.py"]
