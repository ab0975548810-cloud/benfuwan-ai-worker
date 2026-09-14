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

# Keep the default endpoint small and fast: it only handles background removal.
# The generative outpaint model now lives on the dedicated outpaint-worker branch
# so a multi-GB diffusion image can never slow down normal background removal.
RUN python - <<'PY'
from transformers import AutoModelForImageSegmentation
repo='ZhengPeng7/BiRefNet'
AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
print('BiRefNet cached in image')
PY

CMD ["python", "-u", "/app/handler.py"]
