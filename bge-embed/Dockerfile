# BGE-M3 hybrid embed service — CPU build.
# (GPU option: switch the torch install to the CUDA wheel and set
#  BGE_DEVICE=cuda / BGE_FP16=1 in compose. Same application code.)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/models \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU torch first (keeps the image lean — no CUDA), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Only the service is baked in. embed_core.py is BIND-MOUNTED at runtime (see
# compose) so this image never carries a drifting copy.
COPY bge_m3_service.py .

EXPOSE 8055
CMD ["uvicorn", "bge_m3_service:app", "--host", "0.0.0.0", "--port", "8055", "--workers", "1"]
