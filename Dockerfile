FROM python:3.11-slim
WORKDIR /app
ARG OCR_INSTALL_RAPIDOCR=0
COPY requirements.txt requirements-rapidocr.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && if [ "$OCR_INSTALL_RAPIDOCR" = "1" ]; then pip install --no-cache-dir -r requirements-rapidocr.txt; fi \
    && useradd --system --uid 10001 --create-home ocr \
    && mkdir -p /data/ocr-jobs && chown ocr:ocr /data/ocr-jobs
COPY app.py engine.py rapidocr_engine.py hybrid_engine.py pdf_layer.py worker.py ./
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OCR_JOB_ROOT=/data/ocr-jobs
USER ocr
EXPOSE 8092
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8092/health', timeout=4)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8092", "--workers", "1", "--no-access-log"]
