FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
RUN addgroup --system app && adduser --system --ingroup app app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY agents ./agents
COPY api ./api
COPY prompts ./prompts
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "127.0.0.1"]
