FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements.txt .
COPY src/ src/
COPY api/ api/

VOLUME /app/data

ENV TIMESCALE_DSN=postgres://ccode:ccode@timescaledb:5432/ccode

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

CMD ["python", "-m", "src.main"]
