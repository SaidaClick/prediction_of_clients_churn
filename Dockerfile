FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.7.21 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN uv venv /opt/venv && uv pip sync --python /opt/venv/bin/python requirements.txt
COPY pipeline.py config.json ./
COPY data/telemetry-xs.tar.gz data/

ENTRYPOINT ["/opt/venv/bin/python", "pipeline.py"]
