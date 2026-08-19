FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY config ./config
COPY src ./src
COPY docs ./docs
COPY tests ./tests
COPY docker ./docker

COPY main.py .
COPY scheduler.py .
COPY README.md .
COPY .env.example .

RUN mkdir -p \
        /app/outputs \
        /app/logs \
    && chmod +x /app/docker/entrypoint.sh

ENTRYPOINT ["/app/docker/entrypoint.sh"]

CMD ["python", "main.py"]