#!/usr/bin/env sh

set -e

echo "Starting SEO Organic Growth Intelligence service..."

if [ "${POSTGRES_ENABLED}" = "true" ]; then
    echo "Waiting for PostgreSQL..."

    python - <<'PY'
import os
import time

from sqlalchemy import create_engine, text

host = os.getenv(
    "POSTGRES_HOST",
    "postgres",
)

port = os.getenv(
    "POSTGRES_PORT",
    "5432",
)

database = os.getenv(
    "POSTGRES_DATABASE",
    "seo_intelligence",
)

user = os.getenv(
    "POSTGRES_USER",
    "postgres",
)

password = os.getenv(
    "POSTGRES_PASSWORD",
    "",
)

url = (
    f"postgresql+psycopg2://"
    f"{user}:{password}"
    f"@{host}:{port}/{database}"
)

engine = create_engine(
    url,
    pool_pre_ping=True,
)

for attempt in range(30):
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        print(
            "PostgreSQL is ready."
        )

        break

    except Exception as exc:
        print(
            "PostgreSQL is not ready yet "
            f"({attempt + 1}/30): {exc}"
        )

        time.sleep(2)

else:
    raise RuntimeError(
        "PostgreSQL did not become ready in time."
    )
PY
fi

echo "Executing command: $*"

exec "$@"