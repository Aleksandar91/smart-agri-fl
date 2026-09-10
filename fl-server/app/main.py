import os
from datetime import datetime, timezone

from fastapi import FastAPI


APP_NAME = os.getenv("APP_NAME", "fl-server")

app = FastAPI(title=APP_NAME)


@app.get("/health")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.get("/ping")
def ping():
    return {
        "service": APP_NAME,
        "ts": datetime.now(timezone.utc).isoformat(),
        "message": "pong",
    }

