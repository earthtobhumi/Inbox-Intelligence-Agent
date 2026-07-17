"""
Celery app instance. Broker and result backend are both Redis -- matches
the original architecture doc (Celery + Redis + Celery Beat).

Run a worker with:
    celery -A celery_app worker --loglevel=info

Run the beat scheduler (separate process) with:
    celery -A celery_app beat --loglevel=info
"""
import os

from celery import Celery
from celery.schedules import crontab

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "email_agent",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

# Beat schedule: check every hour whether any user's digest is due.
# The hourly cadence is just how often we CHECK -- each user's actual
# daily/weekly cadence is decided inside the task by comparing
# last_digest_sent_at against their configured frequency (see digest.is_digest_due).
celery_app.conf.beat_schedule = {
    "check-and-send-due-digests": {
        "task": "tasks.check_and_send_due_digests",
        "schedule": crontab(minute=0),  # top of every hour
    },
}
