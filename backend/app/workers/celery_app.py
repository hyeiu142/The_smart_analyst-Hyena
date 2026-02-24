from celery import Celery
from backend.app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "hyena", 
    broker=settings.redis_url, 
    backend=settings.redis_url, 
    include=["backend.app.workers.tasks"], 
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,
    task_time_limit=900,
    broker_transport_options={"visibility_timeout": 3600},
)