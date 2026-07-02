from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    llama_cloud_api_key: str
    google_api_key: str

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    redis_url: str = "redis://localhost:6379/0"

    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    vision_model: str = "gpt-4o"

    chunk_size: int = 1024
    chunk_overlap: int = 200
    upload_dir: str = "uploads"

    doclayout_model_path: str = "research/image_extraction/data/models/doclayout_yolo_docstructbench_imgsz1024.pt"
    doclayout_device: str = "cpu"
    doclayout_conf: float = 0.20
    doclayout_dpi: int = 160
    doclayout_imgsz: int = 1024

    debug: bool = False

    @field_validator("debug", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "prod", "production", "false", "0", "no", "off", "warn", "warning", "info", "error"}:
                return False
            if normalized in {"debug", "dev", "development", "true", "1", "yes", "on"}:
                return True
        return value

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

@lru_cache 
def get_settings() -> Settings:
    return Settings()
