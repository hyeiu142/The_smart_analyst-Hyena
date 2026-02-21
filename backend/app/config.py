from pydantic_settings import BaseSettings
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

    chunk_size: int = 1024
    chunk_overlap: int = 200

    debug: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache 
def get_settings() -> Settings:
    return Settings()