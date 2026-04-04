from pydantic_settings import BaseSettings
from pathlib import Path
import os


class Settings(BaseSettings):
    # postgres
    db_host: str
    db_port: int
    db_user: str
    db_pass: str
    db_name: str

    # chroma
    chroma_host: str
    chroma_port: int

    # LLM provider: "groq" or "openai" (the other is used as fallback)
    llm_provider: str

    # openai
    openai_api_key: str = ""
    openai_model: str

    # groq
    groq_api_key: str = ""
    groq_model: str

    # file storage
    upload_dir: str

    # embedding model
    embedding_model: str
    embedding_batch_size: int

    # retrieval
    top_k: int

    # auth
    jwt_secret: str
    jwt_expiry_hours: int

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"

    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"

    def ensure_upload_dir(self):
        Path(self.upload_dir).mkdir(parents=True, exist_ok=True)


settings = Settings()
