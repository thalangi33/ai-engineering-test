from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True

    docs_dir: Path = PROJECT_ROOT / "docs"
    index_path: Path = PROJECT_ROOT / "data" / "index.json"
    llm_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    deepseek_api_key: str = ""
    ollama_base_url: str = "http://127.0.0.1:11434"
    # gemini-2.0-flash | llama3.2 | llama-3.1-8b-instant | deepseek-v4-flash
    llm_model: str = "gemini-2.0-flash"
    # text-embedding-3-small | gemini-embedding-001 | all-MiniLM-L6-v2
    embedding_model: str = "text-embedding-3-small"
    top_k: int = 5
    temperature: float = 0.0


settings = Settings()
