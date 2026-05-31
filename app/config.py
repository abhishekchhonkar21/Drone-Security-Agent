"""
app/config.py
Application configuration loaded from environment variables via pydantic-settings.
"""
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # --- OpenAI ---
    azure_openai_key:str = Field(env="AZURE_OPENAI_KEY")
    azure_openai_endpoint:str = Field(env="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version:str = Field(env="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment:str = Field(env="AZURE_OPENAI_DEPLOYMENT")

    # --- MongoDB ---
    mongodb_uri: str = Field(..., env="MONGODB_URI")
    mongodb_db_name: str = Field(default="drone_security", env="MONGODB_DB_NAME")

    # --- Pipeline ---
    frame_sample_interval_sec: int = Field(default=2, env="FRAME_SAMPLE_INTERVAL_SEC")
    alert_window_size: int = Field(default=5, env="ALERT_WINDOW_SIZE")
    telemetry_match_tolerance_sec: int = Field(default=5, env="TELEMETRY_MATCH_TOLERANCE_SEC")

    # --- Storage ---
    storage_base_path: str = Field(default="storage", env="STORAGE_BASE_PATH")

    # --- App ---
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    app_host: str = Field(default="0.0.0.0", env="APP_HOST")
    app_port: int = Field(default=8000, env="APP_PORT")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


# Singleton instance used across the app
settings = Settings()
