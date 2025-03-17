import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Server settings
    host: str = os.getenv("host", "0.0.0.0")
    port: int = int(os.getenv("port", 8010))
    debug: bool = os.getenv("debug", "True").lower() == "true"
    # class Config:
    #     env_file = ".env"


settings = Settings()
