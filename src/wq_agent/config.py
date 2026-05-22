from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    WQ_USERNAME: str = ""
    WQ_PASSWORD: str = ""
    WQ_REGION: str = "USA"
    WQ_UNIVERSE: str = "TOP3000"
    WQ_DELAY: int = 1
    WQ_NEUTRALIZATION: str = "INDUSTRY"
    WQ_TRUNCATION: float = 0.08
    WQ_PASTEURIZATION: str = "ON"
    WQ_MAX_CONCURRENT: int = 5

    LLM_PROVIDER: str = "kimi"
    LLM_MODEL: str = ""
    LLM_MAX_TOKENS: int = 32768
    KIMI_API_KEY: str = ""
    KIMI_BASE_URL: str = "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
    KIMI_MODEL: str = "kimi-k2.6"
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1/chat/completions"
    DEEPSEEK_MODEL: str = "deepseek-chat"

    MIN_FITNESS: float = 0.5
    MIN_SHARPE: float = 1.0
    MAX_TURNOVER: float = 0.7
    MIN_RETURNS: float = 0.05

    DB_PATH: str = "./wq_agent.db"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
