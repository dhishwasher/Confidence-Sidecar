from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    anthropic_api_key: str = ""
    upstream_openai_base_url: str = "https://api.openai.com"
    upstream_anthropic_base_url: str = "https://api.anthropic.com"

    database_url: str = "sqlite+aiosqlite:///./traces.db"

    # Customer auth — empty string disables auth (dev mode)
    sidecar_api_key: str = ""

    # Tier escalation thresholds
    tier1_confidence_band_low: float = 0.3
    tier1_confidence_band_high: float = 0.7
    tier1_k_samples: int = 4

    # Logprob capture
    top_logprobs_count: int = 5

    # Semantic entropy (Tier 1)
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_similarity_threshold: float = 0.85

    log_level: str = "INFO"


settings = Settings()
