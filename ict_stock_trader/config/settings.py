"""
Configuration settings for ICT Stock Trader
"""
import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API Configuration
    NEWS_API_KEY: str = "135472927f3147439450e47924c2523b"
    FRED_API_KEY: str = "747c4c16bc76a3dfc54d6d63c0ba9e4d"
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./ict_trader.db"
    
    # FastAPI Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "ICT Stock Trader"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Personal Full-Stack ICT Stock Trading AI Agent"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Data Configuration
    DEFAULT_STOCKS: list = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "CRM"]
    DEFAULT_TIMEFRAMES: list = ["1m", "5m", "15m", "1h", "1d"]
    
    # Trading Hours (ET)
    PREMARKET_START: str = "04:00"
    MARKET_OPEN: str = "09:30"
    MARKET_CLOSE: str = "16:00"
    AFTERHOURS_END: str = "20:00"
    
    # ICT Configuration
    PATTERN_CONFIDENCE_THRESHOLD: float = 0.75
    MAX_CONCURRENT_REQUESTS: int = 10
    DATA_RETENTION_DAYS: int = 365
    
    # Cache Configuration
    CACHE_TTL_SECONDS: int = 60
    MAX_CACHE_SIZE: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()