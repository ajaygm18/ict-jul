import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME: str = "ICT Stock Trading AI Agent"
API_V1_STR: str = "/api/v1"

# FRED API Key
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "747c4c16bc76a3dfc54d6d63c0ba9e4d")

# NewsAPI Key
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "135472927f3147439450e47924c2523b")

# Database
DATABASE_URL = "sqlite:///./ict_trader.db"
