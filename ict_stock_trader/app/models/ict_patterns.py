from sqlalchemy import Column, Integer, String, Float, DateTime, JSON
from ict_stock_trader.app.database import Base

class ICTPatterns(Base):
    __tablename__ = "ict_patterns"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    confidence = Column(Float, nullable=False)

    pattern_data = Column(JSON, nullable=False)

    success_rate = Column(Float, nullable=True)
    avg_return = Column(Float, nullable=True)
