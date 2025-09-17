from sqlalchemy import Column, Integer, String, Float, DateTime
from ict_stock_trader.app.database import Base

class TradingSignals(Base):
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), nullable=False, index=True)
    strategy_name = Column(String(100), nullable=False, index=True)
    signal_type = Column(String(10), nullable=False)  # BUY/SELL
    confidence = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)

    entry_price = Column(Float, nullable=True)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    risk_reward_ratio = Column(Float, nullable=True)

    outcome = Column(String(20), nullable=True)  # WIN/LOSS/PENDING
    pnl = Column(Float, nullable=True)
