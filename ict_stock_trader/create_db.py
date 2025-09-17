from ict_stock_trader.app.database import engine, Base
from ict_stock_trader.app.models.stock_data import StockData
from ict_stock_trader.app.models.ict_patterns import ICTPatterns
from ict_stock_trader.app.models.trading_signals import TradingSignals

def create_database():
    print("Creating database and tables...")
    Base.metadata.create_all(bind=engine)
    print("Database and tables created successfully.")

if __name__ == "__main__":
    create_database()
