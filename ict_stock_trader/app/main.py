from fastapi import FastAPI, HTTPException
from ict_stock_trader.config.settings import PROJECT_NAME, API_V1_STR
from ict_stock_trader.app.data.yfinance_client import StockDataManager
from ict_stock_trader.app.database import engine, Base

# Create all tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title=PROJECT_NAME)

stock_manager = StockDataManager()

@app.get("/")
def read_root():
    return {"message": f"Welcome to {PROJECT_NAME}"}

@app.get(f"{API_V1_STR}/stock/{{symbol}}", tags=["Stock Data"])
def get_stock_data(symbol: str, period: str = "1d", interval: str = "1m"):
    """
    Get stock data for a given symbol from Yahoo Finance.
    """
    try:
        data = stock_manager.get_real_time_stock_data(symbol, period, interval)
        if data.empty:
            # This handles cases where yfinance returns an empty dataframe for a valid symbol
            # but with no data in the requested period.
            raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}' for the given period.")

        # Reset index to make Datetime a column and format it for JSON response
        data.reset_index(inplace=True)
        # Convert timezone-aware Datetime to string
        if 'Datetime' in data.columns:
            data['Datetime'] = data['Datetime'].astype(str)
        if 'Date' in data.columns:
            data['Date'] = data['Date'].astype(str)

        return data.to_dict(orient="records")
    except Exception as e:
        # This will catch exceptions from yfinance (e.g., for an invalid symbol that causes an API error)
        # or from the data processing steps above.
        # We'll return a 404, assuming the symbol is the most likely cause of failure.
        raise HTTPException(status_code=404, detail=f"Could not retrieve data for symbol '{symbol}'.")
