from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
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
    Get raw stock data for a given symbol from Yahoo Finance.
    """
    try:
        data = stock_manager.get_real_time_stock_data(symbol, period, interval)
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol '{symbol}' for the given period.")

        data = data.reset_index()
        # Convert all timestamp columns to string for JSON compatibility
        for col in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data[col]):
                data[col] = data[col].astype(str)

        return data.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Could not retrieve data for symbol '{symbol}'.")
