from fredapi import Fred
from ict_stock_trader.config.settings import FRED_API_KEY

class FredClient:
    def __init__(self):
        if not FRED_API_KEY:
            raise ValueError("FRED_API_KEY is not set in the configuration.")
        self.fred = Fred(api_key=FRED_API_KEY)

    def get_series(self, series_id: str):
        """
        Fetches a data series from the FRED API.
        """
        try:
            return self.fred.get_series(series_id)
        except Exception as e:
            print(f"Error fetching FRED series {series_id}: {e}")
            return None
