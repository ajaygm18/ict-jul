"""
Yahoo Finance client for real-time and historical stock data
"""
import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from config.settings import settings
import asyncio
import aiohttp

logger = logging.getLogger(__name__)

class YFinanceClient:
    def __init__(self):
        self.default_period = "1d"
        self.default_interval = "1m"
        self.max_retries = 3
        self.retry_delay = 1  # seconds
        
    async def get_real_time_stock_data(
        self, 
        symbol: str, 
        period: str = "1d", 
        interval: str = "1m",
        prepost: bool = True
    ) -> pd.DataFrame:
        """
        Get real-time stock price data via yfinance
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            prepost: Include pre and post market data
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                period=period, 
                interval=interval, 
                prepost=prepost,
                auto_adjust=True,
                back_adjust=True
            )
            
            if data.empty:
                logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
            # Add symbol and timeframe columns
            data['symbol'] = symbol
            data['timeframe'] = interval
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            
            # Standardize column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            if 'datetime' in data.columns:
                data.rename(columns={'datetime': 'timestamp'}, inplace=True)
            elif 'date' in data.columns:
                data.rename(columns={'date': 'timestamp'}, inplace=True)
            
            # Calculate bid/ask spread approximation
            data['bid_price'] = data['close'] - (data['high'] - data['low']) * 0.1
            data['ask_price'] = data['close'] + (data['high'] - data['low']) * 0.1
            data['spread'] = data['ask_price'] - data['bid_price']
            
            return self._clean_and_validate_data(data)
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_stocks_data(
        self, 
        symbols: List[str], 
        period: str = "1d", 
        interval: str = "1m"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks concurrently
        """
        tasks = []
        for symbol in symbols:
            task = self.get_real_time_stock_data(symbol, period, interval)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        stock_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching data for {symbol}: {result}")
                stock_data[symbol] = pd.DataFrame()
            else:
                stock_data[symbol] = result
                
        return stock_data
    
    def get_stock_fundamentals(self, symbol: str) -> Dict:
        """
        Get stock fundamental data
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            fundamentals = {
                'symbol': symbol,
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE'),
                'eps': info.get('trailingEps'),
                'dividend_yield': info.get('dividendYield'),
                'beta': info.get('beta'),
                'sector': info.get('sector'),
                'industry': info.get('industry'),
                'employees': info.get('fullTimeEmployees'),
                'analyst_target_price': info.get('targetMeanPrice'),
                'analyst_recommendation': info.get('recommendationKey'),
                'company_name': info.get('longName'),
                'business_summary': info.get('longBusinessSummary'),
                'website': info.get('website'),
                'country': info.get('country'),
                'exchange': info.get('exchange'),
                'currency': info.get('currency'),
            }
            
            return fundamentals
            
        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return {'symbol': symbol}
    
    def get_earnings_calendar(self, symbol: str) -> pd.DataFrame:
        """
        Get earnings calendar data
        """
        try:
            ticker = yf.Ticker(symbol)
            calendar = ticker.calendar
            
            if calendar is not None and not calendar.empty:
                calendar['symbol'] = symbol
                return calendar
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_institutional_holders(self, symbol: str) -> pd.DataFrame:
        """
        Get institutional holders data
        """
        try:
            ticker = yf.Ticker(symbol)
            holders = ticker.institutional_holders
            
            if holders is not None and not holders.empty:
                holders['symbol'] = symbol
                return holders
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching institutional holders for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_options_data(self, symbol: str) -> Dict:
        """
        Get options data for a stock
        """
        try:
            ticker = yf.Ticker(symbol)
            exp_dates = ticker.options
            
            if not exp_dates:
                return {}
            
            # Get the nearest expiration
            nearest_exp = exp_dates[0]
            opt_chain = ticker.option_chain(nearest_exp)
            
            return {
                'symbol': symbol,
                'expiration_dates': exp_dates,
                'calls': opt_chain.calls,
                'puts': opt_chain.puts
            }
            
        except Exception as e:
            logger.error(f"Error fetching options data for {symbol}: {e}")
            return {}
    
    def _clean_and_validate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate stock data
        """
        if data.empty:
            return data
        
        # Remove rows with missing OHLC data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        data = data.dropna(subset=required_columns)
        
        # Ensure positive prices and volumes
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data = data[data[col] > 0]
        
        if 'volume' in data.columns:
            data = data[data['volume'] >= 0]
        
        # Ensure high >= low and validate OHLC relationships
        data = data[data['high'] >= data['low']]
        data = data[data['high'] >= data['open']]
        data = data[data['high'] >= data['close']]
        data = data[data['low'] <= data['open']]
        data = data[data['low'] <= data['close']]
        
        # Sort by timestamp
        if 'timestamp' in data.columns:
            data = data.sort_values('timestamp')
        
        # Remove duplicates
        if 'timestamp' in data.columns and 'symbol' in data.columns:
            data = data.drop_duplicates(subset=['timestamp', 'symbol'], keep='last')
        
        return data
    
    def get_market_hours_data(self) -> Dict:
        """
        Get market hours information
        """
        # Note: yfinance doesn't provide market hours directly
        # This is a simplified implementation
        return {
            'premarket_start': '04:00',
            'market_open': '09:30',
            'market_close': '16:00',
            'afterhours_end': '20:00',
            'timezone': 'US/Eastern',
            'is_trading_day': True  # This would need to be enhanced with holiday calendar
        }

# Global instance
yfinance_client = YFinanceClient()