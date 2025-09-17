"""
Data processor for combining and processing stock market data from multiple sources
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import logging
from app.data.yfinance_client import yfinance_client
from app.data.fred_client import fred_client
import asyncio
import requests
from config.settings import settings

logger = logging.getLogger(__name__)

class NewsAPIClient:
    def __init__(self):
        self.api_key = settings.NEWS_API_KEY
        self.base_url = "https://newsapi.org/v2"
        
    async def get_financial_news(self, symbol: str = None, limit: int = 10) -> List[Dict]:
        """
        Get financial news from NewsAPI
        """
        try:
            url = f"{self.base_url}/everything"
            params = {
                'apiKey': self.api_key,
                'language': 'en',
                'sortBy': 'publishedAt',
                'pageSize': limit
            }
            
            if symbol:
                params['q'] = f"{symbol} stock OR {symbol} earnings OR {symbol} financial"
                params['domains'] = 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,yahoo.com'
            else:
                params['q'] = 'stock market OR NYSE OR NASDAQ OR S&P 500 OR trading'
                params['domains'] = 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,yahoo.com'
            
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('articles', [])
            else:
                logger.error(f"NewsAPI error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []

class DataProcessor:
    def __init__(self):
        self.yf_client = yfinance_client
        self.fred_client = fred_client
        self.news_client = NewsAPIClient()
        self.cache = {}
        self.cache_ttl = timedelta(seconds=settings.CACHE_TTL_SECONDS)
        
    async def get_comprehensive_market_data(
        self, 
        symbols: List[str],
        timeframes: List[str] = None,
        include_fundamentals: bool = True,
        include_economic: bool = True,
        include_news: bool = True
    ) -> Dict:
        """
        Get comprehensive market data combining all sources
        """
        if timeframes is None:
            timeframes = settings.DEFAULT_TIMEFRAMES
            
        result = {
            'timestamp': datetime.now(),
            'stocks': {},
            'economic_indicators': {},
            'news': [],
            'market_context': {}
        }
        
        # Get stock data for all symbols and timeframes
        logger.info(f"Fetching stock data for {len(symbols)} symbols")
        for timeframe in timeframes:
            stock_data = await self.yf_client.get_multiple_stocks_data(
                symbols, period="5d", interval=timeframe
            )
            
            for symbol, data in stock_data.items():
                if symbol not in result['stocks']:
                    result['stocks'][symbol] = {}
                result['stocks'][symbol][timeframe] = data
        
        # Get fundamentals if requested
        if include_fundamentals:
            logger.info("Fetching fundamental data")
            fundamentals_tasks = []
            for symbol in symbols:
                task = asyncio.create_task(
                    self._get_cached_fundamentals(symbol)
                )
                fundamentals_tasks.append(task)
            
            fundamentals_results = await asyncio.gather(*fundamentals_tasks)
            for symbol, fundamentals in zip(symbols, fundamentals_results):
                if symbol in result['stocks']:
                    result['stocks'][symbol]['fundamentals'] = fundamentals
        
        # Get economic indicators if requested
        if include_economic:
            logger.info("Fetching economic indicators")
            economic_data = await self._get_key_economic_indicators()
            result['economic_indicators'] = economic_data
        
        # Get news if requested
        if include_news:
            logger.info("Fetching financial news")
            news_data = await self._get_financial_news_for_symbols(symbols)
            result['news'] = news_data
        
        # Calculate market context
        result['market_context'] = await self._calculate_market_context(result)
        
        return result
    
    async def _get_cached_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamentals with caching
        """
        cache_key = f"fundamentals_{symbol}"
        
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_ttl:
                return cached_data
        
        fundamentals = self.yf_client.get_stock_fundamentals(symbol)
        self.cache[cache_key] = (fundamentals, datetime.now())
        
        return fundamentals
    
    async def _get_key_economic_indicators(self) -> Dict:
        """
        Get key economic indicators for market context
        """
        key_indicators = [
            'fed_funds_rate',
            'treasury_10y',
            'vix',
            'unemployment_rate',
            'cpi'
        ]
        
        indicators_data = await self.fred_client.get_multiple_indicators(
            key_indicators,
            start_date=datetime.now() - timedelta(days=30)
        )
        
        # Get latest values
        latest_values = {}
        for indicator, data in indicators_data.items():
            if not data.empty:
                latest_values[indicator] = {
                    'current_value': data['value'].iloc[-1],
                    'previous_value': data['value'].iloc[-2] if len(data) > 1 else None,
                    'change': data['value'].iloc[-1] - data['value'].iloc[-2] if len(data) > 1 else None,
                    'date': data['date'].iloc[-1]
                }
        
        return latest_values
    
    async def _get_financial_news_for_symbols(self, symbols: List[str]) -> List[Dict]:
        """
        Get financial news for multiple symbols
        """
        all_news = []
        
        # Get general market news
        general_news = await self.news_client.get_financial_news(limit=5)
        all_news.extend(general_news)
        
        # Get symbol-specific news
        for symbol in symbols[:3]:  # Limit to first 3 symbols to avoid API limits
            symbol_news = await self.news_client.get_financial_news(symbol, limit=2)
            all_news.extend(symbol_news)
        
        # Remove duplicates and sort by date
        seen_urls = set()
        unique_news = []
        for article in all_news:
            if article.get('url') not in seen_urls:
                seen_urls.add(article.get('url'))
                unique_news.append(article)
        
        # Sort by publication date
        unique_news.sort(key=lambda x: x.get('publishedAt', ''), reverse=True)
        
        return unique_news[:10]  # Return top 10 most recent
    
    async def _calculate_market_context(self, market_data: Dict) -> Dict:
        """
        Calculate overall market context and sentiment
        """
        context = {
            'overall_sentiment': 'neutral',
            'market_trend': 'sideways',
            'volatility_level': 'normal',
            'economic_backdrop': 'neutral',
            'session_analysis': {}
        }
        
        try:
            # Analyze VIX for volatility
            economic_indicators = market_data.get('economic_indicators', {})
            if 'vix' in economic_indicators:
                vix_value = economic_indicators['vix']['current_value']
                if vix_value > 30:
                    context['volatility_level'] = 'high'
                elif vix_value > 20:
                    context['volatility_level'] = 'elevated'
                elif vix_value < 15:
                    context['volatility_level'] = 'low'
            
            # Analyze interest rates for economic backdrop
            if 'fed_funds_rate' in economic_indicators and 'treasury_10y' in economic_indicators:
                fed_rate = economic_indicators['fed_funds_rate']['current_value']
                treasury_10y = economic_indicators['treasury_10y']['current_value']
                
                if fed_rate > 4:
                    context['economic_backdrop'] = 'restrictive'
                elif fed_rate < 2:
                    context['economic_backdrop'] = 'accommodative'
            
            # Analyze stock performance for overall sentiment
            stock_performance = []
            stocks = market_data.get('stocks', {})
            
            for symbol, data in stocks.items():
                if '1d' in data and not data['1d'].empty:
                    df = data['1d']
                    if len(df) > 1:
                        change = (df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]
                        stock_performance.append(change)
            
            if stock_performance:
                avg_performance = np.mean(stock_performance)
                if avg_performance > 0.01:
                    context['overall_sentiment'] = 'bullish'
                    context['market_trend'] = 'uptrend'
                elif avg_performance < -0.01:
                    context['overall_sentiment'] = 'bearish'
                    context['market_trend'] = 'downtrend'
            
            # Current session analysis
            now = datetime.now()
            current_hour = now.hour
            
            if 4 <= current_hour < 9:
                context['session_analysis']['current_session'] = 'premarket'
            elif 9 <= current_hour < 16:
                context['session_analysis']['current_session'] = 'market_hours'
            elif 16 <= current_hour < 20:
                context['session_analysis']['current_session'] = 'afterhours'
            else:
                context['session_analysis']['current_session'] = 'closed'
            
        except Exception as e:
            logger.error(f"Error calculating market context: {e}")
        
        return context
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate common technical indicators
        """
        if df.empty or len(df) < 20:
            return df
        
        try:
            # Moving averages
            df['sma_9'] = df['close'].rolling(window=9).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['ema_9'] = df['close'].ewm(span=9).mean()
            df['ema_20'] = df['close'].ewm(span=20).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price action
            df['price_change'] = df['close'].pct_change()
            df['high_low_range'] = df['high'] - df['low']
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['true_range'].rolling(window=14).mean()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
        
        return df
    
    def detect_market_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and label market sessions
        """
        if df.empty or 'timestamp' not in df.columns:
            return df
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            
            # Define session labels
            conditions = [
                (df['hour'] >= 4) & (df['hour'] < 9),  # Premarket: 4:00-9:00 AM
                (df['hour'] == 9) & (df['minute'] >= 30),  # Market open: 9:30 AM
                (df['hour'] >= 10) & (df['hour'] < 16),  # Market hours: 10:00 AM - 4:00 PM
                (df['hour'] >= 16) & (df['hour'] < 20),  # After hours: 4:00-8:00 PM
            ]
            
            choices = ['premarket', 'market_open', 'market_hours', 'afterhours']
            df['session'] = np.select(conditions, choices, default='closed')
            
        except Exception as e:
            logger.error(f"Error detecting market sessions: {e}")
        
        return df
    
    async def process_real_time_data(self, symbol: str, timeframe: str = "1m") -> Dict:
        """
        Process real-time data for a single symbol with all enhancements
        """
        try:
            # Determine appropriate period based on timeframe
            if timeframe in ["1d", "1D", "daily"]:
                period = "1y"  # 1 year of daily data
            elif timeframe in ["1h", "1H", "hour", "hourly"]:
                period = "1mo"  # 1 month of hourly data  
            elif timeframe in ["5m", "15m", "30m"]:
                period = "5d"  # 5 days of minute data
            else:
                period = "1d"  # Default for minute data
                
            # Get raw data
            raw_data = await self.yf_client.get_real_time_stock_data(
                symbol, period=period, interval=timeframe
            )
            
            if raw_data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Add technical indicators
            enhanced_data = self.calculate_technical_indicators(raw_data)
            
            # Add session detection
            enhanced_data = self.detect_market_sessions(enhanced_data)
            
            # Get fundamentals
            fundamentals = await self._get_cached_fundamentals(symbol)
            
            # Get latest economic context
            economic_context = await self._get_key_economic_indicators()
            
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': enhanced_data,
                'fundamentals': fundamentals,
                'economic_context': economic_context,
                'last_update': datetime.now(),
                'data_points': len(enhanced_data)
            }
            
        except Exception as e:
            logger.error(f"Error processing real-time data for {symbol}: {e}")
            return {'error': str(e)}

# Global instance
data_processor = DataProcessor()