"""
FRED API client for economic indicators and market data
"""
import fredapi
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import logging
from config.settings import settings

logger = logging.getLogger(__name__)

class FREDClient:
    def __init__(self):
        self.fred = fredapi.Fred(api_key=settings.FRED_API_KEY)
        self.economic_indicators = {
            # Interest Rates
            'fed_funds_rate': 'FEDFUNDS',
            'treasury_10y': 'GS10',
            'treasury_2y': 'GS2',
            'treasury_3m': 'GS3M',
            
            # Inflation
            'cpi': 'CPIAUCSL',
            'core_cpi': 'CPILFESL',
            'pce': 'PCE',
            'core_pce': 'PCEPILFE',
            
            # GDP and Growth
            'gdp': 'GDP',
            'gdp_growth': 'GDPC1',
            'industrial_production': 'INDPRO',
            'retail_sales': 'RSXFS',
            
            # Employment
            'unemployment_rate': 'UNRATE',
            'nonfarm_payrolls': 'PAYEMS',
            'labor_force_participation': 'CIVPART',
            'initial_claims': 'ICSA',
            
            # Market Indicators
            'vix': 'VIXCLS',
            'dollar_index': 'DTWEXBGS',
            'sp500': 'SP500',
            
            # Money Supply
            'm1_money_supply': 'M1SL',
            'm2_money_supply': 'M2SL',
            
            # Consumer Sentiment
            'consumer_sentiment': 'UMCSENT',
            'consumer_confidence': 'CSCICP03USM665S',
            
            # Housing
            'home_sales': 'EXHOSLUSM495S',
            'housing_starts': 'HOUST',
            'home_price_index': 'CSUSHPISA',
        }
    
    async def get_economic_indicator(
        self, 
        indicator: str, 
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get economic indicator data from FRED
        
        Args:
            indicator: Either a key from self.economic_indicators or a FRED series ID
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with economic indicator data
        """
        try:
            # Use predefined indicator or the provided string as series ID
            series_id = self.economic_indicators.get(indicator, indicator)
            
            # Default to last 2 years if no dates provided
            if start_date is None:
                start_date = datetime.now() - timedelta(days=730)
            if end_date is None:
                end_date = datetime.now()
            
            data = self.fred.get_series(
                series_id, 
                start=start_date, 
                end=end_date
            )
            
            if data.empty:
                logger.warning(f"No data received for indicator: {indicator}")
                return pd.DataFrame()
            
            # Convert to DataFrame with proper columns
            df = data.to_frame(name='value')
            df['indicator'] = indicator
            df['series_id'] = series_id
            df = df.reset_index()
            df.rename(columns={'index': 'date'}, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching FRED data for {indicator}: {e}")
            return pd.DataFrame()
    
    async def get_multiple_indicators(
        self, 
        indicators: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Get multiple economic indicators
        """
        results = {}
        
        for indicator in indicators:
            try:
                data = await self.get_economic_indicator(indicator, start_date, end_date)
                results[indicator] = data
            except Exception as e:
                logger.error(f"Error fetching {indicator}: {e}")
                results[indicator] = pd.DataFrame()
        
        return results
    
    async def get_market_sentiment_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get key market sentiment indicators
        """
        sentiment_indicators = [
            'vix',
            'consumer_sentiment',
            'consumer_confidence',
            'initial_claims'
        ]
        
        return await self.get_multiple_indicators(sentiment_indicators)
    
    async def get_monetary_policy_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get Federal Reserve and monetary policy indicators
        """
        monetary_indicators = [
            'fed_funds_rate',
            'treasury_10y',
            'treasury_2y',
            'treasury_3m',
            'm1_money_supply',
            'm2_money_supply'
        ]
        
        return await self.get_multiple_indicators(monetary_indicators)
    
    async def get_inflation_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get inflation-related indicators
        """
        inflation_indicators = [
            'cpi',
            'core_cpi',
            'pce',
            'core_pce'
        ]
        
        return await self.get_multiple_indicators(inflation_indicators)
    
    async def get_employment_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get employment-related indicators
        """
        employment_indicators = [
            'unemployment_rate',
            'nonfarm_payrolls',
            'labor_force_participation',
            'initial_claims'
        ]
        
        return await self.get_multiple_indicators(employment_indicators)
    
    async def get_economic_growth_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get economic growth indicators
        """
        growth_indicators = [
            'gdp',
            'gdp_growth',
            'industrial_production',
            'retail_sales'
        ]
        
        return await self.get_multiple_indicators(growth_indicators)
    
    def get_series_info(self, series_id: str) -> Dict:
        """
        Get information about a FRED series
        """
        try:
            info = self.fred.get_series_info(series_id)
            return info.to_dict()
        except Exception as e:
            logger.error(f"Error getting series info for {series_id}: {e}")
            return {}
    
    def search_series(self, search_text: str, limit: int = 10) -> pd.DataFrame:
        """
        Search for FRED series
        """
        try:
            results = self.fred.search(search_text, limit=limit)
            return results
        except Exception as e:
            logger.error(f"Error searching FRED series: {e}")
            return pd.DataFrame()
    
    async def get_yield_curve_data(self) -> pd.DataFrame:
        """
        Get current yield curve data
        """
        yield_indicators = {
            '3m': 'GS3M',
            '6m': 'GS6M',
            '1y': 'GS1',
            '2y': 'GS2',
            '3y': 'GS3',
            '5y': 'GS5',
            '7y': 'GS7',
            '10y': 'GS10',
            '20y': 'GS20',
            '30y': 'GS30'
        }
        
        curve_data = []
        for maturity, series_id in yield_indicators.items():
            try:
                data = self.fred.get_series(series_id, limit=1)
                if not data.empty:
                    curve_data.append({
                        'maturity': maturity,
                        'yield': data.iloc[-1],
                        'date': data.index[-1]
                    })
            except Exception as e:
                logger.error(f"Error fetching yield for {maturity}: {e}")
        
        return pd.DataFrame(curve_data)
    
    async def get_market_stress_indicators(self) -> Dict[str, float]:
        """
        Get current market stress indicators
        """
        stress_indicators = {
            'vix': 'VIXCLS',
            'term_spread': ['GS10', 'GS3M'],  # 10Y - 3M spread
            'credit_spread': 'BAMLC0A0CM',  # Corporate bond spread
            'dollar_strength': 'DTWEXBGS'
        }
        
        results = {}
        
        try:
            # VIX
            vix_data = self.fred.get_series('VIXCLS', limit=1)
            if not vix_data.empty:
                results['vix'] = vix_data.iloc[-1]
            
            # Term spread (10Y - 3M)
            gs10_data = self.fred.get_series('GS10', limit=1)
            gs3m_data = self.fred.get_series('GS3M', limit=1)
            if not gs10_data.empty and not gs3m_data.empty:
                results['term_spread'] = gs10_data.iloc[-1] - gs3m_data.iloc[-1]
            
            # Dollar index
            dollar_data = self.fred.get_series('DTWEXBGS', limit=1)
            if not dollar_data.empty:
                results['dollar_strength'] = dollar_data.iloc[-1]
                
        except Exception as e:
            logger.error(f"Error fetching market stress indicators: {e}")
        
        return results

# Global instance
fred_client = FREDClient()