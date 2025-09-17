from typing import Dict, List, Any
import pandas as pd
import numpy as np

from ict_stock_trader.app.models.placeholder_types import LiquidityRaid

class StockTimeAndPriceAnalyzer:

    def concept_21_stock_killzones(self, stock_data: pd.DataFrame) -> Dict[str, bool]:
        if not isinstance(stock_data.index, pd.DatetimeIndex): return {"error": "Index must be a DatetimeIndex."}
        if stock_data.index.tz is None: stock_data.index = stock_data.index.tz_localize('UTC')
        last_time_et = stock_data.index[-1].tz_convert('US/Eastern')
        time_et = last_time_et.time()
        is_pre_market = pd.to_datetime('04:00').time() <= time_et < pd.to_datetime('09:30').time()
        is_market_open = pd.to_datetime('09:30').time() <= time_et < pd.to_datetime('11:00').time()
        is_power_hour = pd.to_datetime('15:00').time() <= time_et < pd.to_datetime('16:00').time()
        return {"is_pre_market_killzone": is_pre_market, "is_market_open_killzone": is_market_open, "is_power_hour_killzone": is_power_hour}

    def concept_22_stock_session_opens(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        if not isinstance(stock_data.index, pd.DatetimeIndex): return {}
        if stock_data.index.tz is None: stock_data.index = stock_data.index.tz_localize('UTC')
        day_data = stock_data.loc[stock_data.index.date == stock_data.index[-1].date()]
        day_data_et = day_data.tz_convert('US/Eastern')
        pre_market_open = day_data_et.between_time('04:00', '04:01')
        market_open = day_data_et.between_time('09:30', '09:31')
        return {"pre_market_open": pre_market_open['Open'].iloc[0] if not pre_market_open.empty else None, "market_open": market_open['Open'].iloc[0] if not market_open.empty else None}

    def concept_23_fibonacci_ratios(self, high: float, low: float) -> Dict[str, float]:
        price_range = high - low
        return {"50%": high - price_range * 0.5, "62%": high - price_range * 0.62, "70.5%": high - price_range * 0.705, "79%": high - price_range * 0.79}

    def concept_24_daily_weekly_range_expectations(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        if 'High' not in stock_data or 'Low' not in stock_data or 'Close' not in stock_data: return {}
        atr = (stock_data['High'] - stock_data['Low']).mean()
        return {"average_daily_range": atr, "projected_high": stock_data['Close'].iloc[-1] + atr, "projected_low": stock_data['Close'].iloc[-1] - atr}

    def concept_25_session_liquidity_raids(self, stock_data: pd.DataFrame) -> List[LiquidityRaid]:
        return []

    def concept_26_weekly_profiles(self, stock_data: pd.DataFrame) -> Dict[str, float]:
        if not isinstance(stock_data.index, pd.DatetimeIndex): return {}
        if stock_data.index.tz is None: stock_data.index = stock_data.index.tz_localize('UTC')
        last_day = stock_data.index[-1]
        start_of_week = pd.Timestamp((last_day - pd.to_timedelta(last_day.weekday(), unit='D')).date(), tz='UTC')
        week_data = stock_data.loc[start_of_week:last_day]
        if week_data.empty: return {}
        return {"week_open": week_data['Open'].iloc[0], "week_high": week_data['High'].max(), "week_low": week_data['Low'].min(), "week_close": week_data['Close'].iloc[-1]}

    def concept_27_daily_bias(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        if len(stock_data) < 2: return {}
        today, yesterday = stock_data.iloc[-1], stock_data.iloc[-2]
        bias = "Neutral"
        if today['Close'] > yesterday['High']: bias = "Bullish"
        if today['Close'] < yesterday['Low']: bias = "Bearish"
        return {"bias": bias, "previous_high": yesterday['High'], "previous_low": yesterday['Low']}

    def concept_28_weekly_bias(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        weekly_profile = self.concept_26_weekly_profiles(stock_data)
        if not weekly_profile: return {}
        bias = "Neutral"
        if weekly_profile['week_close'] > weekly_profile['week_open']: bias = "Bullish"
        if weekly_profile['week_close'] < weekly_profile['week_open']: bias = "Bearish"
        return {"bias": bias}

    def concept_29_monthly_bias(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(stock_data.index, pd.DatetimeIndex): return {}
        if stock_data.index.tz is None: stock_data.index = stock_data.index.tz_localize('UTC')
        last_day = stock_data.index[-1]
        start_of_month = pd.Timestamp(last_day.replace(day=1).date(), tz='UTC')
        month_data = stock_data.loc[start_of_month:last_day]
        if month_data.empty: return {}
        bias = "Neutral"
        if month_data['Close'].iloc[-1] > month_data['Open'].iloc[0]: bias = "Bullish"
        if month_data['Close'].iloc[-1] < month_data['Open'].iloc[0]: bias = "Bearish"
        return {"bias": bias}

    def concept_30_time_of_day_highs_lows(self, stock_data: pd.DataFrame) -> Dict[str, Any]:
        if not isinstance(stock_data.index, pd.DatetimeIndex): return {}
        if stock_data.index.tz is None: stock_data.index = stock_data.index.tz_localize('UTC')
        day_data_et = stock_data.loc[stock_data.index.date == stock_data.index[-1].date()].tz_convert('US/Eastern')
        am_session = day_data_et.between_time('09:30', '12:00')
        pm_session = day_data_et.between_time('13:00', '16:00')
        return {"am_high": am_session['High'].max() if not am_session.empty else None, "am_low": am_session['Low'].min() if not am_session.empty else None, "pm_high": pm_session['High'].max() if not pm_session.empty else None, "pm_low": pm_session['Low'].min() if not pm_session.empty else None}
