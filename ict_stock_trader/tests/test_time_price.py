import pandas as pd
import pytest
from ict_stock_trader.app.ict_engine.time_price import StockTimeAndPriceAnalyzer

@pytest.fixture
def time_analyzer():
    """Provides an instance of the StockTimeAndPriceAnalyzer."""
    return StockTimeAndPriceAnalyzer()

def create_time_series_data(start_time_utc, periods, freq='h'):
    """Helper to create time-series data."""
    dates = pd.to_datetime(pd.date_range(start=start_time_utc, periods=periods, freq=freq, tz='UTC'))
    data = {'Open': range(100, 100 + periods), 'High': range(101, 101 + periods), 'Low': range(99, 99 + periods), 'Close': range(102, 102 + periods)}
    return pd.DataFrame(data, index=dates)

def test_stock_killzones(time_analyzer):
    # 9:30 AM ET is 13:30 UTC during DST
    data = create_time_series_data('2023-10-26 13:30:00', 1, 'min')
    result = time_analyzer.concept_21_stock_killzones(data)
    assert result['is_market_open_killzone'] == True
    assert result['is_power_hour_killzone'] == False

def test_fibonacci_ratios(time_analyzer):
    result = time_analyzer.concept_23_fibonacci_ratios(high=200, low=100)
    assert result['50%'] == 150
    assert result['62%'] == pytest.approx(138.0)

def test_daily_weekly_range_expectations(time_analyzer):
    data = create_time_series_data('2023-10-26 10:00:00', 5, 'h')
    result = time_analyzer.concept_24_daily_weekly_range_expectations(data)
    assert 'average_daily_range' in result

def test_weekly_profiles(time_analyzer):
    data = create_time_series_data('2023-10-23 10:00:00', 5*24, 'h') # 5 days
    result = time_analyzer.concept_26_weekly_profiles(data)
    assert result['week_open'] == 100
    assert result['week_high'] > 100

def test_daily_bias(time_analyzer):
    data = create_time_series_data('2023-10-25 10:00:00', 2, 'd')
    # Make today's close higher than yesterday's high for a clear bullish bias
    data.loc[data.index[1], 'Close'] = data['High'].iloc[0] + 1
    result = time_analyzer.concept_27_daily_bias(data)
    assert result['bias'] == 'Bullish'

def test_weekly_bias(time_analyzer):
    data = create_time_series_data('2023-10-23 10:00:00', 5, 'd')
    result = time_analyzer.concept_28_weekly_bias(data)
    assert result['bias'] == 'Bullish'

def test_monthly_bias(time_analyzer):
    data = create_time_series_data('2023-10-01 10:00:00', 26, 'd')
    result = time_analyzer.concept_29_monthly_bias(data)
    assert result['bias'] == 'Bullish'

def test_time_of_day_highs_lows(time_analyzer):
    # 10:30 AM ET = 14:30 UTC, 1:30 PM ET = 17:30 UTC
    data = create_time_series_data('2023-10-26 00:00:00', 24, 'h')
    result = time_analyzer.concept_30_time_of_day_highs_lows(data)
    assert result['am_high'] is not None
    assert result['pm_low'] is not None
