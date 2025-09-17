import pandas as pd
import pytest
from ict_stock_trader.app.ict_engine.advanced_concepts import StockAdvancedConceptsEngine

@pytest.fixture
def advanced_analyzer():
    """Provides an instance of the StockAdvancedConceptsEngine."""
    return StockAdvancedConceptsEngine()

def create_daily_data(data_list):
    """Helper to create daily time-series data."""
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(data_list)))
    return pd.DataFrame(data_list, columns=['High', 'Low', 'Open', 'Close'], index=dates)

def test_liquidity_runs(advanced_analyzer):
    data = create_daily_data([
        (12, 10, 11, 11),
        (11, 9.5, 10, 10), # Swing Low at 9.5
        (13, 11, 12, 12),
        (12, 9.0, 11, 11)  # Takes out the low of 9.5
    ])
    result = advanced_analyzer.concept_41_liquidity_runs(data)
    assert len(result) > 0
    assert result[0]['type'] == 'Sellside Run'

def test_high_low_of_day(advanced_analyzer):
    data = create_daily_data([(15, 8, 10, 14), (16, 9, 14, 15)])
    result = advanced_analyzer.concept_45_high_low_day_identification(data)
    assert result['high_of_day'] == 16
    assert result['low_of_day'] == 8

def test_range_expansion(advanced_analyzer):
    data = create_daily_data([
        (11, 10, 10, 11), # Range 1
        (12, 10, 10, 12), # Range 2
        (14, 9, 9, 14)    # Range 5 (Expansion)
    ])
    result = advanced_analyzer.concept_46_range_expansion(data, lookback=2)
    assert len(result) == 1
    assert result[0]['range'] == 5

def test_inside_outside_days(advanced_analyzer):
    data = create_daily_data([
        (15, 10, 11, 14), # Day 1
        (14, 11, 12, 13), # Inside Day
        (16, 9, 10, 15)   # Outside Day
    ])
    result = advanced_analyzer.concept_47_inside_outside_days(data)
    assert len(result) == 2
    types = {d['type'] for d in result}
    assert "Inside Day" in types
    assert "Outside Day" in types
