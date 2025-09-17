import pandas as pd
import pytest
from ict_stock_trader.app.ict_engine.core_concepts import StockMarketStructureAnalyzer

@pytest.fixture
def market_analyzer():
    """Provides an instance of the StockMarketStructureAnalyzer."""
    return StockMarketStructureAnalyzer()

def create_test_data(data_list):
    """Helper function to create a DataFrame from a list of High/Low tuples."""
    df = pd.DataFrame(data_list, columns=['High', 'Low'])
    df['Open'] = df['Low']
    df['Close'] = df['High']
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(df)))
    return df

def test_market_structure_bullish(market_analyzer):
    """
    Tests a clear bullish market structure (HH, HL).
    """
    data = create_test_data([
        (10, 8),
        (12, 10), # Swing High 1
        (11, 9),  # Swing Low 1
        (14, 12), # Swing High 2
        (13, 11), # Swing Low 2
        (16, 14), # Swing High 3
        (15, 13)
    ])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)

    assert result['market_structure'] == 'Bullish'
    point_types = [p['type'] for p in result['structure_points']]
    # Correct assertion based on the actual logic trace
    assert point_types == ['HH', 'HL', 'HH']

def test_market_structure_bearish(market_analyzer):
    """
    Tests a clear bearish market structure (LH, LL).
    """
    data = create_test_data([
        (16, 14),
        (14, 12), # Swing Low 1
        (15, 13), # Swing High 1
        (12, 10), # Swing Low 2
        (13, 11), # Swing High 2
        (10, 8),  # Swing Low 3
        (11, 9)
    ])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)

    assert result['market_structure'] == 'Bearish'
    point_types = [p['type'] for p in result['structure_points']]
    # Correct assertion based on the actual logic trace
    assert point_types == ['LL', 'LH', 'LL']

def test_market_structure_ranging(market_analyzer):
    """
    Tests a ranging market where the trend is mixed.
    """
    data = create_test_data([
       (10,8),
       (12,10), # H
       (9,7),   # L
       (13,11), # H
       (8,6),   # L
       (12,10)
    ])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)
    # Highs: 12, 13 -> HH
    # Lows: 7, 6 -> LL
    # Last two points are LL and HH, which is ranging.
    assert result['market_structure'] == 'Ranging'

def test_market_structure_insufficient_data(market_analyzer):
    """
    Tests handling of data with too few points to determine structure.
    """
    data = create_test_data([
        (12, 10), (13, 11), (12, 10)
    ])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)

    assert result['market_structure'] == 'Indeterminate'
    assert result['structure_points'] == []

def test_liquidity_detection(market_analyzer):
    """
    Tests the identification of buy-side and sell-side liquidity pools.
    """
    data = create_test_data([
        (10, 8),
        (12, 10), # Swing High
        (11, 9),  # Swing Low
        (14, 12), # Swing High
        (13, 11), # Swing Low
        (16, 14), # Swing High
        (15, 13)
    ])

    result = market_analyzer.concept_2_liquidity_buyside_sellside(data, order=1)

    # Expected swing highs at 12, 14, 16
    assert result['buy_side_liquidity'] == [12.0, 14.0, 16.0]
    # Expected swing lows at 9, 11
    assert result['sell_side_liquidity'] == [9.0, 11.0]
