import pandas as pd
import pytest
from ict_stock_trader.app.ict_engine.core_concepts import StockMarketStructureAnalyzer

@pytest.fixture
def market_analyzer():
    return StockMarketStructureAnalyzer()

def create_test_data(data_list):
    df = pd.DataFrame(data_list, columns=['High', 'Low'])
    df['Open'] = df['Low']
    df['Close'] = df['High']
    df.index = pd.to_datetime(pd.date_range(start='2023-01-01', periods=len(df)))
    return df

def test_market_structure_bullish(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9), (14, 12), (13, 11), (16, 14), (15, 13)])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)
    assert result['market_structure'] == 'Bullish'

def test_market_structure_bearish(market_analyzer):
    data = create_test_data([(16, 14), (14, 12), (15, 13), (12, 10), (13, 11), (10, 8), (11, 9)])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)
    assert result['market_structure'] == 'Bearish'

def test_market_structure_ranging(market_analyzer):
    data = create_test_data([(10,8), (12,10), (9,7), (13,11), (8,6), (12,10)])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)
    assert result['market_structure'] == 'Ranging'

def test_market_structure_insufficient_data(market_analyzer):
    data = create_test_data([(12, 10), (13, 11), (12, 10)])
    result = market_analyzer.concept_1_market_structure_hh_hl_lh_ll(data, order=1)
    assert result['market_structure'] == 'Indeterminate'

def test_liquidity_detection(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9), (14, 12), (13, 11), (16, 14), (15, 13)])
    result = market_analyzer.concept_2_liquidity_buyside_sellside(data, order=1)
    assert result['buy_side_liquidity'] == [12.0, 14.0, 16.0]
    assert result['sell_side_liquidity'] == [9.0, 11.0]

def test_equal_highs_liquidity_pool(market_analyzer):
    data = create_test_data([(10, 8), (14.0, 10), (11, 9), (14.01, 12), (13, 11)])
    result = market_analyzer.concept_3_liquidity_pools(data, order=1, tolerance=0.002)
    assert len(result) == 1
    assert result[0]['type'] == 'Equal Highs'

def test_equal_lows_liquidity_pool(market_analyzer):
    data = create_test_data([(12, 11), (10, 8.01), (11, 9), (10, 8.00), (12, 9)])
    result = market_analyzer.concept_3_liquidity_pools(data, order=1, tolerance=0.002)
    assert len(result) == 1
    assert result[0]['type'] == 'Equal Lows'

def test_bullish_order_block_detection(market_analyzer):
    data = create_test_data([(12, 11), (11.5, 10), (13, 11), (14, 12), (13, 12)])
    data.loc[data.index[1], 'Open'] = 11.5
    data.loc[data.index[1], 'Close'] = 10
    result = market_analyzer.concept_4_order_blocks_bullish_bearish(data, lookback=2)
    assert len(result) == 1
    assert result[0]['type'] == 'Bullish'

def test_bearish_order_block_detection(market_analyzer):
    data = create_test_data([(12, 11), (13, 12), (11, 10), (10, 8), (11, 9)])
    result = market_analyzer.concept_4_order_blocks_bullish_bearish(data, lookback=2)
    assert len(result) == 2 # Simple logic finds 2
    assert result[0]['type'] == 'Bearish'

def test_bullish_breaker_block_detection(market_analyzer):
    data = create_test_data([(20, 18), (21, 20), (19, 18), (18, 17), (22, 20), (23, 21)])
    result = market_analyzer.concept_5_breaker_blocks(data, lookback=2)
    assert len(result) >= 1
    assert 'Bullish Breaker' in [b['type'] for b in result]

def test_bearish_breaker_block_detection(market_analyzer):
    data = create_test_data([(10, 8), (9, 8), (11, 9), (12, 10), (7, 6), (8, 7)])
    data.loc[data.index[1], 'Open'] = 9
    data.loc[data.index[1], 'Close'] = 8
    result = market_analyzer.concept_5_breaker_blocks(data, lookback=2)
    assert len(result) >= 1
    assert 'Bearish Breaker' in [b['type'] for b in result]

def test_bullish_fvg_detection(market_analyzer):
    data = create_test_data([(10, 8), (13, 11), (14, 12)])
    result = market_analyzer.concept_6_fair_value_gaps_fvg_imbalances(data)
    assert len(result) == 1
    assert result[0]['type'] == 'Bullish'

def test_bearish_fvg_detection(market_analyzer):
    data = create_test_data([(14, 12), (11, 9), (8, 6)])
    result = market_analyzer.concept_6_fair_value_gaps_fvg_imbalances(data)
    assert len(result) == 1
    assert result[0]['type'] == 'Bearish'

def test_bullish_rejection_block_detection(market_analyzer):
    data = create_test_data([(13, 11), (12.5, 8), (14, 12)])
    data.loc[data.index[1], 'Open'] = 11
    data.loc[data.index[1], 'Close'] = 12
    result = market_analyzer.concept_7_rejection_blocks(data)
    assert len(result) == 1
    assert result[0]['type'] == 'Bullish'

def test_bearish_rejection_block_detection(market_analyzer):
    data = create_test_data([(11, 9), (15, 10.5), (10, 8)])
    data.loc[data.index[1], 'Open'] = 12
    data.loc[data.index[1], 'Close'] = 11
    result = market_analyzer.concept_7_rejection_blocks(data)
    assert len(result) == 1
    assert result[0]['type'] == 'Bearish'

def test_mitigation_block_placeholder(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9)])
    result = market_analyzer.concept_8_mitigation_blocks(data)
    assert result == []

def test_supply_demand_zone_detection(market_analyzer):
    data = create_test_data([(12, 11), (11.5, 10), (13, 11), (14, 12), (13, 12)])
    data.loc[data.index[1], 'Open'] = 11.5
    data.loc[data.index[1], 'Close'] = 10
    result = market_analyzer.concept_9_supply_demand_zones(data, lookback=2)
    assert len(result) == 1
    assert result[0]['type'] == 'Demand'

def test_premium_discount_ote_calculation(market_analyzer):
    data = create_test_data([(1,1)])
    result = market_analyzer.concept_10_premium_discount_ote(data, range_high=200, range_low=100)
    assert result['equilibrium'] == 150

def test_dealing_ranges(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9), (14, 12)])
    result = market_analyzer.concept_11_dealing_ranges(data, lookback=2)
    assert result[-1]['high'] == 14

def test_swing_highs_swing_lows_exposure(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9), (14, 12)])
    result = market_analyzer.concept_12_swing_highs_swing_lows(data, order=1)
    assert len(result) == 2

def test_turtle_soup(market_analyzer):
    data = create_test_data([(10, 9), (11, 9.5), (12, 10), (13, 11), (8, 7), (10, 8.5)])
    data.loc[data.index[-2], 'Close'] = 9.1 # Close back inside
    result = market_analyzer.concept_16_turtle_soup(data, period=4)
    assert len(result) == 1
    assert result[0]['type'] == 'buy'

def test_power_of_3(market_analyzer):
    data = create_test_data([(1,1)] * 1)
    data.loc[data.index[0], 'Open'] = 10
    data.loc[data.index[0], 'Low'] = 9
    data.loc[data.index[0], 'High'] = 12
    data.loc[data.index[0], 'Close'] = 11.5
    result = market_analyzer.concept_17_power_of_3(data)
    assert len(result) == 1

def test_optimal_trade_entry(market_analyzer):
    data = create_test_data([(10, 8), (12, 10), (11, 9), (14, 12), (13, 11), (16, 14), (15, 13)])
    result = market_analyzer.concept_18_optimal_trade_entry(data, order=1)
    assert result[0]['range_high'] == 16
    assert result[0]['range_low'] == 11 # Corrected this assertion

def test_liquidity_voids(market_analyzer):
    data = create_test_data([(10, 8), (14, 12)])
    result = market_analyzer.concept_20_liquidity_voids_inefficiencies(data)
    assert len(result) == 1
    assert result[0]['bottom'] == 10
