from typing import Dict, List, Any
import pandas as pd
import numpy as np

from ict_stock_trader.app.models.placeholder_types import (
    LiquidityPool, OrderBlock, BreakerBlock, FVG, RejectionBlock,
    MitigationBlock, SupplyDemandZone, DealingRange, SwingPoint,
    JudasSwing, TurtleSoup, OTE, SMTDivergence, LiquidityVoid
)
from ict_stock_trader.app.data.yfinance_client import StockDataManager

class StockSwingAnalyzer:
    def find_swing_points(self, stock_data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
        df = stock_data.copy()
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        for i in range(order, len(df) - order):
            is_swing_high = all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, order + 1)) and \
                            all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, order + 1))
            if is_swing_high: df.loc[df.index[i], 'swing_high'] = df['High'].iloc[i]
            is_swing_low = all(df['Low'].iloc[i] < df['Low'].iloc[i-j] for j in range(1, order + 1)) and \
                           all(df['Low'].iloc[i] < df['Low'].iloc[i+j] for j in range(1, order + 1))
            if is_swing_low: df.loc[df.index[i], 'swing_low'] = df['Low'].iloc[i]
        return df

class StockTrendAnalyzer:
    def determine_trend(self, swing_data: pd.DataFrame) -> str:
        swing_highs, swing_lows = swing_data['swing_high'].dropna(), swing_data['swing_low'].dropna()
        if len(swing_highs) < 2 or len(swing_lows) < 2: return "Ranging"
        if (swing_highs.iloc[-1] > swing_highs.iloc[-2]) and (swing_lows.iloc[-1] > swing_lows.iloc[-2]): return "Uptrend"
        if (swing_highs.iloc[-1] < swing_highs.iloc[-2]) and (swing_lows.iloc[-1] < swing_lows.iloc[-2]): return "Downtrend"
        return "Ranging"

class StockMarketStructureAnalyzer:
    def __init__(self):
        self.swing_analyzer = StockSwingAnalyzer()
        self.trend_analyzer = StockTrendAnalyzer()

    def concept_1_market_structure_hh_hl_lh_ll(self, stock_data: pd.DataFrame, order: int = 1) -> Dict[str, Any]:
        swing_data = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        swing_highs, swing_lows = swing_data['swing_high'].dropna(), swing_data['swing_low'].dropna()
        if len(swing_highs) < 2 or len(swing_lows) < 2: return {"market_structure": "Indeterminate", "structure_points": []}
        high_points = [{"timestamp": swing_highs.index[i], "price": swing_highs.iloc[i], "type": "HH" if swing_highs.iloc[i] > swing_highs.iloc[i-1] else "LH"} for i in range(1, len(swing_highs))]
        low_points = [{"timestamp": swing_lows.index[i], "price": swing_lows.iloc[i], "type": "HL" if swing_lows.iloc[i] > swing_lows.iloc[i-1] else "LL"} for i in range(1, len(swing_lows))]
        structure_points = sorted(high_points + low_points, key=lambda p: p['timestamp'])
        market_structure = "Ranging"
        if len(structure_points) >= 2:
            last_two_types = {p['type'] for p in structure_points[-2:]}
            if last_two_types == {"HH", "HL"}: market_structure = "Bullish"
            elif last_two_types == {"LH", "LL"}: market_structure = "Bearish"
        return {"market_structure": market_structure, "structure_points": structure_points}

    def concept_2_liquidity_buyside_sellside(self, stock_data: pd.DataFrame, order: int = 1) -> Dict[str, List[float]]:
        swing_data = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        return {"buy_side_liquidity": swing_data['swing_high'].dropna().tolist(), "sell_side_liquidity": swing_data['swing_low'].dropna().tolist()}

    def concept_3_liquidity_pools(self, stock_data: pd.DataFrame, order: int = 1, tolerance: float = 0.001) -> List[Dict[str, Any]]:
        swing_data = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        swing_highs, swing_lows = sorted(swing_data['swing_high'].dropna().tolist()), sorted(swing_data['swing_low'].dropna().tolist())
        liquidity_pools = []
        for i in range(len(swing_highs) - 1):
            if abs(swing_highs[i] - swing_highs[i+1]) <= swing_highs[i] * tolerance:
                liquidity_pools.append({"type": "Equal Highs", "level": (swing_highs[i] + swing_highs[i+1]) / 2})
        for i in range(len(swing_lows) - 1):
            if abs(swing_lows[i] - swing_lows[i+1]) <= swing_lows[i] * tolerance:
                liquidity_pools.append({"type": "Equal Lows", "level": (swing_lows[i] + swing_lows[i+1]) / 2})
        return liquidity_pools

    def concept_4_order_blocks_bullish_bearish(self, stock_data: pd.DataFrame, lookback: int = 5) -> List[Dict[str, Any]]:
        order_blocks, df = [], stock_data.copy()
        for i in range(1, len(df) - lookback):
            candle = df.iloc[i]
            subsequent_candles = df.iloc[i+1 : i+1+lookback]
            if candle['Close'] < candle['Open'] and any(subsequent_candles['High'] > candle['High']):
                order_blocks.append({"type": "Bullish", "timestamp": candle.name, "Open": candle['Open'], "High": candle['High'], "Low": candle['Low'], "Close": candle['Close']})
            elif candle['Close'] > candle['Open'] and any(subsequent_candles['Low'] < candle['Low']):
                order_blocks.append({"type": "Bearish", "timestamp": candle.name, "Open": candle['Open'], "High": candle['High'], "Low": candle['Low'], "Close": candle['Close']})
        return order_blocks

    def concept_5_breaker_blocks(self, stock_data: pd.DataFrame, lookback: int = 5) -> List[Dict[str, Any]]:
        breaker_blocks, df = [], stock_data.copy()
        for i in range(1, len(df) - lookback):
            candle = df.iloc[i]
            if candle['Close'] < candle['Open']:
                initial_move = df.iloc[i+1 : i+1+lookback]
                if any(initial_move['High'] > candle['High']):
                    future_candles = df.iloc[i+1:]
                    if any(future_candles['Low'] < candle['Low']):
                        breaker_blocks.append({"type": "Bearish Breaker", "timestamp": candle.name, "level": candle['Low']})
            elif candle['Close'] > candle['Open']:
                initial_move = df.iloc[i+1 : i+1+lookback]
                if any(initial_move['Low'] < candle['Low']):
                    future_candles = df.iloc[i+1:]
                    if any(future_candles['High'] > candle['High']):
                        breaker_blocks.append({"type": "Bullish Breaker", "timestamp": candle.name, "level": candle['High']})
        return breaker_blocks

    def concept_6_fair_value_gaps_fvg_imbalances(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        fvgs, df = [], stock_data.copy()
        for i in range(len(df) - 2):
            c1, c2, c3 = df.iloc[i], df.iloc[i+1], df.iloc[i+2]
            if c3['Low'] > c1['High']: fvgs.append({"type": "Bullish", "timestamp": c2.name, "top": c3['Low'], "bottom": c1['High']})
            if c3['High'] < c1['Low']: fvgs.append({"type": "Bearish", "timestamp": c2.name, "top": c1['Low'], "bottom": c3['High']})
        return fvgs

    def concept_7_rejection_blocks(self, stock_data: pd.DataFrame, body_wick_ratio: float = 2.0) -> List[Dict[str, Any]]:
        rejection_blocks, df = [], stock_data.copy()
        for i in range(len(df)):
            candle = df.iloc[i]
            body_size = abs(candle['Open'] - candle['Close'])
            if body_size == 0: continue
            upper_wick, lower_wick = candle['High'] - max(candle['Open'], candle['Close']), min(candle['Open'], candle['Close']) - candle['Low']
            if upper_wick > body_size * body_wick_ratio: rejection_blocks.append({"type": "Bearish", "timestamp": candle.name, "price": candle['High']})
            if lower_wick > body_size * body_wick_ratio: rejection_blocks.append({"type": "Bullish", "timestamp": candle.name, "price": candle['Low']})
        return rejection_blocks

    def concept_8_mitigation_blocks(self, stock_data: pd.DataFrame) -> List[MitigationBlock]: return []

    def concept_9_supply_demand_zones(self, stock_data: pd.DataFrame, lookback: int = 5) -> List[Dict[str, Any]]:
        zones = []
        for ob in self.concept_4_order_blocks_bullish_bearish(stock_data, lookback=lookback):
            zones.append({"type": "Demand" if ob['type'] == 'Bullish' else 'Supply', "timestamp": ob['timestamp'], "top": ob['High'], "bottom": ob['Low']})
        return zones

    def concept_10_premium_discount_ote(self, stock_data: pd.DataFrame, range_high: float = None, range_low: float = None) -> Dict[str, Any]:
        if range_high is None: range_high = stock_data['High'].max()
        if range_low is None: range_low = stock_data['Low'].min()
        price_range = range_high - range_low
        equilibrium = range_low + (price_range / 2)
        return {"range_high": range_high, "range_low": range_low, "equilibrium": equilibrium, "sell_ote": {"62%": range_high - (price_range * 0.62), "79%": range_high - (price_range * 0.79)}, "buy_ote": {"62%": range_low + (price_range * 0.62), "79%": range_low + (price_range * 0.79)}}

    def concept_11_dealing_ranges(self, stock_data: pd.DataFrame, lookback: int = 20) -> List[Dict[str, Any]]:
        high, low = stock_data['High'].rolling(window=lookback).max(), stock_data['Low'].rolling(window=lookback).min()
        return [{"timestamp": stock_data.index[i], "high": high.iloc[i], "low": low.iloc[i]} for i in range(len(stock_data)) if pd.notna(high.iloc[i])]

    def concept_12_swing_highs_swing_lows(self, stock_data: pd.DataFrame, order: int = 1) -> List[Dict[str, Any]]:
        swing_data = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        highs = swing_data[swing_data['swing_high'].notna()]
        lows = swing_data[swing_data['swing_low'].notna()]
        return [{"type": "high", "timestamp": r.Index, "price": r.swing_high} for r in highs.itertuples()] + [{"type": "low", "timestamp": r.Index, "price": r.swing_low} for r in lows.itertuples()]

    def concept_13_market_maker_buy_sell_models(self, stock_data: pd.DataFrame) -> Dict: return {}

    def concept_14_market_maker_programs(self, stock_data: pd.DataFrame) -> Dict: return {}

    def concept_15_judas_swing(self, stock_data: pd.DataFrame) -> List[JudasSwing]: return []

    def concept_16_turtle_soup(self, stock_data: pd.DataFrame, period: int = 20) -> List[Dict[str, Any]]:
        df = stock_data.copy()
        df['20d_high'] = df['High'].rolling(window=period).max().shift(1)
        df['20d_low'] = df['Low'].rolling(window=period).min().shift(1)
        buy_soup = df[(df['Low'] < df['20d_low']) & (df['Close'] > df['20d_low'])]
        sell_soup = df[(df['High'] > df['20d_high']) & (df['Close'] < df['20d_high'])]
        return [{"type": "buy", "timestamp": r.Index} for r in buy_soup.itertuples()] + [{"type": "sell", "timestamp": r.Index} for r in sell_soup.itertuples()]

    def concept_17_power_of_3(self, stock_data: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(stock_data)):
            candle = stock_data.iloc[i]
            is_amd = candle['Close'] > candle['Open'] and candle['Low'] < candle['Open'] and candle['High'] > candle['Close']
            if is_amd: results.append({"type": "Bullish AMD", "timestamp": candle.name})
            is_dma = candle['Close'] < candle['Open'] and candle['High'] > candle['Open'] and candle['Low'] < candle['Close']
            if is_dma: results.append({"type": "Bearish DMA", "timestamp": candle.name})
        return results

    def concept_18_optimal_trade_entry(self, stock_data: pd.DataFrame, order: int = 5) -> List[Dict[str, Any]]:
        swing_data = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        swing_highs, swing_lows = swing_data['swing_high'].dropna(), swing_data['swing_low'].dropna()
        if len(swing_highs) < 1 or len(swing_lows) < 1: return []
        recent_high, recent_low = swing_highs.iloc[-1], swing_lows.iloc[-1]
        ote_info = self.concept_10_premium_discount_ote(stock_data, range_high=recent_high, range_low=recent_low)
        return [ote_info]

    def concept_19_smt_divergence(self, stock_data: pd.DataFrame, correlated_symbol: str, order: int = 5) -> List[Dict[str, Any]]:
        data_manager = StockDataManager()
        other_data = data_manager.get_real_time_stock_data(correlated_symbol, period=f"{len(stock_data)}d", interval="1d")
        if len(other_data) != len(stock_data): return []
        swing_data1 = self.swing_analyzer.find_swing_points(stock_data.copy(), order=order)
        swing_data2 = self.swing_analyzer.find_swing_points(other_data.copy(), order=order)
        lows1, lows2 = swing_data1['swing_low'].dropna(), swing_data2['swing_low'].dropna()
        if len(lows1) < 2 or len(lows2) < 2: return []
        if (lows1.iloc[-1] < lows1.iloc[-2]) and (lows2.iloc[-1] > lows2.iloc[-2]):
            return [{"type": "Bullish SMT", "timestamp": lows1.index[-1]}]
        return []

    def concept_20_liquidity_voids_inefficiencies(self, stock_data: pd.DataFrame, min_gap_size: float = 0.01) -> List[Dict[str, Any]]:
        voids, df = [], stock_data.copy()
        for i in range(len(df) - 1):
            c1, c2 = df.iloc[i], df.iloc[i+1]
            gap_size = c2['Low'] - c1['High']
            if gap_size > (c1['High'] * min_gap_size):
                voids.append({"type": "Bullish", "timestamp": c1.name, "top": c2['Low'], "bottom": c1['High']})
        return voids
