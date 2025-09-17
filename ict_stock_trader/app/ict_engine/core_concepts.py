from typing import Dict, List, Any
import pandas as pd
import numpy as np

from ict_stock_trader.app.models.placeholder_types import (
    LiquidityPool, OrderBlock, BreakerBlock, FVG, RejectionBlock,
    MitigationBlock, SupplyDemandZone, DealingRange, SwingPoint,
    JudasSwing, TurtleSoup, OTE, SMTDivergence, LiquidityVoid
)

class StockSwingAnalyzer:
    def find_swing_points(self, stock_data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
        df = stock_data.copy()
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan
        for i in range(order, len(df) - order):
            # Strict definition: peak is higher than neighbors on both sides
            is_swing_high = all(df['High'].iloc[i] > df['High'].iloc[i-j] for j in range(1, order + 1)) and \
                            all(df['High'].iloc[i] > df['High'].iloc[i+j] for j in range(1, order + 1))
            if is_swing_high: df.loc[df.index[i], 'swing_high'] = df['High'].iloc[i]

            # Strict definition: trough is lower than neighbors on both sides
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
            if c3['Low'] > c1['High']:
                fvgs.append({"type": "Bullish", "timestamp": c2.name, "top": c3['Low'], "bottom": c1['High']})
            if c3['High'] < c1['Low']:
                fvgs.append({"type": "Bearish", "timestamp": c2.name, "top": c1['Low'], "bottom": c3['High']})
        return fvgs

    def concept_7_rejection_blocks(self, stock_data: pd.DataFrame, body_wick_ratio: float = 2.0) -> List[Dict[str, Any]]:
        rejection_blocks, df = [], stock_data.copy()
        for i in range(len(df)):
            candle = df.iloc[i]
            body_size = abs(candle['Open'] - candle['Close'])
            if body_size == 0: continue
            upper_wick = candle['High'] - max(candle['Open'], candle['Close'])
            lower_wick = min(candle['Open'], candle['Close']) - candle['Low']
            if upper_wick > body_size * body_wick_ratio:
                rejection_blocks.append({"type": "Bearish", "timestamp": candle.name, "price": candle['High']})
            if lower_wick > body_size * body_wick_ratio:
                rejection_blocks.append({"type": "Bullish", "timestamp": candle.name, "price": candle['Low']})
        return rejection_blocks

    def concept_8_mitigation_blocks(self, stock_data: pd.DataFrame) -> List[MitigationBlock]:
        return []

    def concept_9_supply_demand_zones(self, stock_data: pd.DataFrame, lookback: int = 5) -> List[Dict[str, Any]]:
        zones = []
        order_blocks = self.concept_4_order_blocks_bullish_bearish(stock_data, lookback=lookback)
        for ob in order_blocks:
            zones.append({"type": "Demand" if ob['type'] == 'Bullish' else 'Supply', "timestamp": ob['timestamp'], "top": ob['High'], "bottom": ob['Low']})
        return zones

    def concept_10_premium_discount_ote(self, stock_data: pd.DataFrame, range_high: float = None, range_low: float = None) -> Dict[str, Any]:
        if range_high is None: range_high = stock_data['High'].max()
        if range_low is None: range_low = stock_data['Low'].min()
        price_range = range_high - range_low
        equilibrium = range_low + (price_range / 2)
        sell_ote_62 = range_high - (price_range * 0.62)
        sell_ote_79 = range_high - (price_range * 0.79)
        buy_ote_62 = range_low + (price_range * 0.62)
        buy_ote_79 = range_low + (price_range * 0.79)
        return {
            "range_high": range_high, "range_low": range_low, "equilibrium": equilibrium,
            "premium_zone_top": range_high, "premium_zone_bottom": equilibrium,
            "discount_zone_top": equilibrium, "discount_zone_bottom": range_low,
            "sell_ote": {"62%": sell_ote_62, "79%": sell_ote_79},
            "buy_ote": {"62%": buy_ote_62, "79%": buy_ote_79}
        }

    def concept_11_dealing_ranges(self, stock_data: pd.DataFrame) -> List[DealingRange]:
        pass

    def concept_12_swing_highs_swing_lows(self, stock_data: pd.DataFrame) -> List[SwingPoint]:
        pass

    def concept_13_market_maker_buy_sell_models(self, stock_data: pd.DataFrame) -> Dict:
        pass

    def concept_14_market_maker_programs(self, stock_data: pd.DataFrame) -> Dict:
        pass

    def concept_15_judas_swing(self, stock_data: pd.DataFrame) -> List[JudasSwing]:
        pass

    def concept_16_turtle_soup(self, stock_data: pd.DataFrame) -> List[TurtleSoup]:
        pass

    def concept_17_power_of_3(self, stock_data: pd.DataFrame) -> Dict:
        pass

    def concept_18_optimal_trade_entry(self, stock_data: pd.DataFrame) -> List[OTE]:
        pass

    def concept_19_smt_divergence(self, stock_data: pd.DataFrame, correlated_stocks: List[str]) -> List[SMTDivergence]:
        pass

    def concept_20_liquidity_voids_inefficiencies(self, stock_data: pd.DataFrame) -> List[LiquidityVoid]:
        pass
