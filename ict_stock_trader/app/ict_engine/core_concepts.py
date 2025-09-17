"""
ICT Core Concepts Implementation (Concepts 1-20)
Market Structure, Liquidity, Order Blocks, Fair Value Gaps, and more
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class MarketStructure(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL" 
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EH"
    EQUAL_LOW = "EL"

@dataclass
class SwingPoint:
    timestamp: datetime
    price: float
    swing_type: str  # 'high' or 'low'
    index: int
    strength: float = 0.0

@dataclass
class LiquidityPool:
    timestamp: datetime
    price_level: float
    pool_type: str  # 'buyside', 'sellside', 'equal_highs', 'equal_lows'
    strength: float
    volume: int = 0
    touches: int = 1

@dataclass
class OrderBlock:
    timestamp: datetime
    high_price: float
    low_price: float
    block_type: str  # 'bullish', 'bearish'
    formation_candle_index: int
    strength: float
    is_breaker: bool = False

@dataclass
class FairValueGap:
    timestamp: datetime
    gap_high: float
    gap_low: float
    gap_type: str  # 'bullish', 'bearish'
    gap_size: float
    mitigation_level: float
    is_mitigated: bool = False

class StockMarketStructureAnalyzer:
    def __init__(self):
        self.swing_lookback = 5  # Periods to look back for swing identification
        self.liquidity_threshold = 0.001  # 0.1% for equal levels
        self.min_volume_ratio = 1.5  # Minimum volume ratio for significance
        
    def concept_1_market_structure_hh_hl_lh_ll(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 1: Market Structure (HH, HL, LH, LL)
        - Higher Highs (HH) detection with stock-specific logic
        - Higher Lows (HL) pattern recognition for stock trends
        - Lower Highs (LH) identification in stock downtrends  
        - Lower Lows (LL) pattern detection
        """
        if stock_data.empty or len(stock_data) < 10:
            return {'error': 'Insufficient data for market structure analysis'}
        
        try:
            # Identify swing points
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            # Classify market structure
            structure_analysis = {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'current_structure': self._classify_current_structure(swing_highs, swing_lows),
                'structure_breaks': self._detect_structure_breaks(swing_highs, swing_lows),
                'trend_direction': self._determine_trend_direction(swing_highs, swing_lows),
                'confidence': self._calculate_structure_confidence(swing_highs, swing_lows)
            }
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {'error': str(e)}
    
    def concept_2_liquidity_buyside_sellside(self, stock_data: pd.DataFrame) -> Dict:
        """
        CONCEPT 2: Liquidity (buy-side & sell-side)
        - Buy-side liquidity above stock resistance levels
        - Sell-side liquidity below stock support levels
        - Institutional liquidity pools in stocks
        """
        try:
            swing_highs = self._find_swing_highs(stock_data)
            swing_lows = self._find_swing_lows(stock_data)
            
            buyside_liquidity = []
            sellside_liquidity = []
            
            # Identify buy-side liquidity (above recent highs)
            for high in swing_highs[-5:]:  # Last 5 swing highs
                if self._is_significant_level(stock_data, high.price):
                    buyside_liquidity.append({
                        'price_level': high.price,
                        'timestamp': high.timestamp,
                        'strength': high.strength,
                        'type': 'buyside',
                        'distance_from_current': abs(high.price - stock_data['close'].iloc[-1]) / stock_data['close'].iloc[-1]
                    })
            
            # Identify sell-side liquidity (below recent lows)
            for low in swing_lows[-5:]:  # Last 5 swing lows
                if self._is_significant_level(stock_data, low.price):
                    sellside_liquidity.append({
                        'price_level': low.price,
                        'timestamp': low.timestamp,
                        'strength': low.strength,
                        'type': 'sellside',
                        'distance_from_current': abs(low.price - stock_data['close'].iloc[-1]) / stock_data['close'].iloc[-1]
                    })
            
            return {
                'buyside_liquidity': sorted(buyside_liquidity, key=lambda x: x['strength'], reverse=True),
                'sellside_liquidity': sorted(sellside_liquidity, key=lambda x: x['strength'], reverse=True),
                'nearest_buyside': min(buyside_liquidity, key=lambda x: x['distance_from_current']) if buyside_liquidity else None,
                'nearest_sellside': min(sellside_liquidity, key=lambda x: x['distance_from_current']) if sellside_liquidity else None,
                'liquidity_balance': len(buyside_liquidity) - len(sellside_liquidity)
            }
            
        except Exception as e:
            logger.error(f"Error in liquidity analysis: {e}")
            return {'error': str(e)}
    
    def concept_3_liquidity_pools(self, stock_data: pd.DataFrame) -> List[LiquidityPool]:
        """
        CONCEPT 3: Liquidity Pools (equal highs/lows, trendline liquidity)
        - Equal highs/lows in stock price action
        - Trendline liquidity for stock breakouts
        - Relative Equal Highs/Lows (REH/REL) in stocks
        """
        try:
            pools = []
            
            # Find equal highs
            swing_highs = self._find_swing_highs(stock_data)
            equal_highs = self._find_equal_levels(swing_highs, 'high')
            
            for level_group in equal_highs:
                if len(level_group) >= 2:
                    avg_price = np.mean([point.price for point in level_group])
                    pools.append(LiquidityPool(
                        timestamp=level_group[-1].timestamp,
                        price_level=avg_price,
                        pool_type='equal_highs',
                        strength=len(level_group) * 0.3,  # More touches = stronger
                        touches=len(level_group)
                    ))
            
            # Find equal lows
            swing_lows = self._find_swing_lows(stock_data)
            equal_lows = self._find_equal_levels(swing_lows, 'low')
            
            for level_group in equal_lows:
                if len(level_group) >= 2:
                    avg_price = np.mean([point.price for point in level_group])
                    pools.append(LiquidityPool(
                        timestamp=level_group[-1].timestamp,
                        price_level=avg_price,
                        pool_type='equal_lows',
                        strength=len(level_group) * 0.3,
                        touches=len(level_group)
                    ))
            
            # Find trendline liquidity
            trendline_pools = self._find_trendline_liquidity(stock_data)
            pools.extend(trendline_pools)
            
            return sorted(pools, key=lambda x: x.strength, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in liquidity pools analysis: {e}")
            return []
    
    def concept_4_order_blocks_bullish_bearish(self, stock_data: pd.DataFrame) -> List[OrderBlock]:
        """
        CONCEPT 4: Order Blocks (Bullish & Bearish)
        - Institutional order blocks in stock movements
        - Last down candle before stock impulse up
        - Last up candle before stock impulse down
        """
        try:
            order_blocks = []
            
            if len(stock_data) < 10:
                return order_blocks
            
            # Look for bullish order blocks (last down candle before impulse up)
            for i in range(3, len(stock_data) - 3):
                # Check if current candle is bearish and followed by bullish impulse
                current_candle = stock_data.iloc[i]
                
                if current_candle['close'] < current_candle['open']:  # Bearish candle
                    # Check for bullish impulse in next 3 candles
                    next_candles = stock_data.iloc[i+1:i+4]
                    impulse_strength = (next_candles['close'].max() - current_candle['low']) / current_candle['low']
                    
                    if impulse_strength > 0.02:  # 2% impulse move
                        order_blocks.append(OrderBlock(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            high_price=current_candle['high'],
                            low_price=current_candle['low'],
                            block_type='bullish',
                            formation_candle_index=i,
                            strength=impulse_strength
                        ))
            
            # Look for bearish order blocks (last up candle before impulse down)
            for i in range(3, len(stock_data) - 3):
                current_candle = stock_data.iloc[i]
                
                if current_candle['close'] > current_candle['open']:  # Bullish candle
                    # Check for bearish impulse in next 3 candles
                    next_candles = stock_data.iloc[i+1:i+4]
                    impulse_strength = (current_candle['high'] - next_candles['close'].min()) / current_candle['high']
                    
                    if impulse_strength > 0.02:  # 2% impulse move
                        order_blocks.append(OrderBlock(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            high_price=current_candle['high'],
                            low_price=current_candle['low'],
                            block_type='bearish',
                            formation_candle_index=i,
                            strength=impulse_strength
                        ))
            
            return sorted(order_blocks, key=lambda x: x.strength, reverse=True)
            
        except Exception as e:
            logger.error(f"Error in order blocks analysis: {e}")
            return []
    
    def concept_5_breaker_blocks(self, stock_data: pd.DataFrame) -> List[OrderBlock]:
        """
        CONCEPT 5: Breaker Blocks
        - Former resistance becomes support in stocks
        - Former support becomes resistance in stocks
        - Polarity switch detection
        """
        try:
            breaker_blocks = []
            order_blocks = self.concept_4_order_blocks_bullish_bearish(stock_data)
            
            for ob in order_blocks:
                # Check if order block has been broken and is now acting as support/resistance
                ob_index = ob.formation_candle_index
                
                if ob_index < len(stock_data) - 5:  # Need some candles after formation
                    future_data = stock_data.iloc[ob_index+1:]
                    
                    if ob.block_type == 'bullish':
                        # Check if price broke below and then found support
                        broken_below = future_data['low'].min() < ob.low_price
                        if broken_below:
                            # Check for subsequent support
                            support_tests = future_data[future_data['low'] <= ob.high_price * 1.01]
                            support_tests = support_tests[support_tests['low'] >= ob.low_price * 0.99]
                            
                            if len(support_tests) >= 2:
                                breaker = OrderBlock(
                                    timestamp=ob.timestamp,
                                    high_price=ob.high_price,
                                    low_price=ob.low_price,
                                    block_type='bullish_breaker',
                                    formation_candle_index=ob.formation_candle_index,
                                    strength=ob.strength * 1.2,  # Breakers are stronger
                                    is_breaker=True
                                )
                                breaker_blocks.append(breaker)
                    
                    elif ob.block_type == 'bearish':
                        # Check if price broke above and then found resistance
                        broken_above = future_data['high'].max() > ob.high_price
                        if broken_above:
                            # Check for subsequent resistance
                            resistance_tests = future_data[future_data['high'] >= ob.low_price * 0.99]
                            resistance_tests = resistance_tests[resistance_tests['high'] <= ob.high_price * 1.01]
                            
                            if len(resistance_tests) >= 2:
                                breaker = OrderBlock(
                                    timestamp=ob.timestamp,
                                    high_price=ob.high_price,
                                    low_price=ob.low_price,
                                    block_type='bearish_breaker',
                                    formation_candle_index=ob.formation_candle_index,
                                    strength=ob.strength * 1.2,
                                    is_breaker=True
                                )
                                breaker_blocks.append(breaker)
            
            return breaker_blocks
            
        except Exception as e:
            logger.error(f"Error in breaker blocks analysis: {e}")
            return []
    
    def concept_6_fair_value_gaps_fvg_imbalances(self, stock_data: pd.DataFrame) -> List[FairValueGap]:
        """
        CONCEPT 6: Fair Value Gaps (FVG) / Imbalances
        - 3-candle gap patterns in stock charts
        - Bullish FVG: gap up in stock price
        - Bearish FVG: gap down in stock price
        - 50% mitigation rule for stocks
        """
        try:
            fvgs = []
            
            if len(stock_data) < 3:
                return fvgs
            
            for i in range(1, len(stock_data) - 1):
                prev_candle = stock_data.iloc[i-1]
                current_candle = stock_data.iloc[i]
                next_candle = stock_data.iloc[i+1]
                
                # Bullish FVG: previous high < next low (gap up)
                if prev_candle['high'] < next_candle['low']:
                    gap_size = next_candle['low'] - prev_candle['high']
                    gap_percentage = gap_size / prev_candle['high']
                    
                    if gap_percentage > 0.001:  # Minimum 0.1% gap
                        fvg = FairValueGap(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            gap_high=next_candle['low'],
                            gap_low=prev_candle['high'],
                            gap_type='bullish',
                            gap_size=gap_size,
                            mitigation_level=(next_candle['low'] + prev_candle['high']) / 2
                        )
                        fvgs.append(fvg)
                
                # Bearish FVG: previous low > next high (gap down)
                elif prev_candle['low'] > next_candle['high']:
                    gap_size = prev_candle['low'] - next_candle['high']
                    gap_percentage = gap_size / prev_candle['low']
                    
                    if gap_percentage > 0.001:  # Minimum 0.1% gap
                        fvg = FairValueGap(
                            timestamp=current_candle['timestamp'] if 'timestamp' in current_candle else current_candle.name,
                            gap_high=prev_candle['low'],
                            gap_low=next_candle['high'],
                            gap_type='bearish',
                            gap_size=gap_size,
                            mitigation_level=(prev_candle['low'] + next_candle['high']) / 2
                        )
                        fvgs.append(fvg)
            
            # Check for mitigation
            for fvg in fvgs:
                fvg.is_mitigated = self._check_fvg_mitigation(stock_data, fvg)
            
            return fvgs
            
        except Exception as e:
            logger.error(f"Error in FVG analysis: {e}")
            return []
    
    # Helper methods
    def _find_swing_highs(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing high points"""
        swing_highs = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_high = df['high'].iloc[i]
            is_swing_high = True
            
            # Check if current point is higher than surrounding points
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                strength = self._calculate_swing_strength(df, i, 'high')
                swing_highs.append(SwingPoint(
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    price=current_high,
                    swing_type='high',
                    index=i,
                    strength=strength
                ))
        
        return swing_highs
    
    def _find_swing_lows(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Find swing low points"""
        swing_lows = []
        
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            current_low = df['low'].iloc[i]
            is_swing_low = True
            
            # Check if current point is lower than surrounding points
            for j in range(i - self.swing_lookback, i + self.swing_lookback + 1):
                if j != i and df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                strength = self._calculate_swing_strength(df, i, 'low')
                swing_lows.append(SwingPoint(
                    timestamp=df.iloc[i]['timestamp'] if 'timestamp' in df.columns else df.index[i],
                    price=current_low,
                    swing_type='low',
                    index=i,
                    strength=strength
                ))
        
        return swing_lows
    
    def _calculate_swing_strength(self, df: pd.DataFrame, index: int, swing_type: str) -> float:
        """Calculate the strength of a swing point"""
        try:
            if swing_type == 'high':
                # Volume and range at the swing high
                volume_ratio = df['volume'].iloc[index] / df['volume'].rolling(20).mean().iloc[index]
                price_range = (df['high'].iloc[index] - df['low'].iloc[index]) / df['close'].iloc[index]
            else:
                # Volume and range at the swing low
                volume_ratio = df['volume'].iloc[index] / df['volume'].rolling(20).mean().iloc[index]
                price_range = (df['high'].iloc[index] - df['low'].iloc[index]) / df['close'].iloc[index]
            
            # Combine volume and range for strength
            strength = min(volume_ratio * price_range * 10, 1.0)  # Cap at 1.0
            return max(strength, 0.1)  # Minimum strength
            
        except:
            return 0.5  # Default strength
    
    def _classify_current_structure(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """Classify current market structure"""
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return 'insufficient_data'
        
        recent_highs = swing_highs[-2:]
        recent_lows = swing_lows[-2:]
        
        # Compare recent swing points
        if recent_highs[-1].price > recent_highs[-2].price and recent_lows[-1].price > recent_lows[-2].price:
            return 'bullish_structure'  # HH and HL
        elif recent_highs[-1].price < recent_highs[-2].price and recent_lows[-1].price < recent_lows[-2].price:
            return 'bearish_structure'  # LH and LL
        else:
            return 'mixed_structure'
    
    def _detect_structure_breaks(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> List[Dict]:
        """Detect market structure breaks"""
        breaks = []
        
        # Detect breaks in bullish structure (break of recent swing low)
        if len(swing_lows) >= 2:
            for i in range(1, len(swing_lows)):
                if swing_lows[i].price < swing_lows[i-1].price:
                    breaks.append({
                        'type': 'bearish_break',
                        'timestamp': swing_lows[i].timestamp,
                        'price': swing_lows[i].price,
                        'previous_low': swing_lows[i-1].price
                    })
        
        # Detect breaks in bearish structure (break of recent swing high)
        if len(swing_highs) >= 2:
            for i in range(1, len(swing_highs)):
                if swing_highs[i].price > swing_highs[i-1].price:
                    breaks.append({
                        'type': 'bullish_break',
                        'timestamp': swing_highs[i].timestamp,
                        'price': swing_highs[i].price,
                        'previous_high': swing_highs[i-1].price
                    })
        
        return breaks
    
    def _determine_trend_direction(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> str:
        """Determine overall trend direction"""
        if len(swing_highs) < 3 or len(swing_lows) < 3:
            return 'sideways'
        
        # Analyze last 3 swing points
        recent_highs = [h.price for h in swing_highs[-3:]]
        recent_lows = [l.price for l in swing_lows[-3:]]
        
        highs_trend = 'up' if recent_highs[-1] > recent_highs[0] else 'down'
        lows_trend = 'up' if recent_lows[-1] > recent_lows[0] else 'down'
        
        if highs_trend == 'up' and lows_trend == 'up':
            return 'uptrend'
        elif highs_trend == 'down' and lows_trend == 'down':
            return 'downtrend'
        else:
            return 'sideways'
    
    def _calculate_structure_confidence(self, swing_highs: List[SwingPoint], swing_lows: List[SwingPoint]) -> float:
        """Calculate confidence in market structure analysis"""
        if not swing_highs or not swing_lows:
            return 0.0
        
        # Base confidence on number of swing points and their strength
        total_points = len(swing_highs) + len(swing_lows)
        avg_strength = np.mean([h.strength for h in swing_highs] + [l.strength for l in swing_lows])
        
        confidence = min((total_points / 10) * avg_strength, 1.0)
        return confidence
    
    def _is_significant_level(self, df: pd.DataFrame, price: float) -> bool:
        """Check if a price level is significant"""
        current_price = df['close'].iloc[-1]
        distance = abs(price - current_price) / current_price
        
        # Level should be within reasonable distance but not too close
        return 0.005 < distance < 0.1  # Between 0.5% and 10%
    
    def _find_equal_levels(self, swing_points: List[SwingPoint], point_type: str) -> List[List[SwingPoint]]:
        """Find equal levels among swing points"""
        if len(swing_points) < 2:
            return []
        
        equal_levels = []
        used_points = set()
        
        for i, point1 in enumerate(swing_points):
            if i in used_points:
                continue
                
            level_group = [point1]
            used_points.add(i)
            
            for j, point2 in enumerate(swing_points[i+1:], i+1):
                if j in used_points:
                    continue
                    
                # Check if points are at equal levels
                price_diff = abs(point1.price - point2.price) / point1.price
                if price_diff <= self.liquidity_threshold:
                    level_group.append(point2)
                    used_points.add(j)
            
            if len(level_group) >= 2:
                equal_levels.append(level_group)
        
        return equal_levels
    
    def _find_trendline_liquidity(self, df: pd.DataFrame) -> List[LiquidityPool]:
        """Find trendline liquidity pools"""
        # Simplified trendline liquidity detection
        # This would need more sophisticated trendline analysis in production
        pools = []
        
        try:
            # Find upward trendlines (connecting swing lows)
            swing_lows = self._find_swing_lows(df)
            if len(swing_lows) >= 2:
                for i in range(len(swing_lows) - 1):
                    low1, low2 = swing_lows[i], swing_lows[i+1]
                    
                    # Calculate trendline level at current time
                    time_diff = (df.index[-1] - low2.index) if hasattr(df.index[-1], '__sub__') else 1
                    price_diff = low2.price - low1.price
                    time_span = low2.index - low1.index if time_span != 0 else 1
                    
                    if time_span > 0:
                        trendline_price = low2.price + (price_diff / time_span) * time_diff
                        
                        pools.append(LiquidityPool(
                            timestamp=datetime.now(),
                            price_level=trendline_price,
                            pool_type='upward_trendline',
                            strength=0.6,
                            touches=2
                        ))
            
        except Exception as e:
            logger.error(f"Error in trendline liquidity: {e}")
        
        return pools
    
    def _check_fvg_mitigation(self, df: pd.DataFrame, fvg: FairValueGap) -> bool:
        """Check if FVG has been mitigated"""
        try:
            # Find data after FVG formation
            fvg_timestamp = fvg.timestamp
            
            if hasattr(df, 'timestamp'):
                future_data = df[df['timestamp'] > fvg_timestamp]
            else:
                # If no timestamp column, assume chronological order
                fvg_index = df[df.index == fvg_timestamp].index
                if len(fvg_index) > 0:
                    future_data = df.iloc[fvg_index[0]+1:]
                else:
                    return False
            
            if future_data.empty:
                return False
            
            # Check if price has returned to 50% of the gap
            if fvg.gap_type == 'bullish':
                # Price should come back down to mitigation level
                return future_data['low'].min() <= fvg.mitigation_level
            else:
                # Price should come back up to mitigation level
                return future_data['high'].max() >= fvg.mitigation_level
                
        except Exception:
            return False

# Global instance
market_structure_analyzer = StockMarketStructureAnalyzer()