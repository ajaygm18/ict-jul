"""
FastAPI Main Application for ICT Stock Trader
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import uvicorn
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging
import numpy as np
import pandas as pd

# Import configurations and dependencies
from config.settings import settings
from app.database import get_db, create_tables
from app.data.data_processor import data_processor
from app.ict_engine.core_concepts import market_structure_analyzer
from app.ict_engine.time_price import time_price_analyzer
from app.ict_engine.risk_management import risk_management_engine
from app.ict_engine.advanced_concepts import advanced_concepts_analyzer
from app.ict_engine.strategies import ict_strategies_engine
from app.system_monitor import system_monitor, db_optimizer, get_health_status
from app.ai.ai_integration import ai_engine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.DESCRIPTION,
    version=settings.VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create database tables on startup
@app.on_event("startup")
async def startup_event():
    """Initialize database and background tasks"""
    try:
        create_tables()
        # Initialize AI components
        await ai_engine.initialize()
        logger.info("Database tables created successfully")
        logger.info("AI/ML components initialized")
        logger.info(f"ICT Stock Trader API started on {settings.API_V1_STR}")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "ICT Stock Trader API",
        "version": settings.VERSION,
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """
    Comprehensive health check endpoint with system monitoring
    """
    try:
        health_status = get_health_status()
        return {
            "status": "healthy" if health_status["health_status"] == "healthy" else "degraded",
            "timestamp": datetime.now(),
            "version": settings.VERSION,
            "system_health": health_status
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.now(),
            "error": str(e)
        }

@app.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with performance metrics
    """
    try:
        health_status = get_health_status()
        performance_report = system_monitor.get_performance_report()
        
        return {
            "timestamp": datetime.now(),
            "system_health": health_status,
            "performance_metrics": performance_report,
            "database_status": db_optimizer.get_database_stats()
        }
    except Exception as e:
        logging.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/optimize")
async def optimize_system():
    """
    Trigger system optimization (cache cleanup, database optimization)
    """
    try:
        # Clear cache
        system_monitor.clear_cache()
        
        # Optimize database
        db_optimizer.optimize_database()
        
        # Run performance optimization
        system_monitor.optimize_performance()
        
        return {
            "message": "System optimization completed",
            "timestamp": datetime.now(),
            "actions_performed": [
                "Cache cleared",
                "Database optimized", 
                "Performance analysis completed"
            ]
        }
    except Exception as e:
        logging.error(f"System optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/metrics")
async def get_system_metrics():
    """
    Get real-time system performance metrics
    """
    try:
        return system_monitor.get_performance_report()
    except Exception as e:
        logging.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Stock Data Endpoints
@app.get(f"{settings.API_V1_STR}/stocks/{{symbol}}/data")
async def get_stock_data(
    symbol: str,
    timeframe: str = "1d",
    period: str = "5d",
    include_indicators: bool = True
):
    """
    Get stock data with technical indicators
    """
    try:
        symbol = symbol.upper()
        
        # Get comprehensive market data
        data = await data_processor.process_real_time_data(symbol, timeframe)
        
        if 'error' in data:
            raise HTTPException(status_code=404, detail=data['error'])
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": data['data'].to_dict('records') if not data['data'].empty else [],
            "fundamentals": data.get('fundamentals', {}),
            "economic_context": data.get('economic_context', {}),
            "last_update": data['last_update'],
            "data_points": data['data_points']
        }
        
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/stocks/multiple")
async def get_multiple_stocks_data(
    symbols: str,  # Comma-separated symbols
    timeframe: str = "1d",
    include_fundamentals: bool = True
):
    """
    Get data for multiple stocks
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 symbols allowed")
        
        # Get comprehensive market data
        data = await data_processor.get_comprehensive_market_data(
            symbols=symbol_list,
            timeframes=[timeframe],
            include_fundamentals=include_fundamentals,
            include_economic=True,
            include_news=True
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Error fetching multiple stocks data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper function to convert numpy types to Python types for JSON serialization
def convert_numpy_types(obj):
    """Convert numpy types to Python types for JSON serialization"""
    import dataclasses
    from enum import Enum
    
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif dataclasses.is_dataclass(obj):
        # Convert dataclass to dict
        return convert_numpy_types(dataclasses.asdict(obj))
    elif isinstance(obj, Enum):
        # Convert enum to its value
        return obj.value
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None  # Convert NaN and infinity to null
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        # Handle arrays by converting to list and processing each element
        return [convert_numpy_types(item) for item in obj.tolist()]
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, 'isoformat') and callable(obj.isoformat):
        return obj.isoformat()
    elif hasattr(obj, 'item') and callable(obj.item):  # Handle numpy scalars
        return convert_numpy_types(obj.item())
    else:
        return obj

# ICT Analysis Endpoints
@app.get(f"{settings.API_V1_STR}/ict/analysis/{{symbol}}")
@system_monitor.performance_monitor(cache_duration=300)  # Cache for 5 minutes
async def get_ict_analysis(
    symbol: str,
    timeframe: str = "1d",
    concepts: Optional[str] = None  # Comma-separated list of concept numbers
):
    """
    Get comprehensive ICT analysis for a stock with performance monitoring
    """
    try:
        symbol = symbol.upper()
        
        # Get stock data
        stock_data_result = await data_processor.process_real_time_data(symbol, timeframe)
        
        if 'error' in stock_data_result:
            raise HTTPException(status_code=404, detail=stock_data_result['error'])
        
        stock_data = stock_data_result['data']
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail="No data available for analysis")
        
        # Parse requested concepts
        requested_concepts = []
        if concepts:
            try:
                requested_concepts = [int(c.strip()) for c in concepts.split(',')]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid concept numbers")
        
        # Perform ICT analysis
        analysis_results = {}
        
        # Core Concepts (1-20)
        if not requested_concepts or any(1 <= c <= 20 for c in requested_concepts):
            if not requested_concepts or 1 in requested_concepts:
                analysis_results['concept_1_market_structure'] = market_structure_analyzer.concept_1_market_structure_hh_hl_lh_ll(stock_data)
            
            if not requested_concepts or 2 in requested_concepts:
                analysis_results['concept_2_liquidity'] = market_structure_analyzer.concept_2_liquidity_buyside_sellside(stock_data)
            
            if not requested_concepts or 3 in requested_concepts:
                analysis_results['concept_3_liquidity_pools'] = [
                    {
                        'timestamp': pool.timestamp.isoformat() if hasattr(pool.timestamp, 'isoformat') else str(pool.timestamp),
                        'price_level': pool.price_level,
                        'pool_type': pool.pool_type,
                        'strength': pool.strength,
                        'touches': pool.touches
                    } for pool in market_structure_analyzer.concept_3_liquidity_pools(stock_data)
                ]
            
            if not requested_concepts or 4 in requested_concepts:
                analysis_results['concept_4_order_blocks'] = [
                    {
                        'timestamp': ob.timestamp.isoformat() if hasattr(ob.timestamp, 'isoformat') else str(ob.timestamp),
                        'high_price': ob.high_price,
                        'low_price': ob.low_price,
                        'block_type': ob.block_type,
                        'strength': ob.strength,
                        'is_breaker': ob.is_breaker
                    } for ob in market_structure_analyzer.concept_4_order_blocks_bullish_bearish(stock_data)
                ]
            
            if not requested_concepts or 5 in requested_concepts:
                analysis_results['concept_5_breaker_blocks'] = [
                    {
                        'timestamp': bb.timestamp.isoformat() if hasattr(bb.timestamp, 'isoformat') else str(bb.timestamp),
                        'high_price': bb.high_price,
                        'low_price': bb.low_price,
                        'block_type': bb.block_type,
                        'strength': bb.strength,
                        'is_breaker': bb.is_breaker
                    } for bb in market_structure_analyzer.concept_5_breaker_blocks(stock_data)
                ]
            
            if not requested_concepts or 6 in requested_concepts:
                analysis_results['concept_6_fair_value_gaps'] = [
                    {
                        'timestamp': fvg.timestamp.isoformat() if hasattr(fvg.timestamp, 'isoformat') else str(fvg.timestamp),
                        'gap_high': fvg.gap_high,
                        'gap_low': fvg.gap_low,
                        'gap_type': fvg.gap_type,
                        'gap_size': fvg.gap_size,
                        'mitigation_level': fvg.mitigation_level,
                        'is_mitigated': fvg.is_mitigated
                    } for fvg in market_structure_analyzer.concept_6_fair_value_gaps_fvg_imbalances(stock_data)
                ]
            
            if not requested_concepts or 7 in requested_concepts:
                analysis_results['concept_7_rejection_blocks'] = market_structure_analyzer.concept_7_rejection_blocks(stock_data)
            
            if not requested_concepts or 8 in requested_concepts:
                analysis_results['concept_8_mitigation_blocks'] = market_structure_analyzer.concept_8_mitigation_blocks(stock_data)
            
            if not requested_concepts or 9 in requested_concepts:
                analysis_results['concept_9_supply_demand_zones'] = market_structure_analyzer.concept_9_supply_demand_zones(stock_data)
            
            if not requested_concepts or 10 in requested_concepts:
                analysis_results['concept_10_premium_discount'] = market_structure_analyzer.concept_10_premium_discount_ote(stock_data)
            
            if not requested_concepts or 11 in requested_concepts:
                analysis_results['concept_11_dealing_ranges'] = market_structure_analyzer.concept_11_dealing_ranges(stock_data)
            
            if not requested_concepts or 12 in requested_concepts:
                analysis_results['concept_12_swing_analysis'] = market_structure_analyzer.concept_12_swing_highs_swing_lows(stock_data)
            
            if not requested_concepts or 13 in requested_concepts:
                analysis_results['concept_13_market_maker_models'] = market_structure_analyzer.concept_13_market_maker_buy_sell_models(stock_data)
            
            if not requested_concepts or 14 in requested_concepts:
                analysis_results['concept_14_market_maker_programs'] = market_structure_analyzer.concept_14_market_maker_programs(stock_data)
            
            if not requested_concepts or 15 in requested_concepts:
                analysis_results['concept_15_judas_swing'] = market_structure_analyzer.concept_15_judas_swing(stock_data)
            
            if not requested_concepts or 16 in requested_concepts:
                analysis_results['concept_16_turtle_soup'] = market_structure_analyzer.concept_16_turtle_soup(stock_data)
            
            if not requested_concepts or 17 in requested_concepts:
                analysis_results['concept_17_power_of_3'] = market_structure_analyzer.concept_17_power_of_3(stock_data)
            
            if not requested_concepts or 18 in requested_concepts:
                analysis_results['concept_18_optimal_trade_entry'] = market_structure_analyzer.concept_18_optimal_trade_entry(stock_data)
            
            if not requested_concepts or 19 in requested_concepts:
                analysis_results['concept_19_smt_divergence'] = market_structure_analyzer.concept_19_smt_divergence(stock_data)
            
            if not requested_concepts or 20 in requested_concepts:
                analysis_results['concept_20_liquidity_voids'] = market_structure_analyzer.concept_20_liquidity_voids_inefficiencies(stock_data)
        
        # Time & Price concepts (21-30)
        if not requested_concepts or any(21 <= c <= 30 for c in requested_concepts):
            if not requested_concepts or 21 in requested_concepts:
                analysis_results['concept_21_killzones'] = time_price_analyzer.concept_21_stock_killzones(stock_data)
            if not requested_concepts or 22 in requested_concepts:
                analysis_results['concept_22_session_opens'] = time_price_analyzer.concept_22_stock_session_opens(stock_data)
            if not requested_concepts or 23 in requested_concepts:
                analysis_results['concept_23_fibonacci'] = time_price_analyzer.concept_23_fibonacci_ratios(stock_data)
            if not requested_concepts or 24 in requested_concepts:
                analysis_results['concept_24_daily_weekly_ranges'] = time_price_analyzer.concept_24_daily_weekly_range_expectations(stock_data)
            if not requested_concepts or 25 in requested_concepts:
                analysis_results['concept_25_liquidity_raids'] = time_price_analyzer.concept_25_session_liquidity_raids(stock_data)
            if not requested_concepts or 26 in requested_concepts:
                analysis_results['concept_26_weekly_profiles'] = time_price_analyzer.concept_26_weekly_profiles(stock_data)
            if not requested_concepts or 27 in requested_concepts:
                analysis_results['concept_27_daily_bias'] = time_price_analyzer.concept_27_daily_bias(stock_data)
            if not requested_concepts or 28 in requested_concepts:
                analysis_results['concept_28_weekly_bias'] = time_price_analyzer.concept_28_weekly_bias(stock_data)
            if not requested_concepts or 29 in requested_concepts:
                analysis_results['concept_29_monthly_bias'] = time_price_analyzer.concept_29_monthly_bias(stock_data)
            if not requested_concepts or 30 in requested_concepts:
                analysis_results['concept_30_time_of_day_highs_lows'] = time_price_analyzer.concept_30_time_of_day_highs_lows(stock_data)
        
        # Risk Management concepts (31-39)
        if not requested_concepts or any(31 <= c <= 39 for c in requested_concepts):
            # Example data for risk management concepts (in production, this would come from database)
            example_trades = [
                {
                    'trade_id': f'trade_{symbol}_1',
                    'timestamp': '2024-01-15 10:30:00',
                    'symbol': symbol,
                    'direction': 'LONG',
                    'entry_price': stock_data['close'].iloc[-10] if len(stock_data) > 10 else stock_data['close'].iloc[0],
                    'stop_loss': stock_data['close'].iloc[-10] * 0.98 if len(stock_data) > 10 else stock_data['close'].iloc[0] * 0.98,
                    'take_profit': stock_data['close'].iloc[-10] * 1.04 if len(stock_data) > 10 else stock_data['close'].iloc[0] * 1.04,
                    'position_size': 100,
                    'risk_amount': stock_data['close'].iloc[-10] * 2 if len(stock_data) > 10 else stock_data['close'].iloc[0] * 2,
                    'setup_type': 'A+',
                    'ict_concepts': ['Order Block', 'Fair Value Gap'],
                    'confluence_score': 0.85,
                    'outcome': 'WIN',
                    'pnl': 400,
                    'pnl_percentage': 2.67
                }
            ]
            
            active_trades = [
                {
                    'trade_id': f'active_{symbol}_1',
                    'entry_price': stock_data['close'].iloc[-5] if len(stock_data) > 5 else stock_data['close'].iloc[0],
                    'stop_loss': stock_data['close'].iloc[-5] * 0.98 if len(stock_data) > 5 else stock_data['close'].iloc[0] * 0.98,
                    'direction': 'LONG'
                }
            ]
            
            trade_setups = [
                {
                    'entry_price': stock_data['close'].iloc[-1],
                    'stop_loss': stock_data['close'].iloc[-1] * 0.99,
                    'take_profit': stock_data['close'].iloc[-1] * 1.03,
                    'direction': 'LONG',
                    'win_probability': 0.65,
                    'confluence_score': 0.75,
                    'supporting_concepts': ['Order Block', 'Trend Alignment']
                }
            ]
            
            trading_history = [
                {'timestamp': '2024-01-10', 'pnl': 150},
                {'timestamp': '2024-01-11', 'pnl': -75},
                {'timestamp': '2024-01-12', 'pnl': 200},
                {'timestamp': '2024-01-13', 'pnl': -50},
                {'timestamp': '2024-01-14', 'pnl': 100}
            ]
            
            if not requested_concepts or 31 in requested_concepts:
                analysis_results['concept_31_trade_journaling'] = risk_management_engine.concept_31_trade_journaling_backtesting(stock_data, example_trades)
            
            if not requested_concepts or 32 in requested_concepts:
                analysis_results['concept_32_entry_models'] = risk_management_engine.concept_32_entry_models_fvg_ob_breaker(stock_data)
            
            if not requested_concepts or 33 in requested_concepts:
                analysis_results['concept_33_exit_models'] = risk_management_engine.concept_33_exit_models_partial_full_scaling(stock_data, active_trades)
            
            if not requested_concepts or 34 in requested_concepts:
                analysis_results['concept_34_risk_reward_optimization'] = risk_management_engine.concept_34_risk_to_reward_optimization(stock_data, trade_setups)
            
            if not requested_concepts or 35 in requested_concepts:
                analysis_results['concept_35_position_sizing'] = risk_management_engine.concept_35_position_sizing_algorithms(50000, trade_setups)
            
            if not requested_concepts or 36 in requested_concepts:
                analysis_results['concept_36_drawdown_control'] = risk_management_engine.concept_36_drawdown_control(trading_history, 50000)
            
            if not requested_concepts or 37 in requested_concepts:
                analysis_results['concept_37_compounding_models'] = risk_management_engine.concept_37_compounding_models(10000, 3.0, 5)
            
            if not requested_concepts or 38 in requested_concepts:
                analysis_results['concept_38_daily_loss_limits'] = risk_management_engine.concept_38_daily_loss_limits(trading_history, 50000)
            
            if not requested_concepts or 39 in requested_concepts:
                analysis_results['concept_39_probability_profiles'] = risk_management_engine.concept_39_probability_profiles_abc_setups(trade_setups)
        
        # Advanced Concepts (40-50)
        if not requested_concepts or any(40 <= c <= 50 for c in requested_concepts):
            if not requested_concepts or 40 in requested_concepts:
                # Multi-timeframe data for high probability scenarios
                multi_tf_data = {
                    '1d': stock_data,
                    '1h': stock_data.tail(24),  # Simplified
                    '15m': stock_data.tail(96),  # Simplified
                    'symbol': symbol
                }
                analysis_results['concept_40_high_probability_scenarios'] = advanced_concepts_analyzer.concept_40_high_probability_scenarios(multi_tf_data)
            
            if not requested_concepts or 41 in requested_concepts:
                analysis_results['concept_41_liquidity_runs'] = advanced_concepts_analyzer.concept_41_liquidity_runs(stock_data)
            
            if not requested_concepts or 42 in requested_concepts:
                analysis_results['concept_42_reversals_vs_continuations'] = advanced_concepts_analyzer.concept_42_reversals_vs_continuations(stock_data)
            
            if not requested_concepts or 43 in requested_concepts:
                analysis_results['concept_43_accumulation_distribution'] = advanced_concepts_analyzer.concept_43_accumulation_distribution_schematics(stock_data)
            
            if not requested_concepts or 44 in requested_concepts:
                analysis_results['concept_44_order_flow_narrative'] = advanced_concepts_analyzer.concept_44_order_flow_institutional_narrative(stock_data)
            
            if not requested_concepts or 45 in requested_concepts:
                analysis_results['concept_45_high_low_day'] = advanced_concepts_analyzer.concept_45_high_low_day_identification(stock_data)
            
            if not requested_concepts or 46 in requested_concepts:
                analysis_results['concept_46_range_expansion'] = advanced_concepts_analyzer.concept_46_range_expansion(stock_data)
            
            if not requested_concepts or 47 in requested_concepts:
                analysis_results['concept_47_inside_outside_days'] = advanced_concepts_analyzer.concept_47_inside_outside_days(stock_data)
            
            if not requested_concepts or 48 in requested_concepts:
                analysis_results['concept_48_weekly_profiles'] = advanced_concepts_analyzer.concept_48_weekly_profile_analysis(stock_data)
            
            if not requested_concepts or 49 in requested_concepts:
                analysis_results['concept_49_ipda_theory'] = advanced_concepts_analyzer.concept_49_ipda_theory(stock_data)
            
            if not requested_concepts or 50 in requested_concepts:
                analysis_results['concept_50_algo_price_delivery'] = advanced_concepts_analyzer.concept_50_algo_price_delivery(stock_data)
        
        # Strategies (51-65)
        if not requested_concepts or any(51 <= c <= 65 for c in requested_concepts):
            if not requested_concepts or 51 in requested_concepts:
                analysis_results['concept_51_silver_bullet'] = ict_strategies_engine.concept_51_silver_bullet_strategy(stock_data)
            
            if not requested_concepts or 52 in requested_concepts:
                analysis_results['concept_52_premarket_breakout'] = ict_strategies_engine.concept_52_pre_market_breakout_strategy(stock_data)
            
            if not requested_concepts or 53 in requested_concepts:
                analysis_results['concept_53_market_open_reversal'] = ict_strategies_engine.concept_53_market_open_reversal(stock_data)
            
            if not requested_concepts or 54 in requested_concepts:
                analysis_results['concept_54_power_hour'] = ict_strategies_engine.concept_54_power_hour_strategy(stock_data)
            
            if not requested_concepts or 55 in requested_concepts:
                analysis_results['concept_55_fvg_sniper'] = ict_strategies_engine.concept_55_fvg_sniper_entry(stock_data)
            
            if not requested_concepts or 56 in requested_concepts:
                analysis_results['concept_56_order_block_strategy'] = ict_strategies_engine.concept_56_order_block_strategy(stock_data)
            
            if not requested_concepts or 57 in requested_concepts:
                analysis_results['concept_57_breaker_block_strategy'] = ict_strategies_engine.concept_57_breaker_block_strategy(stock_data)
            
            if not requested_concepts or 58 in requested_concepts:
                analysis_results['concept_58_rejection_block_strategy'] = ict_strategies_engine.concept_58_rejection_block_strategy(stock_data)
            
            if not requested_concepts or 59 in requested_concepts:
                # For SMT divergence, we need correlated stocks data
                correlated_stocks = {
                    symbol: stock_data,
                    'SPY': stock_data,  # Simplified - would fetch actual SPY data
                    'QQQ': stock_data   # Simplified - would fetch actual QQQ data
                }
                analysis_results['concept_59_smt_divergence_strategy'] = ict_strategies_engine.concept_59_smt_divergence_strategy(correlated_stocks)
            
            if not requested_concepts or 60 in requested_concepts:
                analysis_results['concept_60_turtle_soup'] = ict_strategies_engine.concept_60_turtle_soup_strategy(stock_data)
            
            if not requested_concepts or 61 in requested_concepts:
                analysis_results['concept_61_power_of_3_strategy'] = ict_strategies_engine.concept_61_power_of_3_strategy(stock_data)
            
            if not requested_concepts or 62 in requested_concepts:
                analysis_results['concept_62_daily_bias_liquidity'] = ict_strategies_engine.concept_62_daily_bias_liquidity_strategy(stock_data)
            
            if not requested_concepts or 63 in requested_concepts:
                analysis_results['concept_63_morning_session'] = ict_strategies_engine.concept_63_morning_session_strategy(stock_data)
            
            if not requested_concepts or 64 in requested_concepts:
                analysis_results['concept_64_afternoon_reversal'] = ict_strategies_engine.concept_64_afternoon_reversal_strategy(stock_data)
            
            if not requested_concepts or 65 in requested_concepts:
                analysis_results['concept_65_optimal_trade_entry'] = ict_strategies_engine.concept_65_optimal_trade_entry_strategy(stock_data)
        
        # Convert numpy types for JSON serialization
        response_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(),
            "data_points_analyzed": len(stock_data),
            "ict_analysis": analysis_results,
            "summary": _generate_analysis_summary(analysis_results)
        }
        
        # Convert numpy types and return as JSONResponse
        converted_data = convert_numpy_types(response_data)
        return JSONResponse(content=converted_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ICT analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/market/overview")
async def get_market_overview():
    """
    Get overall market overview with key indicators
    """
    try:
        # Get data for major market indices
        major_symbols = ['SPY', 'QQQ', 'IWM', 'DIA']
        
        market_data = await data_processor.get_comprehensive_market_data(
            symbols=major_symbols,
            timeframes=['1d'],
            include_fundamentals=False,
            include_economic=True,
            include_news=True
        )
        
        # Calculate market sentiment
        market_sentiment = _calculate_market_sentiment(market_data)
        
        return {
            "timestamp": datetime.now(),
            "market_indices": market_data.get('stocks', {}),
            "economic_indicators": market_data.get('economic_indicators', {}),
            "market_context": market_data.get('market_context', {}),
            "market_sentiment": market_sentiment,
            "news": market_data.get('news', [])[:5]  # Top 5 news items
        }
        
    except Exception as e:
        logger.error(f"Error in market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/watchlist/default")
async def get_default_watchlist():
    """
    Get analysis for default stock watchlist
    """
    try:
        watchlist_data = await data_processor.get_comprehensive_market_data(
            symbols=settings.DEFAULT_STOCKS,
            timeframes=['1d'],
            include_fundamentals=True,
            include_economic=False,
            include_news=False
        )
        
        # Add quick ICT analysis for each stock
        enhanced_watchlist = {}
        
        for symbol, data in watchlist_data.get('stocks', {}).items():
            if '1d' in data and not data['1d'].empty:
                # Quick market structure analysis
                market_structure = market_structure_analyzer.concept_1_market_structure_hh_hl_lh_ll(data['1d'])
                
                # Quick liquidity analysis
                liquidity = market_structure_analyzer.concept_2_liquidity_buyside_sellside(data['1d'])
                
                # Premium/Discount analysis
                premium_discount = market_structure_analyzer.concept_10_premium_discount_ote(data['1d'])
                
                enhanced_watchlist[symbol] = {
                    'price_data': {
                        'current_price': data['1d']['close'].iloc[-1] if not data['1d'].empty else 0,
                        'daily_change': data['1d']['close'].pct_change().iloc[-1] if len(data['1d']) > 1 else 0,
                        'volume': data['1d']['volume'].iloc[-1] if not data['1d'].empty else 0
                    },
                    'fundamentals': data.get('fundamentals', {}),
                    'ict_snapshot': {
                        'market_structure': market_structure.get('current_structure', 'unknown'),
                        'trend_direction': market_structure.get('trend_direction', 'unknown'),
                        'liquidity_balance': liquidity.get('liquidity_balance', 0),
                        'market_bias': premium_discount.get('market_bias', 'neutral'),
                        'in_premium': premium_discount.get('premium_zone', False),
                        'in_discount': premium_discount.get('discount_zone', False)
                    }
                }
        
        return {
            "timestamp": datetime.now(),
            "watchlist": enhanced_watchlist,
            "market_context": watchlist_data.get('market_context', {})
        }
        
    except Exception as e:
        logger.error(f"Error in watchlist analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/sector/correlation/{{symbol}}")
@system_monitor.performance_monitor(cache_duration=600)  # Cache for 10 minutes
async def get_sector_correlation_analysis(
    symbol: str,
    sector_stocks: Optional[str] = None  # Comma-separated list of sector stocks
):
    """
    Get sector correlation analysis for ICT trading strategies with performance monitoring
    - Cross-sector divergence analysis
    - Relative strength within sector
    - Institutional rotation signals
    - Sector-wide liquidity analysis
    """
    try:
        symbol = symbol.upper()
        
        # Get primary stock data
        data = await data_processor.process_real_time_data(symbol, "1d")
        
        if 'error' in data:
            raise HTTPException(status_code=404, detail=data['error'])
        
        stock_data = data['data']
        
        # Parse sector stocks (if provided)
        sector_stock_list = []
        if sector_stocks:
            sector_stock_list = [s.strip().upper() for s in sector_stocks.split(',')]
        else:
            # Default sector stocks based on symbol (simplified logic)
            if symbol in ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']:
                sector_stock_list = ['AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA']
            elif symbol in ['JPM', 'BAC', 'WFC', 'GS']:
                sector_stock_list = ['JPM', 'BAC', 'WFC', 'GS']
            else:
                sector_stock_list = ['SPY', 'QQQ', 'IWM']  # Market indices
        
        # Perform sector correlation analysis
        sector_analysis = advanced_concepts_analyzer.analyze_sector_correlations(
            stock_data, sector_stock_list
        )
        
        # Add ICT-specific sector insights
        ict_sector_insights = {
            'smt_divergence_opportunities': [],
            'sector_liquidity_raids': [],
            'rotation_strategies': [],
            'relative_strength_ranking': sector_analysis.get('sector_strength', {})
        }
        
        # Check for SMT divergences within sector
        if len(sector_stock_list) > 1:
            # Simplified SMT analysis
            correlated_data = {symbol: stock_data}
            smt_analysis = ict_strategies_engine.concept_59_smt_divergence_strategy(correlated_data)
            ict_sector_insights['smt_divergence_opportunities'] = smt_analysis
        
        return {
            "symbol": symbol,
            "sector_stocks_analyzed": sector_stock_list,
            "analysis_timestamp": datetime.now(),
            "sector_correlation_analysis": sector_analysis,
            "ict_sector_insights": ict_sector_insights,
            "trading_recommendations": _generate_sector_trading_recommendations(
                sector_analysis, ict_sector_insights
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in sector correlation analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility functions
def _generate_analysis_summary(analysis_results: Dict) -> Dict:
    """Generate a summary of ICT analysis results"""
    summary = {
        "total_concepts_analyzed": len(analysis_results),
        "key_findings": [],
        "overall_bias": "neutral",
        "confidence_score": 0.5
    }
    
    try:
        # Market structure summary
        if 'concept_1_market_structure' in analysis_results:
            ms = analysis_results['concept_1_market_structure']
            summary["key_findings"].append({
                "concept": "Market Structure",
                "finding": f"Current structure: {ms.get('current_structure', 'unknown')}, Trend: {ms.get('trend_direction', 'unknown')}"
            })
            
            if ms.get('trend_direction') == 'uptrend':
                summary["overall_bias"] = "bullish"
            elif ms.get('trend_direction') == 'downtrend':
                summary["overall_bias"] = "bearish"
        
        # Premium/Discount summary
        if 'concept_10_premium_discount' in analysis_results:
            pd = analysis_results['concept_10_premium_discount']
            if not isinstance(pd, dict) or 'error' not in pd:
                market_bias = pd.get('market_bias', 'neutral')
                summary["key_findings"].append({
                    "concept": "Premium/Discount",
                    "finding": f"Market bias: {market_bias}"
                })
        
        # Order blocks summary
        if 'concept_4_order_blocks' in analysis_results:
            obs = analysis_results['concept_4_order_blocks']
            if obs:
                bullish_obs = len([ob for ob in obs if ob.get('block_type') == 'bullish'])
                bearish_obs = len([ob for ob in obs if ob.get('block_type') == 'bearish'])
                summary["key_findings"].append({
                    "concept": "Order Blocks",
                    "finding": f"Bullish: {bullish_obs}, Bearish: {bearish_obs}"
                })
        
        # FVG summary
        if 'concept_6_fair_value_gaps' in analysis_results:
            fvgs = analysis_results['concept_6_fair_value_gaps']
            if fvgs:
                bullish_fvgs = len([fvg for fvg in fvgs if fvg.get('gap_type') == 'bullish'])
                bearish_fvgs = len([fvg for fvg in fvgs if fvg.get('gap_type') == 'bearish'])
                summary["key_findings"].append({
                    "concept": "Fair Value Gaps",
                    "finding": f"Bullish: {bullish_fvgs}, Bearish: {bearish_fvgs}"
                })
        
    except Exception as e:
        logger.error(f"Error generating analysis summary: {e}")
    
    return summary

def _calculate_market_sentiment(market_data: Dict) -> Dict:
    """Calculate overall market sentiment"""
    sentiment = {
        "overall": "neutral",
        "volatility": "normal",
        "trend": "sideways",
        "confidence": 0.5
    }
    
    try:
        # Analyze major indices performance
        stocks = market_data.get('stocks', {})
        
        if stocks:
            daily_changes = []
            for symbol, data in stocks.items():
                if '1d' in data and not data['1d'].empty and len(data['1d']) > 1:
                    change = data['1d']['close'].pct_change().iloc[-1]
                    daily_changes.append(change)
            
            if daily_changes:
                avg_change = sum(daily_changes) / len(daily_changes)
                
                if avg_change > 0.01:
                    sentiment["overall"] = "bullish"
                    sentiment["trend"] = "uptrend"
                elif avg_change < -0.01:
                    sentiment["overall"] = "bearish"
                    sentiment["trend"] = "downtrend"
        
        # Check VIX for volatility
        economic = market_data.get('economic_indicators', {})
        if 'vix' in economic:
            vix_value = economic['vix'].get('current_value', 20)
            if vix_value > 30:
                sentiment["volatility"] = "high"
            elif vix_value > 20:
                sentiment["volatility"] = "elevated"
            elif vix_value < 15:
                sentiment["volatility"] = "low"
        
    except Exception as e:
        logger.error(f"Error calculating market sentiment: {e}")
    
    return sentiment

def _generate_sector_trading_recommendations(sector_analysis: Dict, ict_insights: Dict) -> Dict:
    """Generate ICT-based sector trading recommendations"""
    recommendations = {
        "primary_strategy": "neutral",
        "confidence": 0.5,
        "key_levels": [],
        "time_based_recommendations": {},
        "risk_factors": []
    }
    
    try:
        # Analyze sector strength
        sector_strength = sector_analysis.get('sector_strength', {})
        relative_strength = sector_strength.get('current_symbol_strength', 0.5)
        
        if relative_strength > 0.7:
            recommendations["primary_strategy"] = "bullish_bias"
            recommendations["confidence"] = 0.8
            recommendations["key_levels"].append("Look for OTE entries on pullbacks")
            recommendations["time_based_recommendations"]["morning_session"] = "Consider long positions during pullbacks"
            recommendations["time_based_recommendations"]["power_hour"] = "Watch for institutional accumulation"
        elif relative_strength < 0.3:
            recommendations["primary_strategy"] = "bearish_bias"
            recommendations["confidence"] = 0.8
            recommendations["key_levels"].append("Look for rejection block entries on rallies")
            recommendations["time_based_recommendations"]["morning_session"] = "Consider short positions on false breakouts"
            recommendations["time_based_recommendations"]["power_hour"] = "Watch for institutional distribution"
        else:
            recommendations["primary_strategy"] = "range_trading"
            recommendations["confidence"] = 0.6
            recommendations["key_levels"].append("Trade between support and resistance")
            recommendations["time_based_recommendations"]["morning_session"] = "Wait for clearer direction"
            recommendations["time_based_recommendations"]["power_hour"] = "Watch for breakout setups"
        
        # Add correlation-based recommendations
        correlations = sector_analysis.get('correlations', {})
        sector_correlation = correlations.get('sector_correlation', 0.75)
        
        if sector_correlation < 0.5:
            recommendations["risk_factors"].append("Low sector correlation - independent movement possible")
            recommendations["key_levels"].append("Monitor for SMT divergence opportunities")
        
        # Add SMT divergence recommendations
        smt_opportunities = ict_insights.get('smt_divergence_opportunities', [])
        if smt_opportunities:
            recommendations["key_levels"].append("SMT divergence detected - consider counter-trend positions")
        
        # Add session-specific recommendations
        recommendations["time_based_recommendations"]["silver_bullet"] = "9:45-10:00 AM optimal entry window"
        recommendations["time_based_recommendations"]["lunch_session"] = "Avoid new positions during low volume periods"
        
    except Exception as e:
        logger.error(f"Error generating sector recommendations: {e}")
    
    return recommendations

# AI/ML Endpoints
@app.get(f"{settings.API_V1_STR}/ai/analysis/{{symbol}}")
async def get_ai_analysis(
    symbol: str,
    timeframe: str = "5m",
    include_features: bool = False
):
    """
    Get AI-powered pattern analysis for a stock with 200+ technical indicators
    """
    try:
        symbol = symbol.upper()
        
        # Get stock data
        stock_data_result = await data_processor.process_real_time_data(symbol, timeframe)
        
        if 'error' in stock_data_result:
            raise HTTPException(status_code=404, detail=stock_data_result['error'])
        
        stock_data = stock_data_result['data']
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail="No data available for AI analysis")
        
        # Run AI analysis
        ai_analysis = await ai_engine.analyze_symbol_with_ai(symbol, stock_data, timeframe)
        
        # Include feature data if requested
        if include_features and ai_analysis['status'] == 'success':
            feature_set = ai_engine.create_features_for_symbol(symbol, stock_data, timeframe)
            ai_analysis['features'] = {
                'feature_names': feature_set.feature_names,
                'feature_count': len(feature_set.feature_names),
                'latest_values': feature_set.features.iloc[-1].to_dict() if not feature_set.features.empty else {}
            }
        
        return ai_analysis
        
    except Exception as e:
        logger.error(f"Error in AI analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/ai/train")
async def train_ai_models(
    background_tasks: BackgroundTasks,
    symbols: str = "AAPL,GOOGL,MSFT,TSLA,AMZN",  # Default training symbols
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """
    Train AI models on historical stock data (Background task)
    """
    try:
        symbol_list = [s.strip().upper() for s in symbols.split(',')]
        
        if len(symbol_list) > 20:
            raise HTTPException(status_code=400, detail="Maximum 20 symbols allowed for training")
        
        # Add training task to background
        background_tasks.add_task(
            ai_engine.train_models_for_symbols,
            symbol_list,
            start_date,
            end_date
        )
        
        return {
            "message": "AI model training started",
            "symbols": symbol_list,
            "start_date": start_date,
            "end_date": end_date,
            "status": "training_initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error initiating AI training: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/ai/features/{{symbol}}")
async def get_technical_indicators(
    symbol: str,
    timeframe: str = "5m",
    format: str = "summary"  # "summary" or "detailed"
):
    """
    Get 200+ technical indicators for a stock
    """
    try:
        symbol = symbol.upper()
        
        # Get stock data
        stock_data_result = await data_processor.process_real_time_data(symbol, timeframe)
        
        if 'error' in stock_data_result:
            raise HTTPException(status_code=404, detail=stock_data_result['error'])
        
        stock_data = stock_data_result['data']
        
        if stock_data.empty:
            raise HTTPException(status_code=404, detail="No data available for feature extraction")
        
        # Create comprehensive features
        feature_set = ai_engine.create_features_for_symbol(symbol, stock_data, timeframe)
        
        if format == "detailed":
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_count": len(feature_set.feature_names),
                "feature_names": feature_set.feature_names,
                "latest_values": feature_set.features.iloc[-1].to_dict() if not feature_set.features.empty else {},
                "historical_data": feature_set.features.tail(50).to_dict('records') if not feature_set.features.empty else [],
                "creation_timestamp": feature_set.creation_timestamp.isoformat()
            }
        else:
            # Summary format
            if not feature_set.features.empty:
                latest_features = feature_set.features.iloc[-1]
                feature_summary = {
                    'price_indicators': {k: v for k, v in latest_features.items() if 'SMA' in k or 'EMA' in k or 'BB' in k}[:10],
                    'volume_indicators': {k: v for k, v in latest_features.items() if 'Volume' in k or 'OBV' in k or 'VWAP' in k}[:5],
                    'momentum_indicators': {k: v for k, v in latest_features.items() if 'RSI' in k or 'MACD' in k or 'Stoch' in k}[:5],
                    'ict_indicators': {k: v for k, v in latest_features.items() if 'FVG' in k or 'OB' in k or 'Premium' in k}[:5]
                }
            else:
                feature_summary = {}
            
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "feature_count": len(feature_set.feature_names),
                "feature_summary": feature_summary,
                "creation_timestamp": feature_set.creation_timestamp.isoformat()
            }
        
    except Exception as e:
        logger.error(f"Error extracting features for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/ai/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from trained AI models
    """
    try:
        importance_data = ai_engine.get_feature_importance()
        return importance_data
        
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get(f"{settings.API_V1_STR}/ai/performance")
async def get_ai_performance():
    """
    Get AI engine performance statistics
    """
    try:
        performance_stats = ai_engine.get_ai_performance_stats()
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "performance": performance_stats
        }
        
    except Exception as e:
        logger.error(f"Error getting AI performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{settings.API_V1_STR}/ai/cache/clear")
async def clear_ai_cache():
    """
    Clear AI feature and pattern caches
    """
    try:
        ai_engine.clear_cache()
        return {
            "message": "AI caches cleared successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing AI cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )