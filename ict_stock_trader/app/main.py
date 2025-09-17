"""
FastAPI Main Application for ICT Stock Trader
"""
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import uvicorn
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import logging

# Import configurations and dependencies
from config.settings import settings
from app.database import get_db, create_tables
from app.data.data_processor import data_processor
from app.ict_engine.core_concepts import market_structure_analyzer

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
        logger.info("Database tables created successfully")
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
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": settings.VERSION
    }

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

# ICT Analysis Endpoints
@app.get(f"{settings.API_V1_STR}/ict/analysis/{{symbol}}")
async def get_ict_analysis(
    symbol: str,
    timeframe: str = "1d",
    concepts: Optional[str] = None  # Comma-separated list of concept numbers
):
    """
    Get comprehensive ICT analysis for a stock
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
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "analysis_timestamp": datetime.now(),
            "data_points_analyzed": len(stock_data),
            "ict_analysis": analysis_results,
            "summary": _generate_analysis_summary(analysis_results)
        }
        
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

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )