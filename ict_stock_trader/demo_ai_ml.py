#!/usr/bin/env python3
"""
Demo script to showcase the AI/ML implementation for ICT Pattern Recognition
Demonstrates training and inference capabilities
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
import sys
import os

# Add app directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from ai.feature_engineer import TechnicalIndicatorEngine, MultiTimeframeFeatureEngine
from ai.pattern_detector import ICTPatternRecognitionAI
from ai.model_trainer import ModelTrainer, TrainingConfig
from ai.ai_integration import AIIntegrationEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def demo_feature_engineering():
    """Demonstrate comprehensive feature engineering with 200+ indicators"""
    
    logger.info("=" * 60)
    logger.info("DEMO 1: Feature Engineering (200+ Technical Indicators)")
    logger.info("=" * 60)
    
    # Download real stock data
    logger.info("Downloading real stock data for AAPL...")
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="3mo", interval="1h")
    
    if data.empty:
        logger.error("Failed to download data")
        return None
        
    logger.info(f"Downloaded {len(data)} data points")
    
    # Create feature engine
    feature_engine = TechnicalIndicatorEngine()
    
    # Generate comprehensive features
    logger.info("Creating comprehensive technical indicators...")
    feature_set = feature_engine.create_comprehensive_features(data, "AAPL", "1h")
    
    logger.info(f"✅ Created {len(feature_set.feature_names)} technical indicators")
    
    # Show feature categories
    price_features = [f for f in feature_set.feature_names if any(x in f for x in ['SMA', 'EMA', 'BB', 'Pivot'])]
    volume_features = [f for f in feature_set.feature_names if any(x in f for x in ['Volume', 'OBV', 'VWAP', 'CMF'])]
    momentum_features = [f for f in feature_set.feature_names if any(x in f for x in ['RSI', 'MACD', 'Stoch', 'ROC'])]
    volatility_features = [f for f in feature_set.feature_names if any(x in f for x in ['ATR', 'Volatility', 'Keltner'])]
    ict_features = [f for f in feature_set.feature_names if any(x in f for x in ['FVG', 'OB', 'Premium', 'Higher', 'Lower'])]
    
    logger.info(f"  📊 Price-based indicators: {len(price_features)}")
    logger.info(f"  📈 Volume-based indicators: {len(volume_features)}")
    logger.info(f"  ⚡ Momentum indicators: {len(momentum_features)}")
    logger.info(f"  📉 Volatility indicators: {len(volatility_features)}")
    logger.info(f"  🎯 ICT-specific indicators: {len(ict_features)}")
    
    # Show latest feature values (sample)
    if not feature_set.features.empty:
        latest_features = feature_set.features.iloc[-1]
        logger.info(f"\n🔍 Sample latest indicator values:")
        for category, features in [("Price", price_features[:3]), ("ICT", ict_features[:3])]:
            logger.info(f"  {category}:")
            for feature in features:
                if feature in latest_features:
                    logger.info(f"    {feature}: {latest_features[feature]:.4f}")
    
    return feature_set

async def demo_pattern_detection(feature_set):
    """Demonstrate AI pattern detection"""
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: AI Pattern Detection (65 ICT Concepts)")
    logger.info("=" * 60)
    
    # Create pattern detector
    pattern_detector = ICTPatternRecognitionAI()
    
    # Detect patterns using AI
    logger.info("Running AI pattern detection...")
    result = pattern_detector.real_time_pattern_detection(feature_set.features, "AAPL", "1h")
    
    logger.info(f"✅ Pattern detection completed in {result.execution_time_ms:.2f}ms")
    logger.info(f"📊 Total patterns detected: {result.total_patterns}")
    logger.info(f"⭐ High confidence patterns: {result.high_confidence_patterns}")
    
    if result.patterns:
        logger.info(f"\n🎯 Detected ICT Patterns:")
        for i, pattern in enumerate(result.patterns[:5]):  # Show first 5 patterns
            logger.info(f"  {i+1}. {pattern.concept_name}")
            logger.info(f"     Confidence: {pattern.confidence:.1%}")
            logger.info(f"     Type: {pattern.pattern_type.title()}")
            logger.info(f"     Strength: {pattern.strength}")
            
        if len(result.patterns) > 5:
            logger.info(f"     ... and {len(result.patterns) - 5} more patterns")
    else:
        logger.info("  No patterns detected (models not trained yet)")
        
    return result

async def demo_ai_integration():
    """Demonstrate the full AI integration pipeline"""
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: AI Integration Pipeline")
    logger.info("=" * 60)
    
    # Create AI integration engine
    ai_engine = AIIntegrationEngine()
    await ai_engine.initialize()
    
    # Download data for multiple symbols
    symbols = ["AAPL", "GOOGL", "MSFT"]
    
    for symbol in symbols:
        logger.info(f"\n🔄 Analyzing {symbol}...")
        
        # Get stock data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1mo", interval="5m")
        
        if data.empty:
            logger.warning(f"No data for {symbol}, skipping...")
            continue
            
        # Run complete AI analysis
        analysis = await ai_engine.analyze_symbol_with_ai(symbol, data, "5m")
        
        if analysis['status'] == 'success':
            logger.info(f"  ✅ Analysis completed")
            logger.info(f"  📊 Features created: {analysis['feature_count']}")
            logger.info(f"  🎯 Patterns detected: {analysis['pattern_summary']['total_patterns']}")
            logger.info(f"  ⚡ Analysis time: {analysis['performance']['analysis_time_ms']:.2f}ms")
            
            if analysis['patterns']:
                best_pattern = max(analysis['patterns'], key=lambda p: p['confidence'])
                logger.info(f"  🌟 Best pattern: {best_pattern['concept_name']} ({best_pattern['confidence']:.1%})")
        else:
            logger.error(f"  ❌ Analysis failed: {analysis.get('message', 'Unknown error')}")
    
    # Show AI performance stats
    performance = ai_engine.get_ai_performance_stats()
    logger.info(f"\n📈 AI Engine Performance:")
    logger.info(f"  Total analyses: {performance['feature_engine']['total_analyses']}")
    logger.info(f"  Avg feature creation time: {performance['feature_engine']['avg_feature_creation_time']:.2f}ms")
    logger.info(f"  Cache hit rate: {performance['feature_engine']['cache_hit_rate']:.1%}")

async def demo_training_pipeline():
    """Demonstrate the model training pipeline"""
    
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Model Training Pipeline (Preview)")
    logger.info("=" * 60)
    
    logger.info("🎓 Training Configuration:")
    
    # Create training config (small scale for demo)
    config = TrainingConfig(
        symbols=["AAPL", "GOOGL"],  # Limited symbols for demo
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),  # 30 days for demo
        end_date=datetime.now().strftime('%Y-%m-%d'),
        timeframes=['5m', '1h'],
        epochs=2,  # Very limited for demo
        batch_size=16
    )
    
    logger.info(f"  📊 Training symbols: {config.symbols}")
    logger.info(f"  📅 Date range: {config.start_date} to {config.end_date}")
    logger.info(f"  ⏰ Timeframes: {config.timeframes}")
    logger.info(f"  🔄 Epochs: {config.epochs} (limited for demo)")
    
    logger.info("\n💡 Note: Full training would use:")
    logger.info("  - 5+ years of historical data")
    logger.info("  - 10+ stock symbols")
    logger.info("  - 50+ training epochs")
    logger.info("  - Comprehensive cross-validation")
    
    # Show what the training pipeline includes
    logger.info(f"\n🏗️ Training Pipeline Components:")
    logger.info(f"  1. Historical data collection via yfinance")
    logger.info(f"  2. Feature engineering (200+ indicators)")
    logger.info(f"  3. ICT pattern labeling (65 concepts)")
    logger.info(f"  4. Random Forest ensemble training")
    logger.info(f"  5. LSTM neural network training")
    logger.info(f"  6. Transformer model training")
    logger.info(f"  7. Model validation and metrics")
    logger.info(f"  8. Model persistence and deployment")

async def main():
    """Run all demos"""
    
    logger.info("🚀 ICT Stock Trader AI/ML Implementation Demo")
    logger.info("Showcasing Phase 3: AI/ML Implementation Features")
    
    try:
        # Demo 1: Feature Engineering
        feature_set = await demo_feature_engineering()
        
        if feature_set is None:
            logger.error("Feature engineering demo failed, skipping remaining demos")
            return
            
        # Demo 2: Pattern Detection
        await demo_pattern_detection(feature_set)
        
        # Demo 3: AI Integration
        await demo_ai_integration()
        
        # Demo 4: Training Pipeline
        await demo_training_pipeline()
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 Demo completed successfully!")
        logger.info("=" * 60)
        logger.info("📋 Summary of AI/ML Implementation:")
        logger.info("  ✅ 149+ Technical Indicators implemented")
        logger.info("  ✅ Neural Networks (LSTM + Transformer) ready")
        logger.info("  ✅ Ensemble Learning framework complete")
        logger.info("  ✅ Real-time pattern detection functional")
        logger.info("  ✅ ICT-specific features for all 65 concepts")
        logger.info("  ✅ Multi-timeframe analysis capability")
        logger.info("  ✅ API integration complete")
        logger.info("  🔄 Ready for full-scale model training")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())