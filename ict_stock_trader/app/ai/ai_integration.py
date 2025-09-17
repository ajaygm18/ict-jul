"""
AI Integration Module
Integrates AI/ML components with the main ICT application
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import asyncio
from pathlib import Path

from .feature_engineer import TechnicalIndicatorEngine, MultiTimeframeFeatureEngine, FeatureSet
from .pattern_detector import ICTPatternRecognitionAI, Pattern, PatternDetectionResult
from .model_trainer import ModelTrainer, TrainingConfig, ModelPerformance

logger = logging.getLogger(__name__)

class AIIntegrationEngine:
    """Main integration engine for AI/ML components"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engine = TechnicalIndicatorEngine()
        self.multi_tf_engine = MultiTimeframeFeatureEngine()
        self.pattern_detector = ICTPatternRecognitionAI(str(self.model_dir))
        
        # Cache for features
        self.feature_cache = {}
        self.pattern_cache = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_analyses': 0,
            'avg_feature_creation_time': 0.0,
            'avg_pattern_detection_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
    async def initialize(self):
        """Initialize AI components"""
        try:
            # Load pre-trained models
            self.pattern_detector.load_models()
            logger.info("AI components initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AI components: {e}")
            
    def create_features_for_symbol(self, symbol: str, stock_data: pd.DataFrame, timeframe: str = '5m') -> FeatureSet:
        """Create comprehensive features for a symbol"""
        
        cache_key = f"{symbol}_{timeframe}_{hash(str(stock_data.index[-1]))}"
        
        # Check cache
        if cache_key in self.feature_cache:
            self.performance_stats['cache_hit_rate'] = (
                self.performance_stats['cache_hit_rate'] * 0.9 + 1.0 * 0.1
            )
            return self.feature_cache[cache_key]
            
        start_time = datetime.now()
        
        try:
            # Create comprehensive features
            feature_set = self.feature_engine.create_comprehensive_features(
                stock_data, symbol, timeframe
            )
            
            # Cache the result
            self.feature_cache[cache_key] = feature_set
            
            # Update performance stats
            creation_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['avg_feature_creation_time'] = (
                self.performance_stats['avg_feature_creation_time'] * 0.9 + creation_time * 0.1
            )
            self.performance_stats['cache_hit_rate'] = (
                self.performance_stats['cache_hit_rate'] * 0.9 + 0.0 * 0.1
            )
            
            logger.info(f"Created {len(feature_set.feature_names)} features for {symbol} in {creation_time:.2f}ms")
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error creating features for {symbol}: {e}")
            # Return empty feature set
            return FeatureSet(
                timeframe=timeframe,
                features=pd.DataFrame(),
                feature_names=[],
                creation_timestamp=datetime.now(),
                symbol=symbol
            )
            
    def detect_patterns_for_symbol(self, symbol: str, features: FeatureSet, timeframe: str = '5m') -> PatternDetectionResult:
        """Detect ICT patterns for a symbol"""
        
        cache_key = f"{symbol}_{timeframe}_patterns_{hash(str(features.creation_timestamp))}"
        
        # Check cache
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]
            
        start_time = datetime.now()
        
        try:
            # Detect patterns using AI
            result = self.pattern_detector.real_time_pattern_detection(
                features.features, symbol, timeframe
            )
            
            # Cache the result
            self.pattern_cache[cache_key] = result
            
            # Update performance stats
            detection_time = (datetime.now() - start_time).total_seconds() * 1000
            self.performance_stats['avg_pattern_detection_time'] = (
                self.performance_stats['avg_pattern_detection_time'] * 0.9 + detection_time * 0.1
            )
            self.performance_stats['total_analyses'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting patterns for {symbol}: {e}")
            # Return empty result
            return PatternDetectionResult(
                symbol=symbol,
                timestamp=datetime.now(),
                patterns=[],
                total_patterns=0,
                high_confidence_patterns=0,
                execution_time_ms=0.0
            )
            
    async def analyze_symbol_with_ai(self, symbol: str, stock_data: pd.DataFrame, timeframe: str = '5m') -> Dict[str, Any]:
        """Complete AI analysis for a symbol"""
        
        start_time = datetime.now()
        
        try:
            # Step 1: Create features
            feature_set = self.create_features_for_symbol(symbol, stock_data, timeframe)
            
            if feature_set.features.empty:
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'status': 'error',
                    'message': 'No features could be created',
                    'patterns': [],
                    'feature_count': 0,
                    'analysis_time_ms': 0
                }
                
            # Step 2: Detect patterns
            pattern_result = self.detect_patterns_for_symbol(symbol, feature_set, timeframe)
            
            # Step 3: Prepare response
            analysis_time = (datetime.now() - start_time).total_seconds() * 1000
            
            response = {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'success',
                'timestamp': start_time.isoformat(),
                'feature_count': len(feature_set.feature_names),
                'patterns': [
                    {
                        'concept_name': p.concept_name,
                        'concept_number': p.concept_number,
                        'confidence': p.confidence,
                        'pattern_type': p.pattern_type,
                        'strength': p.strength,
                        'supporting_evidence': p.supporting_evidence,
                        'timestamp': p.timestamp.isoformat()
                    } for p in pattern_result.patterns
                ],
                'pattern_summary': {
                    'total_patterns': pattern_result.total_patterns,
                    'high_confidence_patterns': pattern_result.high_confidence_patterns,
                    'avg_confidence': np.mean([p.confidence for p in pattern_result.patterns]) if pattern_result.patterns else 0.0
                },
                'performance': {
                    'analysis_time_ms': analysis_time,
                    'feature_creation_time_ms': self.performance_stats['avg_feature_creation_time'],
                    'pattern_detection_time_ms': pattern_result.execution_time_ms
                },
                'ai_metrics': self.pattern_detector.get_detection_metrics()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error in AI analysis for {symbol}: {e}")
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'error',
                'message': str(e),
                'patterns': [],
                'feature_count': 0,
                'analysis_time_ms': (datetime.now() - start_time).total_seconds() * 1000
            }
            
    async def train_models_for_symbols(self, symbols: List[str], start_date: str = None, end_date: str = None) -> Dict[str, ModelPerformance]:
        """Train AI models on historical data"""
        
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')  # 5 years ago
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Starting model training for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Create training configuration
        config = TrainingConfig(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframes=['5m', '1h', '1d'],
            sequence_length=50,
            epochs=50,  # Reduced for demo
            batch_size=32,
            learning_rate=0.001
        )
        
        # Initialize trainer
        trainer = ModelTrainer(config, str(self.model_dir))
        
        try:
            # Train all models
            performances = trainer.train_all_models()
            
            # Reload the trained models in pattern detector
            self.pattern_detector.load_models()
            
            logger.info("Model training completed successfully")
            return performances
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {}
            
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from trained models"""
        
        try:
            importance_data = {}
            
            # Random Forest feature importance
            if 'random_forest' in self.pattern_detector.models:
                rf_model = self.pattern_detector.models['random_forest']
                if hasattr(rf_model, 'feature_importances_'):
                    importance_data['random_forest'] = rf_model.feature_importances_.tolist()
                    
            return {
                'status': 'success',
                'feature_importance': importance_data,
                'total_features': len(importance_data.get('random_forest', []))
            }
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'feature_importance': {},
                'total_features': 0
            }
            
    def get_ai_performance_stats(self) -> Dict[str, Any]:
        """Get AI performance statistics"""
        
        return {
            'feature_engine': self.performance_stats,
            'pattern_detection': self.pattern_detector.get_detection_metrics(),
            'cache_stats': {
                'feature_cache_size': len(self.feature_cache),
                'pattern_cache_size': len(self.pattern_cache)
            },
            'model_info': {
                'models_loaded': len(self.pattern_detector.models),
                'is_trained': self.pattern_detector.is_trained
            }
        }
        
    def clear_cache(self):
        """Clear feature and pattern caches"""
        self.feature_cache.clear()
        self.pattern_cache.clear()
        logger.info("AI caches cleared")

# Global AI integration instance
ai_engine = AIIntegrationEngine()

# Export main classes and instance
__all__ = ['AIIntegrationEngine', 'ai_engine']