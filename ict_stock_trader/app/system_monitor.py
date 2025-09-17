"""
System Monitoring and Performance Optimization for ICT Trading AI
Includes logging, caching, performance metrics, and health monitoring
"""
import logging
import time
import psutil
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from functools import wraps
import asyncio
from dataclasses import dataclass
import json

# Configure advanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ict_trading_ai.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for system monitoring"""
    function_name: str
    execution_time: float
    memory_usage: float
    cpu_usage: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None

class ICTSystemMonitor:
    """System monitoring and performance optimization"""
    
    def __init__(self):
        self.metrics_cache = {}
        self.performance_history = []
        self.cache_store = {}
        self.cache_ttl = {}
        self.default_cache_duration = 300  # 5 minutes
        
    def performance_monitor(self, cache_duration: Optional[int] = None):
        """Decorator to monitor function performance and cache results"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                start_cpu = psutil.Process().cpu_percent()
                
                function_name = f"{func.__module__}.{func.__name__}"
                
                # Check cache first
                cache_key = self._generate_cache_key(function_name, args, kwargs)
                cached_result = self._get_from_cache(cache_key)
                
                if cached_result is not None:
                    logger.debug(f"Cache hit for {function_name}")
                    return cached_result
                
                try:
                    # Execute function
                    result = func(*args, **kwargs)
                    
                    # Calculate performance metrics
                    end_time = time.time()
                    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                    end_cpu = psutil.Process().cpu_percent()
                    
                    execution_time = end_time - start_time
                    memory_delta = end_memory - start_memory
                    cpu_delta = end_cpu - start_cpu
                    
                    # Log performance metrics
                    metrics = PerformanceMetrics(
                        function_name=function_name,
                        execution_time=execution_time,
                        memory_usage=memory_delta,
                        cpu_usage=cpu_delta,
                        timestamp=datetime.now(),
                        success=True
                    )
                    
                    self._record_metrics(metrics)
                    
                    # Cache result
                    duration = cache_duration or self.default_cache_duration
                    self._set_cache(cache_key, result, duration)
                    
                    logger.debug(f"{function_name} executed in {execution_time:.3f}s")
                    
                    return result
                    
                except Exception as e:
                    # Log error metrics
                    end_time = time.time()
                    execution_time = end_time - start_time
                    
                    metrics = PerformanceMetrics(
                        function_name=function_name,
                        execution_time=execution_time,
                        memory_usage=0,
                        cpu_usage=0,
                        timestamp=datetime.now(),
                        success=False,
                        error_message=str(e)
                    )
                    
                    self._record_metrics(metrics)
                    logger.error(f"Error in {function_name}: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def _generate_cache_key(self, function_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        # Simple cache key generation (could be improved with better hashing)
        key_parts = [function_name]
        
        # Add args to key
        for arg in args:
            if hasattr(arg, 'shape'):  # pandas DataFrame/Series
                key_parts.append(f"df_{arg.shape}_{hash(str(arg.iloc[0] if len(arg) > 0 else ''))}")
            else:
                key_parts.append(str(hash(str(arg))))
        
        # Add kwargs to key
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}_{hash(str(v))}")
        
        return "_".join(key_parts)
    
    def _get_from_cache(self, cache_key: str):
        """Get result from cache if not expired"""
        if cache_key in self.cache_store:
            if cache_key in self.cache_ttl:
                if datetime.now() < self.cache_ttl[cache_key]:
                    return self.cache_store[cache_key]
                else:
                    # Cache expired
                    del self.cache_store[cache_key]
                    del self.cache_ttl[cache_key]
        return None
    
    def _set_cache(self, cache_key: str, result, duration: int):
        """Set result in cache with TTL"""
        self.cache_store[cache_key] = result
        self.cache_ttl[cache_key] = datetime.now() + timedelta(seconds=duration)
        
        # Clean old cache entries if cache is getting large
        if len(self.cache_store) > 1000:
            self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries"""
        now = datetime.now()
        expired_keys = [
            key for key, ttl in self.cache_ttl.items()
            if now >= ttl
        ]
        
        for key in expired_keys:
            if key in self.cache_store:
                del self.cache_store[key]
            del self.cache_ttl[key]
    
    def _record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        self.performance_history.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
        
        # Log slow operations
        if metrics.execution_time > 5.0:  # 5 seconds
            logger.warning(f"Slow operation detected: {metrics.function_name} took {metrics.execution_time:.3f}s")
        
        # Log high memory usage
        if metrics.memory_usage > 100:  # 100 MB
            logger.warning(f"High memory usage: {metrics.function_name} used {metrics.memory_usage:.1f}MB")
    
    def get_system_health(self) -> Dict:
        """Get current system health metrics"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Calculate recent performance stats
        recent_metrics = [
            m for m in self.performance_history
            if m.timestamp > datetime.now() - timedelta(minutes=5)
        ]
        
        avg_execution_time = 0
        error_rate = 0
        if recent_metrics:
            avg_execution_time = sum(m.execution_time for m in recent_metrics) / len(recent_metrics)
            error_rate = sum(1 for m in recent_metrics if not m.success) / len(recent_metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "application": {
                "cache_size": len(self.cache_store),
                "metrics_recorded": len(self.performance_history),
                "avg_execution_time": avg_execution_time,
                "error_rate": error_rate
            },
            "health_status": self._calculate_health_status(cpu_percent, memory.percent, error_rate)
        }
    
    def _calculate_health_status(self, cpu_percent: float, memory_percent: float, error_rate: float) -> str:
        """Calculate overall health status"""
        if error_rate > 0.1:  # 10% error rate
            return "critical"
        elif cpu_percent > 80 or memory_percent > 85:
            return "warning"
        elif cpu_percent > 60 or memory_percent > 70:
            return "degraded"
        else:
            return "healthy"
    
    def get_performance_report(self) -> Dict:
        """Generate detailed performance report"""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        # Group metrics by function
        function_stats = {}
        for metric in self.performance_history:
            func_name = metric.function_name
            if func_name not in function_stats:
                function_stats[func_name] = {
                    "execution_times": [],
                    "memory_usage": [],
                    "success_count": 0,
                    "error_count": 0,
                    "last_executed": None
                }
            
            function_stats[func_name]["execution_times"].append(metric.execution_time)
            function_stats[func_name]["memory_usage"].append(metric.memory_usage)
            
            if metric.success:
                function_stats[func_name]["success_count"] += 1
            else:
                function_stats[func_name]["error_count"] += 1
            
            if not function_stats[func_name]["last_executed"] or metric.timestamp > function_stats[func_name]["last_executed"]:
                function_stats[func_name]["last_executed"] = metric.timestamp
        
        # Calculate statistics
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_metrics": len(self.performance_history),
            "functions": {}
        }
        
        for func_name, stats in function_stats.items():
            exec_times = stats["execution_times"]
            memory_usage = stats["memory_usage"]
            
            report["functions"][func_name] = {
                "call_count": len(exec_times),
                "success_rate": stats["success_count"] / (stats["success_count"] + stats["error_count"]),
                "avg_execution_time": sum(exec_times) / len(exec_times),
                "max_execution_time": max(exec_times),
                "min_execution_time": min(exec_times),
                "avg_memory_usage": sum(memory_usage) / len(memory_usage),
                "max_memory_usage": max(memory_usage),
                "last_executed": stats["last_executed"].isoformat() if stats["last_executed"] else None
            }
        
        return report
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache_store.clear()
        self.cache_ttl.clear()
        logger.info("Cache cleared")
    
    def optimize_performance(self):
        """Perform performance optimizations"""
        # Clean up cache
        self._cleanup_cache()
        
        # Log optimization suggestions
        report = self.get_performance_report()
        
        if "functions" in report:
            slow_functions = [
                (name, stats) for name, stats in report["functions"].items()
                if stats["avg_execution_time"] > 2.0
            ]
            
            if slow_functions:
                logger.info("Performance optimization suggestions:")
                for func_name, stats in slow_functions:
                    logger.info(f"  - {func_name}: avg {stats['avg_execution_time']:.3f}s (consider caching or optimization)")

# Global monitor instance
system_monitor = ICTSystemMonitor()

class DatabaseOptimizer:
    """Database optimization and maintenance"""
    
    def __init__(self, db_path: str = "ict_trading.db"):
        self.db_path = db_path
    
    def optimize_database(self):
        """Optimize SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Analyze database
            cursor.execute("ANALYZE")
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            # Update statistics
            cursor.execute("PRAGMA optimize")
            
            conn.close()
            logger.info("Database optimization completed")
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get database size
            cursor.execute("PRAGMA page_count")
            page_count = cursor.fetchone()[0]
            
            cursor.execute("PRAGMA page_size")
            page_size = cursor.fetchone()[0]
            
            database_size = page_count * page_size
            
            # Get table information
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            table_stats = {}
            for table in tables:
                table_name = table[0]
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                row_count = cursor.fetchone()[0]
                table_stats[table_name] = {"row_count": row_count}
            
            conn.close()
            
            return {
                "database_size_bytes": database_size,
                "database_size_mb": database_size / (1024 * 1024),
                "tables": table_stats
            }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {"error": str(e)}

# Create global instances
db_optimizer = DatabaseOptimizer()

# Health check function for FastAPI
def get_health_status():
    """Get comprehensive health status for API endpoint"""
    health = system_monitor.get_system_health()
    db_stats = db_optimizer.get_database_stats()
    
    return {
        **health,
        "database": db_stats,
        "cache_stats": {
            "cached_items": len(system_monitor.cache_store),
            "cache_hit_ratio": "N/A"  # Would need to track this separately
        }
    }