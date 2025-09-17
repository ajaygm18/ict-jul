import pandas as pd
import pytest
from ict_stock_trader.app.ict_engine.risk_management import StockRiskManagementEngine

@pytest.fixture
def risk_analyzer():
    """Provides an instance of the StockRiskManagementEngine."""
    return StockRiskManagementEngine()

def test_rrr_optimization(risk_analyzer):
    result = risk_analyzer.concept_34_rrr_optimization(entry_price=100, stop_loss=98, take_profit=106)
    assert result['rrr'] == 3.0

def test_position_sizing(risk_analyzer):
    result = risk_analyzer.concept_35_position_sizing(account_size=10000, risk_per_trade_percent=1, entry_price=50, stop_loss=48)
    # Risk $100. Risk per share is $2. Can buy 50 shares.
    assert result['shares'] == pytest.approx(50)
    assert result['position_value'] == pytest.approx(2500)

def test_drawdown_control(risk_analyzer):
    equity_curve = [1000, 1100, 1200, 1050]
    result = risk_analyzer.concept_36_drawdown_control(equity_curve)
    # Peak is 1200, trough is 1050. Drawdown is 150. (150/1200)*100 = 12.5%
    assert result['drawdown_percent'] == pytest.approx(12.5)

def test_compounding_models(risk_analyzer):
    result = risk_analyzer.concept_37_compounding_models(initial_principal=1000, rate_of_return=10, periods=2)
    # 1000 * 1.10 = 1100. 1100 * 1.10 = 1210.
    assert result['final_value'] == pytest.approx(1210)

def test_daily_loss_limits(risk_analyzer):
    # Not stopped
    result1 = risk_analyzer.concept_38_daily_loss_limits(daily_pnl=-99, max_daily_loss=100)
    assert result1['stop_trading'] == False
    # Stopped
    result2 = risk_analyzer.concept_38_daily_loss_limits(daily_pnl=-101, max_daily_loss=100)
    assert result2['stop_trading'] == True
