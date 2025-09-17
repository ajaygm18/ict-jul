from fastapi.testclient import TestClient
from ict_stock_trader.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to ICT Stock Trading AI Agent"}

def test_get_stock_data_success():
    response = client.get("/api/v1/stock/AAPL?period=1d&interval=5m")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    # This can fail if there's no data for AAPL, but it's a reasonable test
    assert len(response.json()) > 0

def test_get_stock_data_not_found():
    response = client.get("/api/v1/stock/THISISNOTAREALSYMBOL")
    assert response.status_code == 404
    # The detail message might change, but this is a good starting point
    assert "detail" in response.json()
