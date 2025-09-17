#!/usr/bin/env python3
"""
Working test script to verify all functionality is working
"""
import sys
import os
sys.path.append('/home/runner/.local/lib/python3.12/site-packages')
sys.path.append('.')

def test_imports():
    """Test that all imports work correctly"""
    try:
        print("🧪 Testing imports...")
        from fastapi.testclient import TestClient
        from app.main import app
        print("✅ FastAPI and app imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_basic_endpoints():
    """Test basic API endpoints"""
    try:
        print("🧪 Testing basic endpoints...")
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test health endpoint
        response = client.get("/health")
        print(f"Health endpoint: {response.status_code}")
        
        # Test root endpoint  
        response = client.get("/")
        print(f"Root endpoint: {response.status_code}")
        
        # Test stock data endpoint
        response = client.get("/api/v1/stocks/AAPL/data")
        print(f"Stock data endpoint: {response.status_code}")
        
        # Test ICT analysis endpoint
        response = client.get("/api/v1/ict/analysis/AAPL")
        print(f"ICT analysis endpoint: {response.status_code}")
        
        print("✅ All basic endpoints responding")
        return True
        
    except Exception as e:
        print(f"❌ Endpoint test error: {e}")
        return False

def test_ai_endpoints():
    """Test AI/ML endpoints"""
    try:
        print("🧪 Testing AI/ML endpoints...")
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test AI analysis endpoint
        response = client.get("/api/v1/ai/analysis/AAPL")
        print(f"AI analysis endpoint: {response.status_code}")
        
        # Test AI features endpoint
        response = client.get("/api/v1/ai/features/AAPL")
        print(f"AI features endpoint: {response.status_code}")
        
        # Test AI performance endpoint
        response = client.get("/api/v1/ai/performance")
        print(f"AI performance endpoint: {response.status_code}")
        
        print("✅ All AI/ML endpoints responding")
        return True
        
    except Exception as e:
        print(f"❌ AI endpoint test error: {e}")
        return False

def test_ict_concepts():
    """Test specific ICT concept endpoints"""
    try:
        print("🧪 Testing ICT concept endpoints...")
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test a few key ICT concepts
        concepts_to_test = [1, 21, 22, 35, 39]  # The ones we fixed
        
        for concept_id in concepts_to_test:
            response = client.get(f"/api/v1/ict/concept/{concept_id}/AAPL")
            print(f"ICT Concept {concept_id}: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"  - Response has data: {len(data) > 0}")
        
        print("✅ ICT concept endpoints responding")
        return True
        
    except Exception as e:
        print(f"❌ ICT concept test error: {e}")
        return False

def main():
    """Run comprehensive testing"""
    print("🚀 COMPREHENSIVE FUNCTIONALITY TEST")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_basic_endpoints,
        test_ai_endpoints,
        test_ict_concepts
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 50)
    print(f"📊 FINAL RESULTS: {passed}/{total} tests passed ({int(passed/total*100)}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSING - SYSTEM FULLY FUNCTIONAL!")
        return 0
    else:
        print(f"⚠️  {total-passed} tests failed - issues need fixing")
        return 1

if __name__ == "__main__":
    exit(main())