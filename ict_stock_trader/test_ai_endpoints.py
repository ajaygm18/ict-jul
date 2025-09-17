#!/usr/bin/env python3
"""
Test script for the new AI/ML API endpoints
"""

import asyncio
import aiohttp
import json
from datetime import datetime

async def test_ai_endpoints():
    """Test the new AI/ML API endpoints"""
    
    base_url = "http://localhost:8000"
    
    # Test endpoints
    endpoints = [
        {
            "name": "Health Check",
            "url": f"{base_url}/health",
            "method": "GET"
        },
        {
            "name": "AI Analysis for AAPL",
            "url": f"{base_url}/api/v1/ai/analysis/AAPL",
            "method": "GET"
        },
        {
            "name": "Technical Indicators for AAPL",
            "url": f"{base_url}/api/v1/ai/features/AAPL",
            "method": "GET"
        },
        {
            "name": "AI Performance Stats",
            "url": f"{base_url}/api/v1/ai/performance",
            "method": "GET"
        }
    ]
    
    print("🧪 Testing AI/ML API Endpoints")
    print("=" * 50)
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\n🔍 Testing: {endpoint['name']}")
            print(f"   URL: {endpoint['url']}")
            
            try:
                if endpoint['method'] == 'GET':
                    async with session.get(endpoint['url']) as response:
                        if response.status == 200:
                            data = await response.json()
                            print(f"   ✅ Status: {response.status}")
                            
                            # Show relevant data for each endpoint
                            if 'ai/analysis' in endpoint['url']:
                                if 'feature_count' in data:
                                    print(f"   📊 Features: {data['feature_count']}")
                                if 'patterns' in data:
                                    print(f"   🎯 Patterns: {len(data['patterns'])}")
                                    
                            elif 'ai/features' in endpoint['url']:
                                if 'feature_count' in data:
                                    print(f"   📊 Total indicators: {data['feature_count']}")
                                    
                            elif 'ai/performance' in endpoint['url']:
                                if 'performance' in data:
                                    perf = data['performance']
                                    if 'feature_engine' in perf:
                                        print(f"   ⚡ Total analyses: {perf['feature_engine']['total_analyses']}")
                                        
                            elif 'health' in endpoint['url']:
                                print(f"   💚 System: {data.get('status', 'unknown')}")
                                
                        else:
                            print(f"   ❌ Status: {response.status}")
                            error_text = await response.text()
                            print(f"   Error: {error_text[:100]}...")
                            
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
    
    print(f"\n🎉 API endpoint testing completed!")

if __name__ == "__main__":
    print("Note: This script requires the FastAPI server to be running.")
    print("Start the server with: uvicorn app.main:app --reload")
    print("\nPress Enter to continue with testing or Ctrl+C to exit...")
    try:
        input()
        asyncio.run(test_ai_endpoints())
    except KeyboardInterrupt:
        print("\nTesting cancelled.")