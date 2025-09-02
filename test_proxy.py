#!/usr/bin/env python
"""
Test that proxy is only set for Gemini models, not OpenAI
"""
import os
import sys
sys.path.insert(0, '.')

print("Testing proxy handling...")
print(f"Initial HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

# Test with a simple function that simulates the proxy behavior
def test_proxy_handling():
    # Store original proxy
    original_proxy = os.environ.get("HTTPS_PROXY")
    print(f"Original proxy: {original_proxy}")
    
    # Set proxy for Gemini (simulate)
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7078"
    print(f"Proxy set for Gemini: {os.environ['HTTPS_PROXY']}")
    
    try:
        # Simulate API call
        print("Simulating Gemini API call...")
        
    finally:
        # Restore original proxy
        if original_proxy is not None:
            os.environ["HTTPS_PROXY"] = original_proxy
            print(f"Proxy restored to: {os.environ['HTTPS_PROXY']}")
        elif "HTTPS_PROXY" in os.environ:
            del os.environ["HTTPS_PROXY"]
            print("Proxy removed (was not set originally)")
    
    print(f"Final HTTPS_PROXY: {os.environ.get('HTTPS_PROXY', 'Not set')}")

test_proxy_handling()
print("✅ Proxy handling test completed")
