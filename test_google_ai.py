#!/usr/bin/env python
"""
Test Google AI API connection based on the reference code
"""
import os
import google.generativeai as genai

# Set up proxy as in reference
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7078"

# Configure API key
api_key = "AIzaSyCXhoRyRmp_6Rpbp9eZjjwEvE11KrKIJII"
genai.configure(api_key=api_key)

print("Testing Google AI API connection...")
print(f"Proxy: {os.environ.get('HTTPS_PROXY', 'Not set')}")

try:
    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content("Hello, this is a test. Please respond with 'Test successful!'")
    print("✅ Google AI API test successful!")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"❌ Google AI API test failed: {e}")
