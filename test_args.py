#!/usr/bin/env python3
"""Test argument parsing to verify the fix for modify-existing"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sciresearch_workflow import parse_args

# Test 1: --modify-existing with existing paper
print("Test 1: --modify-existing with existing paper")
sys.argv = [
    'test_args.py',
    '--modify-existing',
    '--output-dir', 'c:\\Users\\Lenovo\\papers\\ag-qec',
    '--model', 'gemini-1.5-pro',
    '--max-iterations', '1'
]

try:
    args = parse_args()
    print(f"✅ Success! Topic: {getattr(args, 'topic', 'None')}")
    print(f"Field: {getattr(args, 'field', 'None')}")
    print(f"Question: {getattr(args, 'question', 'None')}")
    print(f"Modify existing: {getattr(args, 'modify_existing', 'None')}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "="*50 + "\n")

# Test 2: Regular workflow (should prompt for inputs - but we'll provide them)
print("Test 2: Regular workflow with all args provided")
sys.argv = [
    'test_args.py',
    '--topic', 'Test Topic',
    '--field', 'Test Field', 
    '--question', 'Test Question',
    '--model', 'gemini-1.5-pro'
]

try:
    args = parse_args()
    print(f"✅ Success! Topic: {getattr(args, 'topic', 'None')}")
    print(f"Field: {getattr(args, 'field', 'None')}")
    print(f"Question: {getattr(args, 'question', 'None')}")
    print(f"Modify existing: {getattr(args, 'modify_existing', 'None')}")
except Exception as e:
    print(f"❌ Error: {e}")
