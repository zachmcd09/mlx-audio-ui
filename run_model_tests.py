#!/usr/bin/env python3
"""
Test Runner for TTS Model Initialization

This script runs the tests for Kokoro and Bark model initialization and reports the results.
"""
import os
import sys
import subprocess
import time
from datetime import datetime

def run_command(command, description):
    """Run a command and return whether it succeeded."""
    print(f"\n{'=' * 80}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'=' * 80}")
    
    start_time = time.time()
    try:
        result = subprocess.run(command, shell=True, check=False, capture_output=True, text=True)
        success = result.returncode == 0
        
        # Print output
        if result.stdout:
            print("\nSTDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        duration = time.time() - start_time
        
        print(f"\nRESULT: {'SUCCESS' if success else 'FAILURE'}")
        print(f"DURATION: {duration:.2f} seconds")
        
        return success
    except Exception as e:
        print(f"\nERROR: {e}")
        return False

def main():
    """Run all tests."""
    print(f"TTS Model Initialization Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Working directory:", os.getcwd())
    
    # Create results log file
    with open("model_test_results.log", "w") as f:
        f.write(f"TTS Model Initialization Tests - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Working directory: {os.getcwd()}\n\n")
    
    tests = [
        {
            "name": "Kokoro Model Test (HF Config)",
            "command": "python test_model_initialization.py --model kokoro --use-hf-config",
            "description": "Testing Kokoro model initialization using HuggingFace config"
        },
        {
            "name": "Kokoro Model Test (Manual Config)",
            "command": "python test_model_initialization.py --model kokoro",
            "description": "Testing Kokoro model initialization using manual config"
        },
        {
            "name": "Bark Model Test",
            "command": "python test_model_initialization.py --model bark",
            "description": "Testing Bark model initialization"
        },
        {
            "name": "Encodec Parameter Debug (Model Only)",
            "command": "python debug_encodec_params.py --model-only",
            "description": "Analyzing Encodec model structure without weights"
        },
        {
            "name": "Full App Test",
            "command": "FLASK_APP=app.py flask run --port 5050 --no-reload --no-debugger | grep -q 'TTS pipelines initialization complete.' && echo 'App startup succeeded' || echo 'App startup failed'",
            "description": "Testing full app initialization (starts the Flask app and checks for successful initialization)"
        }
    ]
    
    results = {}
    
    for test in tests:
        print(f"\n\n{'#' * 80}")
        print(f"# Running Test: {test['name']}")
        print(f"{'#' * 80}")
        
        success = run_command(test["command"], test["description"])
        results[test["name"]] = success
        
        # Append to log file
        with open("model_test_results.log", "a") as f:
            f.write(f"\n{test['name']}: {'SUCCESS' if success else 'FAILURE'}\n")
            f.write(f"Command: {test['command']}\n")
            f.write(f"Description: {test['description']}\n")
            f.write("-" * 40 + "\n")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_success = True
    for name, success in results.items():
        print(f"{name}: {'SUCCESS' if success else 'FAILURE'}")
        all_success = all_success and success
    
    print("\nOVERALL RESULT:", "SUCCESS" if all_success else "FAILURE")
    
    # Append summary to log file
    with open("model_test_results.log", "a") as f:
        f.write("\n\nTEST SUMMARY\n")
        f.write("=" * 40 + "\n")
        for name, success in results.items():
            f.write(f"{name}: {'SUCCESS' if success else 'FAILURE'}\n")
        f.write(f"\nOVERALL RESULT: {'SUCCESS' if all_success else 'FAILURE'}\n")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())
