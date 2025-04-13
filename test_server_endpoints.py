#!/usr/bin/env python3
"""
Test script for MLX-Audio TTS server endpoints.
This script will test the /voices and /synthesize_pcm endpoints to make sure they are working correctly.
"""

import argparse
import json
import sys
import time
import requests
from typing import Dict, List, Optional, Any

# Default server URL
DEFAULT_URL = "http://127.0.0.1:5000"

def test_voices_endpoint(base_url: str, verbose: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Test the /voices endpoint.
    
    Args:
        base_url: The base URL of the server
        verbose: Whether to print verbose output
        
    Returns:
        List of available voices if successful, None otherwise
    """
    voices_url = f"{base_url}/voices"
    print(f"\n=== Testing /voices endpoint ({voices_url}) ===")
    
    try:
        start_time = time.time()
        response = requests.get(voices_url, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            voices = response.json()
            if voices and isinstance(voices, list):
                print(f"✅ SUCCESS: Received {len(voices)} voices in {elapsed:.2f}s")
                if verbose:
                    for i, voice in enumerate(voices, 1):
                        print(f"  {i}. {voice.get('name', 'Unnamed')} (ID: {voice.get('id', 'unknown')})")
                        print(f"     Description: {voice.get('description', 'No description')}")
                return voices
            else:
                print(f"❌ ERROR: Received empty or invalid voice list: {voices}")
                print(f"Server responded with valid JSON but no voices were returned.")
                return None
        else:
            print(f"❌ ERROR: HTTP {response.status_code}: {response.reason}")
            print(f"Response content: {response.text[:200]}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Failed to connect to {voices_url}")
        print(f"Is the server running? Check with 'ps aux | grep flask'")
        return None
    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Request timed out when connecting to {voices_url}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return None

def test_synthesize_endpoint(base_url: str, voice_id: str, verbose: bool = False) -> bool:
    """
    Test the /synthesize_pcm endpoint with a short text.
    
    Args:
        base_url: The base URL of the server
        voice_id: The ID of the voice to use
        verbose: Whether to print verbose output
        
    Returns:
        True if successful, False otherwise
    """
    synth_url = f"{base_url}/synthesize_pcm"
    print(f"\n=== Testing /synthesize_pcm endpoint ({synth_url}) ===")
    
    # Short test text to avoid long processing time
    test_text = "This is a test of the text-to-speech system."
    
    request_data = {
        "text": test_text,
        "voice": voice_id,
        "speed": 1.0
    }
    
    if verbose:
        print(f"Request data: {json.dumps(request_data, indent=2)}")
    
    try:
        print(f"Sending request with voice: {voice_id}")
        start_time = time.time()
        
        response = requests.post(
            synth_url,
            json=request_data,
            timeout=30,
            stream=True  # Use streaming to handle potentially large audio response
        )
        
        if response.status_code == 200:
            # Read first chunk to make sure data is flowing
            content_size = 0
            for chunk in response.iter_content(chunk_size=4096):
                content_size += len(chunk)
                if verbose:
                    print(f"Received audio chunk: {len(chunk)} bytes")
                # Only read a small amount to verify it's working
                if content_size > 8192:
                    break
                
            elapsed = time.time() - start_time
            
            # Check if audio headers were provided
            sample_rate = response.headers.get('X-Audio-Sample-Rate')
            channels = response.headers.get('X-Audio-Channels')
            bit_depth = response.headers.get('X-Audio-Bit-Depth')
            
            print(f"✅ SUCCESS: Received audio data ({content_size} bytes) in {elapsed:.2f}s")
            if sample_rate and channels and bit_depth:
                print(f"  Audio details: {sample_rate}Hz, {channels} channels, {bit_depth}-bit PCM")
            
            return True
        else:
            print(f"❌ ERROR: HTTP {response.status_code}")
            print(f"Response content: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ ERROR: Failed to connect to {synth_url}")
        print(f"Is the server running? Check with 'ps aux | grep flask'")
        return False
    except requests.exceptions.Timeout:
        print(f"❌ ERROR: Request timed out when connecting to {synth_url}")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Test the MLX-Audio TTS server endpoints")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Base URL of the server (default: {DEFAULT_URL})")
    parser.add_argument("--voice", help="Voice ID to use for synthesis test (default: first available voice)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    print(f"Testing MLX-Audio TTS server at: {args.url}")
    
    # First test the /voices endpoint
    voices = test_voices_endpoint(args.url, args.verbose)
    
    if not voices:
        print("\n❌ Voices endpoint test failed. Cannot proceed with synthesis test.")
        sys.exit(1)
    
    # Determine which voice to use for the synthesis test
    voice_id = args.voice
    if not voice_id:
        if voices:
            voice_id = voices[0]["id"]
            print(f"\nUsing first available voice: {voice_id}")
        else:
            print("\n❌ No voices available. Cannot proceed with synthesis test.")
            sys.exit(1)
    
    # Now test the /synthesize_pcm endpoint
    synthesis_success = test_synthesize_endpoint(args.url, voice_id, args.verbose)
    
    # Final summary
    print("\n=== Test Summary ===")
    print(f"Voices endpoint: {'✅ Passed' if voices else '❌ Failed'}")
    print(f"Synthesis endpoint: {'✅ Passed' if synthesis_success else '❌ Failed'}")
    
    if voices and synthesis_success:
        print("\n✅ All tests passed! Server is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the server logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()
