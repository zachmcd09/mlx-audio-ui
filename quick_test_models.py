#!/usr/bin/env python3
"""
Quick test script for Kokoro and Bark model initialization.

This script is a simplified version of the full test suite, designed to quickly
verify that the fixes we've implemented are working correctly.
"""
import time
import sys

from model_init_utils import initialize_kokoro_model, initialize_bark_model


def test_kokoro_model():
    """Test Kokoro model initialization and simple inference."""
    print("\n=== Testing Kokoro Model ===")
    
    start_time = time.time()
    try:
        # Initialize model
        print("Initializing Kokoro model...")
        model, config = initialize_kokoro_model("prince-canuma/Kokoro-82M")
        
        # Create pipeline
        print("Creating Kokoro pipeline...")
        from mlx_audio.tts.models.kokoro import KokoroPipeline
        pipeline = KokoroPipeline(
            lang_code="a",
            model=model,
            repo_id="prince-canuma/Kokoro-82M"
        )
        
        # Test with simple inference
        print("Testing inference...")
        voice = "af_heart"  # Default voice
        text = "Hello, this is a test."
        for i, (_, _, audio) in enumerate(pipeline(text, voice=voice)):
            print(f"  Generated audio segment {i+1}, shape: {audio.shape if audio is not None else 'None'}")
        
        elapsed = time.time() - start_time
        print(f"✅ Kokoro model test PASSED in {elapsed:.2f}s")
        return True
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Kokoro model test FAILED in {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bark_model():
    """Test Bark model initialization."""
    print("\n=== Testing Bark Model ===")
    
    start_time = time.time()
    try:
        # Initialize model with our custom adapter
        print("Initializing Bark model with parameter adapter...")
        pipeline = initialize_bark_model("suno/bark-small")
        
        # Verify pipeline initialized correctly
        print("Verifying pipeline instance...")
        assert pipeline is not None, "Pipeline initialization returned None"
        assert hasattr(pipeline, "model"), "Pipeline missing model attribute"
        assert hasattr(pipeline, "tokenizer"), "Pipeline missing tokenizer attribute"
        
        # Verify attributes that indicate successful initialization
        print("Verifying codec model...")
        from mlx_audio.codec.models.encodec.encodec import Encodec
        
        # Access the codec model from the pipeline
        if hasattr(pipeline, "codec_model"):
            assert isinstance(pipeline.codec_model, Encodec), "codec_model is not an Encodec instance"
            print("  ✓ codec_model attribute found and is an Encodec instance")
        elif hasattr(pipeline, "_codec"):
            # Some versions might use _codec instead
            assert pipeline._codec is not None, "_codec attribute is None"
            print("  ✓ _codec attribute found")
        else:
            # Try to find any attribute that might contain the codec
            # This is a more flexible approach
            found_codec = False
            for attr_name in dir(pipeline):
                if attr_name.startswith('_') and not attr_name.startswith('__'):
                    attr = getattr(pipeline, attr_name)
                    if isinstance(attr, Encodec):
                        found_codec = True
                        print(f"  ✓ Found codec in attribute: {attr_name}")
                        break
            
            assert found_codec, "Could not find codec model in pipeline attributes"
        
        print(f"Pipeline initialization and codec successfully loaded")
        
        elapsed = time.time() - start_time
        print(f"✅ Bark model test PASSED in {elapsed:.2f}s")
        return True
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"❌ Bark model test FAILED in {elapsed:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run both tests and print summary."""
    print("Quick Model Initialization Test")
    print("==============================")
    
    kokoro_success = test_kokoro_model()
    bark_success = test_bark_model()
    
    print("\n=== Test Summary ===")
    print(f"Kokoro test: {'PASSED' if kokoro_success else 'FAILED'}")
    print(f"Bark test: {'PASSED' if bark_success else 'FAILED'}")
    
    if kokoro_success and bark_success:
        print("\n✅ All tests PASSED! The fixes are working!")
        return 0
    else:
        print("\n❌ Some tests FAILED. Further debugging is needed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
