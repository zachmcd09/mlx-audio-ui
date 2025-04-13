"""
Test utility for debugging model initialization issues.

This script isolates the model initialization to make debugging easier.
"""
import sys
import json
from typing import Dict, Optional, Any
import argparse

from huggingface_hub import hf_hub_download
import mlx.core as mx
import numpy as np

# Import the model classes
from mlx_audio.tts.models.kokoro import KokoroPipeline, Model, ModelConfig
from mlx_audio.tts.models.bark import BarkPipeline


def load_model_config_from_hf(repo_id: str) -> Dict[str, Any]:
    """Load model configuration directly from HuggingFace."""
    print(f"Downloading config from HuggingFace: {repo_id}")
    config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    with open(config_file, 'r') as f:
        config_json = json.load(f)
    
    print(f"Successfully loaded config from {repo_id}")
    return config_json


def test_kokoro_initialization(use_hf_config: bool = True):
    """Test initializing the Kokoro model."""
    print("\n===== Testing Kokoro Model Initialization =====")
    repo_id = "prince-canuma/Kokoro-82M"
    
    try:
        if use_hf_config:
            # Load configuration from HuggingFace
            print("Loading configuration directly from HuggingFace...")
            config_json = load_model_config_from_hf(repo_id)
            
            # Ensure all required fields are present
            required_fields = {
                "dim_in", "dropout", "hidden_dim", "max_dur", "multispeaker",
                "n_layer", "n_mels", "n_token", "style_dim", "text_encoder_kernel_size",
                "istftnet", "plbert", "vocab"
            }
            
            # Check for missing fields
            missing_fields = required_fields - set(config_json.keys())
            if missing_fields:
                print(f"Warning: Missing fields in config: {missing_fields}")
                # Add missing fields with default values
                for field in missing_fields:
                    if field == "max_conv_dim":
                        config_json[field] = 512
                    elif field == "dropout":
                        config_json[field] = 0.1
                    # Add other defaults as needed
            
            # Create model config
            model_config = ModelConfig(**config_json)
        else:
            # Use the manually defined config (from app.py)
            print("Using manually defined configuration...")
            model_config_dict = {
                "dim_in": 64,
                "dropout": 0.2,
                "hidden_dim": 512,
                "max_dur": 50,
                "max_conv_dim": 512,  # Added - Required field
                "multispeaker": True,
                "n_layer": 3,
                "n_mels": 80,
                "n_token": 178,
                "style_dim": 128,
                "text_encoder_kernel_size": 5,
                "istftnet": {
                    "upsample_kernel_sizes": [20, 12],
                    "upsample_rates": [10, 6],
                    "gen_istft_hop_size": 5,
                    "gen_istft_n_fft": 20,
                    "resblock_dilation_sizes": [
                        [1, 3, 5],
                        [1, 3, 5],
                        [1, 3, 5]
                    ],
                    "resblock_kernel_sizes": [3, 7, 11],
                    "upsample_initial_channel": 512
                },
                "plbert": {
                    "hidden_size": 768,
                    "num_attention_heads": 12,
                    "intermediate_size": 2048,
                    "max_position_embeddings": 512,
                    "num_hidden_layers": 12,
                    "dropout": 0.1
                },
                "vocab": {
                    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13, """: 14, """: 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23, "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177
                }
            }
            # Create model config
            model_config = ModelConfig(**model_config_dict)
            
        # Now try to create model instance
        print("Creating Kokoro Model instance...")
        model = Model(model_config, repo_id=repo_id)
        print("Successfully created Kokoro Model instance!")

        # Try to create pipeline
        print("Creating Kokoro Pipeline...")
        pipeline = KokoroPipeline(
            lang_code="a",
            model=model,
            repo_id=repo_id,
        )
        print("Successfully created Kokoro Pipeline!")
        
        # Minimal test to ensure pipeline works
        print("Testing Kokoro Pipeline with a short sentence...")
        voice = "af_heart" # Default voice
        results = list(pipeline("Hello world", voice=voice))
        print(f"Pipeline returned {len(results)} result(s)")
        
        return True
    except Exception as e:
        print(f"\nError initializing Kokoro model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_bark_initialization(custom_codec_path: Optional[str] = None):
    """Test initializing the Bark model."""
    print("\n===== Testing Bark Model Initialization =====")
    repo_id = "suno/bark-small"
    
    try:
        # Create pipeline
        print("Creating Bark Pipeline...")
        pipeline = BarkPipeline(repo_id=repo_id)
        
        # If custom codec path provided, update it
        if custom_codec_path:
            print(f"Using custom codec path: {custom_codec_path}")
            # This would require modifying the internal config
            # We would need to implement this based on what's possible
            
        print("Successfully created Bark Pipeline!")
        
        # Minimal test to ensure pipeline works
        print("Testing Bark Pipeline with a short sentence...")
        voice = "en_speaker_6" # Default voice
        results = list(pipeline("Hello world", voice=voice))
        print(f"Pipeline returned {len(results)} result(s)")
        
        return True
    except Exception as e:
        print(f"\nError initializing Bark model: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_app_with_fixed_models():
    """Create a minimal test app with the fixed model initializations."""
    print("\n===== Creating Test App with Fixed Models =====")
    
    # Test Kokoro model
    kokoro_success = test_kokoro_initialization(use_hf_config=True)
    print(f"Kokoro model test {'PASSED' if kokoro_success else 'FAILED'}")
    
    # Test Bark model
    bark_success = test_bark_initialization()
    print(f"Bark model test {'PASSED' if bark_success else 'FAILED'}")
    
    # Report overall status
    if kokoro_success and bark_success:
        print("\nAll models initialized successfully!")
    else:
        print("\nSome models failed to initialize. See logs above for details.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TTS model initialization")
    parser.add_argument("--model", choices=["kokoro", "bark", "all"], default="all",
                        help="Which model to test (default: all)")
    parser.add_argument("--use-hf-config", action="store_true", 
                        help="Use HuggingFace config for Kokoro model")
    args = parser.parse_args()
    
    if args.model == "kokoro" or args.model == "all":
        test_kokoro_initialization(use_hf_config=args.use_hf_config)
    
    if args.model == "bark" or args.model == "all":
        test_bark_initialization()
