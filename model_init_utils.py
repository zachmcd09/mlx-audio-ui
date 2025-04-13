"""
Utility functions for model initialization to address issues with Kokoro and Bark models.
"""
import json
import sys
from typing import Dict, Any, Optional, Tuple

from huggingface_hub import hf_hub_download
import mlx.core as mx
import mlx.nn as nn


def load_model_config_from_hf(repo_id: str) -> Dict[str, Any]:
    """Load model configuration directly from HuggingFace."""
    print(f"Downloading config from HuggingFace: {repo_id}")
    config_file = hf_hub_download(repo_id=repo_id, filename="config.json")
    
    with open(config_file, 'r') as f:
        config_json = json.load(f)
    
    print(f"Successfully loaded config from {repo_id}")
    return config_json


def get_kokoro_model_config(repo_id: str) -> Dict[str, Any]:
    """
    Get a validated Kokoro model configuration with all required fields.
    Uses config.json from HuggingFace with fallbacks to defaults for missing fields.
    """
    from mlx_audio.tts.models.kokoro import ModelConfig
    
    try:
        # Try loading from HuggingFace first
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
            
            # Add missing fields with sensible defaults
            if "max_dur" in missing_fields:
                config_json["max_dur"] = 50
            if "dropout" in missing_fields:
                config_json["dropout"] = 0.1
            if "multispeaker" in missing_fields:
                config_json["multispeaker"] = True
            
            # Check nested fields
            if "istftnet" not in config_json or not isinstance(config_json["istftnet"], dict):
                config_json["istftnet"] = {
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
                }
            
            if "plbert" not in config_json or not isinstance(config_json["plbert"], dict):
                config_json["plbert"] = {
                    "hidden_size": 768,
                    "num_attention_heads": 12,
                    "intermediate_size": 2048,
                    "max_position_embeddings": 512,
                    "num_hidden_layers": 12,
                    "dropout": 0.1
                }
            
            # The vocab is critical - if missing, we would need to use the hardcoded one
            if "vocab" not in config_json:
                # Fallback to hardcoded vocab if needed
                config_json["vocab"] = {
                    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6, "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13, """: 14, """: 15, " ": 16, "\u0303": 17, "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23, "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41, "ᵊ": 42, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68, "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78, "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90, "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111, "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119, "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130, "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139, "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148, "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164, "↓": 169, "→": 171, "↗": 172, "↘": 173, "ᵻ": 177
                }
        
        # Handle Kokoro-specific adjustments
        # Ensure dim_in is set correctly
        if "dim_in" in config_json and config_json["dim_in"] != 64:
            print(f"Warning: Overriding dim_in from {config_json['dim_in']} to 64 (known working value)")
            config_json["dim_in"] = 64
        
        # ModelConfig validation will occur when instance is created
        return config_json
        
    except Exception as e:
        print(f"Error loading Kokoro model config from HuggingFace: {e}")
        print("Falling back to hardcoded configuration...")
        
        # Fallback to hardcoded config
        return {
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


class EncodecParameterAdapter:
    """
    Adapter class for handling parameter mismatches in the Encodec model.
    This addresses issues where the parameter names in the weights file don't match the expected names.
    """
    @staticmethod
    def adapt_parameters(state_dict):
        """
        Remove parameters that don't match the expected model structure.
        This specifically handles the Encodec parameter mismatch in Bark's codec.
        """
        # Extract parameter names
        original_keys = set(state_dict.keys())
        
        # Parameter patterns to filter out (based on error messages)
        problematic_patterns = [
            "encoder.layers.1.block", 
            "encoder.layers.4.block",
            "encoder.layers.7.block", 
            "encoder.layers.10.block",
            "decoder.layers.4.block", 
            "decoder.layers.7.block",
            "decoder.layers.10.block", 
            "decoder.layers.13.block"
        ]
        
        # Filter out problematic parameters
        filtered_dict = {k: v for k, v in state_dict.items() 
                        if not any(pattern in k for pattern in problematic_patterns)}
        
        # Log stats
        removed = original_keys - set(filtered_dict.keys())
        if removed:
            print(f"EncodecParameterAdapter: Removed {len(removed)} incompatible parameters")
        
        return filtered_dict


def initialize_kokoro_model(repo_id: str) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Initialize the Kokoro model with proper configuration.
    
    Returns:
        Tuple containing (model instance, config dictionary)
    """
    from mlx_audio.tts.models.kokoro import Model, ModelConfig
    
    # Get validated config
    model_config_dict = get_kokoro_model_config(repo_id)
    
    # Create model config and model
    model_config = ModelConfig(**model_config_dict)
    model = Model(model_config, repo_id=repo_id)
    
    return model, model_config_dict


def initialize_bark_model(repo_id: str) -> nn.Module:
    """
    Initialize the Bark model with compatible codec.
    
    Returns:
        BarkPipeline instance
    """
    from mlx_audio.tts.models.bark import BarkPipeline
    import mlx_audio.codec.models.encodec.encodec as encodec_module
    
    # Save the original load_weights method
    original_load_weights = nn.Module.load_weights
    
    # Create a patched load_weights method that applies our parameter adapter
    def patched_load_weights(self, weights_path, **kwargs):
        print(f"Loading weights with adaptation from {weights_path}")
        
        # Only apply adapter for Encodec model
        if isinstance(self, encodec_module.Encodec):
            print("Applying EncodecParameterAdapter for Encodec model")
            # Load weights
            import safetensors.numpy
            state_dict = safetensors.numpy.load_file(weights_path)
            
            # Apply adapter
            adapted_dict = EncodecParameterAdapter.adapt_parameters(state_dict)
            
            # Load adapted weights
            self.update(adapted_dict)
            return
        
        # For other models, use the original method
        return original_load_weights(self, weights_path, **kwargs)
    
    try:
        # Apply the patch
        nn.Module.load_weights = patched_load_weights
        
        # Create the pipeline
        pipeline = BarkPipeline(repo_id=repo_id)
        
        # Success!
        return pipeline
    finally:
        # Always restore the original method
        nn.Module.load_weights = original_load_weights
