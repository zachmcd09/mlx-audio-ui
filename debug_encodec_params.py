"""
Debug utility for examining Encodec parameter structure in the Bark model.

This script helps identify parameter name mismatches between the weights file and the model definition.
"""
import sys
import json
import os
from typing import Dict, Any, Set, List, Tuple
import argparse

from huggingface_hub import hf_hub_download
import mlx.core as mx
import numpy as np

from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig
from mlx_audio.tts.models.bark import BarkPipeline


def load_config_from_hf(repo_id: str, filename: str = "config.json") -> Dict[str, Any]:
    """Load configuration from HuggingFace."""
    print(f"Downloading {filename} from HuggingFace: {repo_id}")
    config_file = hf_hub_download(repo_id=repo_id, filename=filename)
    
    with open(config_file, 'r') as f:
        config_json = json.load(f)
    
    print(f"Successfully loaded {filename} from {repo_id}")
    return config_json


def analyze_encodec_parameters(model: Encodec) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Analyze the parameter structure of the Encodec model.
    
    Returns:
        A tuple containing:
        - A set of parameter names expected by the model
        - A dictionary of parameter shapes by name
    """
    param_names = set()
    param_shapes = {}
    
    def collect_params(module, prefix=""):
        for name, param in module.parameters().items():
            full_name = f"{prefix}.{name}" if prefix else name
            param_names.add(full_name)
            param_shapes[full_name] = param.shape
            
        for name, child in module.named_children():
            new_prefix = f"{prefix}.{name}" if prefix else name
            collect_params(child, new_prefix)
    
    collect_params(model)
    return param_names, param_shapes


def load_weights_and_analyze(weights_path: str) -> Tuple[Set[str], Dict[str, Any]]:
    """
    Load model weights and analyze the parameter structure.
    
    Returns:
        A tuple containing:
        - A set of parameter names in the weights file
        - A dictionary of parameter shapes by name
    """
    try:
        # Try loading as safetensors
        import safetensors.numpy
        weights = safetensors.numpy.load_file(weights_path)
        print(f"Loaded weights using safetensors from {weights_path}")
    except:
        # Fall back to direct numpy loading
        try:
            weights = np.load(weights_path, allow_pickle=True)
            print(f"Loaded weights using numpy from {weights_path}")
        except:
            # Try mlx loading
            weights = mx.load(weights_path)
            print(f"Loaded weights using mlx from {weights_path}")
    
    weight_names = set()
    weight_shapes = {}
    
    # Extract names and shapes
    for name, param in weights.items():
        weight_names.add(name)
        weight_shapes[name] = param.shape
    
    return weight_names, weight_shapes


def find_potential_mappings(param_names: Set[str], weight_names: Set[str]) -> Dict[str, List[str]]:
    """
    Find potential mappings between parameter names and weight names.
    
    This uses heuristics to suggest possible mappings between differently named parameters.
    
    Returns:
        A dictionary mapping expected parameter names to possible weight names
    """
    mappings = {}
    
    # Names that are identical except for specific patterns
    transformations = [
        (".gamma", ".weight"),  # BatchNorm conversion
        (".beta", ".bias"),     # BatchNorm conversion
        ("weight_v", "weight"), # Attention naming
        ("weight_ih_l0_reverse", "Wx_backward"), # LSTM naming
        ("weight_hh_l0_reverse", "Wh_backward"), # LSTM naming
        ("bias_ih_l0_reverse", "bias_ih_backward"), # LSTM naming
        ("bias_hh_l0_reverse", "bias_hh_backward"), # LSTM naming
        ("weight_ih_l0", "Wx_forward"), # LSTM naming
        ("weight_hh_l0", "Wh_forward"), # LSTM naming
        ("bias_ih_l0", "bias_ih_forward"), # LSTM naming
        ("bias_hh_l0", "bias_hh_forward"), # LSTM naming
    ]
    
    # For each expected parameter name
    for param_name in param_names:
        candidates = []
        
        # Check if it directly exists in weights
        if param_name in weight_names:
            candidates.append(param_name)
            
        # Try pattern transformations
        for old_pattern, new_pattern in transformations:
            if old_pattern in param_name:
                transformed_name = param_name.replace(old_pattern, new_pattern)
                if transformed_name in weight_names:
                    candidates.append(transformed_name)
            # Try the reverse direction
            if new_pattern in param_name:
                transformed_name = param_name.replace(new_pattern, old_pattern)
                if transformed_name in weight_names:
                    candidates.append(transformed_name)
        
        # Try suffix-only matching (often helpful with frameworks differences)
        param_suffix = param_name.split('.')[-1]
        for weight_name in weight_names:
            weight_suffix = weight_name.split('.')[-1]
            if param_suffix == weight_suffix and weight_name not in candidates:
                candidates.append(weight_name)
        
        if candidates:
            mappings[param_name] = candidates
    
    return mappings


def debug_encodec_parameters(model_only: bool = False):
    """Debug the parameter structure of the Encodec model used in Bark."""
    print("\n===== Encodec Parameter Structure Debug =====\n")
    
    # Step 1: Create the model instance with proper config
    print("1. Creating Encodec model instance...")
    encodec_config = EncodecConfig(
        model_type="encodec",
        audio_channels=1,
        num_filters=32,
        hidden_size=128,
        num_residual_layers=1,
        sampling_rate=24000,
        target_bandwidths=[1.5, 3.0, 6.0, 12.0],  # Add default bandwidths
        upsampling_ratios=[8, 5, 4, 2]  # Add default upsampling ratios
    )
    model = Encodec(encodec_config)
    print("   ✓ Model created successfully")
    
    # Step 2: Analyze model parameter structure
    print("\n2. Analyzing model parameter structure...")
    param_names, param_shapes = analyze_encodec_parameters(model)
    print(f"   ✓ Found {len(param_names)} parameters in model definition")
    
    if model_only:
        print("\nModel parameter names:")
        for name in sorted(param_names):
            print(f"   {name} : {param_shapes[name]}")
        return
    
    # Step 3: Get the codec path from Bark
    print("\n3. Determining codec path from Bark pipeline...")
    try:
        # First try to initialize BarkPipeline to get the codec path
        pipeline = BarkPipeline(repo_id="suno/bark-small")
        codec_path = pipeline.config.codec_path
        print(f"   ✓ Codec path from Bark: {codec_path}")
    except Exception as e:
        print(f"   ✗ Error getting codec path from Bark: {e}")
        print("   Using default codec path")
        codec_path = "encodec_24khz"
    
    # Step 4: Download and analyze the weights file
    print("\n4. Downloading and analyzing weights file...")
    try:
        weights_dir = hf_hub_download(repo_id=codec_path, filename="model.safetensors", 
                                     local_dir="./temp_downloads")
        print(f"   ✓ Downloaded weights file to: {weights_dir}")
        
        weight_names, weight_shapes = load_weights_and_analyze(weights_dir)
        print(f"   ✓ Found {len(weight_names)} parameters in weights file")
        
        # Step 5: Compare parameters
        print("\n5. Comparing parameters between model and weights...")
        
        # Find parameters in the model but missing from weights
        missing_from_weights = param_names - weight_names
        print(f"   Parameters in model but missing from weights: {len(missing_from_weights)}")
        if missing_from_weights:
            print("   Top 10 missing parameters:")
            for name in sorted(list(missing_from_weights))[:10]:
                print(f"      {name} : {param_shapes[name]}")
            
        # Find parameters in weights but missing from model
        missing_from_model = weight_names - param_names
        print(f"   Parameters in weights but missing from model: {len(missing_from_model)}")
        if missing_from_model:
            print("   Top 10 extra parameters:")
            for name in sorted(list(missing_from_model))[:10]:
                print(f"      {name} : {weight_shapes[name]}")
        
        # Step 6: Suggest mappings
        print("\n6. Suggesting potential parameter mappings...")
        mappings = find_potential_mappings(param_names, weight_names)
        
        # Count parameters that have potential mappings
        mapped_params = set(mappings.keys())
        unmapped_params = param_names - mapped_params
        
        print(f"   Parameters with potential mappings: {len(mapped_params)}")
        print(f"   Parameters without potential mappings: {len(unmapped_params)}")
        
        # Show example mappings
        if mappings:
            print("\n   Example mappings (showing up to 10):")
            for i, (param_name, candidates) in enumerate(mappings.items()):
                if i >= 10:
                    break
                print(f"      {param_name} → {', '.join(candidates)}")
        
        # Step 7: Generate code for parameter adapter
        print("\n7. Generating adapter code snippet...")
        
        adapter_code = """class EncodecParameterAdapter:
    \"\"\"
    Adapter for handling parameter mismatches in the Encodec model.
    \"\"\"
    @staticmethod
    def adapt_parameters(state_dict):
        \"\"\"
        Adapt parameters from state_dict to match model expectations.
        \"\"\"
        adapted_dict = {}
        
        # Parameter mapping based on analysis
        parameter_map = {
"""
        
        # Add mappings to code
        for param_name, candidates in sorted(mappings.items())[:20]:  # Limit to first 20
            if len(candidates) == 1 and candidates[0] != param_name:
                adapter_code += f"            '{candidates[0]}': '{param_name}',\n"
        
        adapter_code += """        }
        
        # Apply mappings and transformations
        for key, value in state_dict.items():
            if key in parameter_map:
                # Rename according to mapping
                new_key = parameter_map[key]
                adapted_dict[new_key] = value
            elif key.endswith('.gamma'):
                # Convert BatchNorm gamma to weight
                new_key = key.replace('.gamma', '.weight')
                adapted_dict[new_key] = value
            elif key.endswith('.beta'):
                # Convert BatchNorm beta to bias
                new_key = key.replace('.beta', '.bias')
                adapted_dict[new_key] = value
            # Add other transformations as needed
            else:
                # Pass through unchanged
                adapted_dict[key] = value
        
        return adapted_dict
"""
        
        print(adapter_code)
        
        print("\n===== Parameter Analysis Complete =====")
        
    except Exception as e:
        print(f"\nError during parameter analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Encodec parameters")
    parser.add_argument("--model-only", action="store_true", 
                        help="Only analyze model structure without weights")
    args = parser.parse_args()
    
    debug_encodec_parameters(model_only=args.model_only)
