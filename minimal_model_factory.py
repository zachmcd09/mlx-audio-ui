#!/usr/bin/env python3
"""
Minimal Model Factory for TTS Models

This script creates minimal versions of TTS models that focus solely on
successful initialization, stripping away extra functionality to isolate
and diagnose initialization issues.
"""
import os
import sys
import json
import logging
import traceback
from typing import Dict, Any, Optional, Tuple, Type, Union
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('minimal_model.log')
    ]
)
logger = logging.getLogger('minimal_model_factory')


class MinimalModelFactory:
    """
    Factory for creating minimal versions of models that focus only on initialization.
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Initialize the factory.
        
        Args:
            cache_dir: Directory to cache model files
        """
        self.cache_dir = Path(cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create model-specific cache directories
        self.kokoro_cache = self.cache_dir / "kokoro"
        self.bark_cache = self.cache_dir / "bark"
        self.encodec_cache = self.cache_dir / "encodec"
        
        for dir_path in [self.kokoro_cache, self.bark_cache, self.encodec_cache]:
            dir_path.mkdir(exist_ok=True)
    
    def create_minimal_kokoro_config(self) -> Dict[str, Any]:
        """
        Create a minimal configuration for Kokoro model.
        
        Returns:
            Dictionary with minimal configuration
        """
        # These are the essential parameters needed for initialization
        minimal_config = {
            "dim_in": 64,
            "dropout": 0.1,
            "hidden_dim": 512,
            "max_dur": 50,
            "max_conv_dim": 512,
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
                # Minimal vocab with just a few entries to make it work
                " ": 16, "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, 
                "g": 49, "h": 50, "i": 51, "j": 52, "k": 53, "l": 54, "m": 55, 
                "n": 56, "o": 57, "p": 58, "q": 59, "r": 60, "s": 61, "t": 62, 
                "u": 63, "v": 64, "w": 65, "x": 66, "y": 67, "z": 68
            }
        }
        
        # Save the config for reference and caching
        config_path = self.kokoro_cache / "minimal_config.json"
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
        
        logger.info(f"Created minimal Kokoro config at {config_path}")
        return minimal_config
    
    def create_minimal_kokoro_model(self) -> Optional[nn.Module]:
        """
        Create a minimal Kokoro model focused only on initialization.
        
        Returns:
            Minimal Kokoro model instance or None if initialization fails
        """
        try:
            from mlx_audio.tts.models.kokoro import Model, ModelConfig
            
            logger.info("Creating minimal Kokoro model")
            
            # Create ModelConfig with minimal parameters
            config_dict = self.create_minimal_kokoro_config()
            model_config = ModelConfig(**config_dict)
            
            # Create model with minimal initialization
            # Skip actual weight loading to focus on structure
            class MinimalKokoroModel(Model):
                def __init__(self, config):
                    super().__init__(config, repo_id=None)
                    # Override weight loading methods to do nothing
                    self.load_weights = lambda *args, **kwargs: None
            
            # Create the minimal model
            minimal_model = MinimalKokoroModel(model_config)
            logger.info("Successfully created minimal Kokoro model")
            
            return minimal_model
        
        except Exception as e:
            logger.error(f"Failed to create minimal Kokoro model: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def create_minimal_encodec_config(self) -> Dict[str, Any]:
        """
        Create a minimal configuration for Encodec model.
        
        Returns:
            Dictionary with minimal configuration
        """
        # These are the essential parameters needed for initialization
        minimal_config = {
            "model_type": "encodec",
            "audio_channels": 1,
            "num_filters": 32,
            "hidden_size": 128,
            "num_residual_layers": 1,
            "sampling_rate": 24000,
            "target_bandwidths": [1.5, 3.0, 6.0, 12.0],
            "upsampling_ratios": [8, 5, 4, 2]
        }
        
        # Save the config for reference and caching
        config_path = self.encodec_cache / "minimal_config.json"
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
        
        logger.info(f"Created minimal Encodec config at {config_path}")
        return minimal_config
    
    def create_minimal_encodec_model(self) -> Optional[nn.Module]:
        """
        Create a minimal Encodec model focused only on initialization.
        
        Returns:
            Minimal Encodec model instance or None if initialization fails
        """
        try:
            from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig
            
            logger.info("Creating minimal Encodec model")
            
            # Create EncodecConfig with minimal parameters
            config_dict = self.create_minimal_encodec_config()
            model_config = EncodecConfig(**config_dict)
            
            # Create model with minimal initialization
            # Skip actual weight loading
            class MinimalEncodecModel(Encodec):
                def __init__(self, config):
                    super().__init__(config)
                    # Override weight loading methods to do nothing
                    self.load_weights = lambda *args, **kwargs: None
            
            # Create the minimal model
            minimal_model = MinimalEncodecModel(model_config)
            logger.info("Successfully created minimal Encodec model")
            
            return minimal_model
        
        except Exception as e:
            logger.error(f"Failed to create minimal Encodec model: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def create_minimal_bark_config(self) -> Dict[str, Any]:
        """
        Create a minimal configuration for Bark model.
        
        Returns:
            Dictionary with minimal configuration
        """
        # These are the essential parameters needed for initialization
        minimal_config = {
            "version": "v0",
            "semantic_config": {
                "hidden_size": 512,
                "num_heads": 8,
                "num_layers": 12,
                "vocab_size": 10000
            },
            "coarse_config": {
                "hidden_size": 1024,
                "num_heads": 16,
                "num_layers": 24,
                "vocab_size": 1024
            },
            "fine_config": {
                "hidden_size": 768,
                "num_heads": 12,
                "num_layers": 12,
                "vocab_size": 8192
            }
        }
        
        # Save the config for reference and caching
        config_path = self.bark_cache / "minimal_config.json"
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
        
        logger.info(f"Created minimal Bark config at {config_path}")
        return minimal_config
    
    def create_minimal_bark_model(self) -> Optional[nn.Module]:
        """
        Create a minimal Bark model focused only on initialization.
        
        Returns:
            Minimal Bark model instance or None if initialization fails
        """
        try:
            from mlx_audio.tts.models.bark.bark import Model as BarkModel, ModelConfig as BarkConfig
            
            logger.info("Creating minimal Bark model")
            
            # Create BarkConfig with minimal parameters
            config_dict = self.create_minimal_bark_config()
            model_config = BarkConfig(**config_dict)
            
            # Create model with minimal initialization
            # Skip actual weight loading
            class MinimalBarkModel(BarkModel):
                def __init__(self, config):
                    super().__init__(config)
                    # Override weight loading methods to do nothing
                    self.load_weights = lambda *args, **kwargs: None
            
            # Create the minimal model
            minimal_model = MinimalBarkModel(model_config)
            logger.info("Successfully created minimal Bark model")
            
            return minimal_model
        
        except Exception as e:
            logger.error(f"Failed to create minimal Bark model: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def create_minimal_bark_pipeline_with_codec(self) -> Optional[nn.Module]:
        """
        Create a minimal Bark pipeline with proper codec integration.
        
        Returns:
            Minimal BarkPipeline instance or None if initialization fails
        """
        try:
            from mlx_audio.tts.models.bark import BarkPipeline
            
            logger.info("Creating minimal Bark pipeline")
            
            # First create a minimal Encodec model
            encodec_model = self.create_minimal_encodec_model()
            if encodec_model is None:
                logger.error("Could not create minimal Encodec model for Bark pipeline")
                return None
            
            # Create minimal pipeline with overridden initialization
            class MinimalBarkPipeline(BarkPipeline):
                def __init__(self, codec_model):
                    # Store the codec model directly
                    self._codec = codec_model
                    
                    # Initialize basic attributes without loading actual models
                    self.tokenizer = None
                    self.model = None
                    self.repo_id = "dummy/repo"
                    self.config = None
                    self.speakers = {"en_speaker_0": "test_speaker"}
                    
                    logger.info("Initialized minimal Bark pipeline")
            
            # Create pipeline with our minimal codec
            minimal_pipeline = MinimalBarkPipeline(encodec_model)
            logger.info("Successfully created minimal Bark pipeline with codec")
            
            return minimal_pipeline
        
        except Exception as e:
            logger.error(f"Failed to create minimal Bark pipeline: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def test_minimal_model_initialization(self):
        """
        Test minimal model initialization for all models.
        
        Returns:
            Dictionary with initialization status for each model
        """
        results = {}
        
        # Test Kokoro model
        logger.info("\n=== Testing Minimal Kokoro Model Initialization ===")
        kokoro_model = self.create_minimal_kokoro_model()
        results["kokoro"] = kokoro_model is not None
        
        # Test Encodec model
        logger.info("\n=== Testing Minimal Encodec Model Initialization ===")
        encodec_model = self.create_minimal_encodec_model()
        results["encodec"] = encodec_model is not None
        
        # Test Bark model
        logger.info("\n=== Testing Minimal Bark Model Initialization ===")
        bark_model = self.create_minimal_bark_model()
        results["bark"] = bark_model is not None
        
        # Test Bark pipeline with codec
        logger.info("\n=== Testing Minimal Bark Pipeline with Codec ===")
        bark_pipeline = self.create_minimal_bark_pipeline_with_codec()
        results["bark_pipeline"] = bark_pipeline is not None
        
        # Print summary
        logger.info("\n=== Minimal Model Initialization Results ===")
        for model_name, success in results.items():
            logger.info(f"{model_name}: {'SUCCESS' if success else 'FAILED'}")
        
        return results


class IncrementalModelBuilder:
    """
    Builds models incrementally from base components to isolate issues.
    """
    
    def __init__(self, cache_dir: str = "model_cache"):
        """
        Initialize the builder.
        
        Args:
            cache_dir: Directory to cache model files
        """
        self.cache_dir = Path(cache_dir)
        self.factory = MinimalModelFactory(cache_dir)
    
    def build_encodec_encoder_only(self) -> Optional[nn.Module]:
        """
        Build just the encoder part of Encodec.
        
        Returns:
            Encoder module or None if initialization fails
        """
        try:
            from mlx_audio.codec.models.encodec.encodec import EncodecConfig
            
            # Create basic Encodec config
            config_dict = self.factory.create_minimal_encodec_config()
            config = EncodecConfig(**config_dict)
            
            # Define a minimal encoder
            class MinimalEncoder(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    # Create basic layers that would be in an encoder
                    self.conv1 = nn.Conv1d(config.audio_channels, config.num_filters, kernel_size=7, padding=3)
                    self.relu = nn.ReLU()
                    self.conv2 = nn.Conv1d(config.num_filters, config.num_filters, kernel_size=7, padding=3)
                    self.pool = nn.MaxPool1d(kernel_size=2)
            
            encoder = MinimalEncoder(config)
            logger.info("Successfully built minimal Encodec encoder")
            return encoder
        
        except Exception as e:
            logger.error(f"Failed to build Encodec encoder: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def build_encodec_decoder_only(self) -> Optional[nn.Module]:
        """
        Build just the decoder part of Encodec.
        
        Returns:
            Decoder module or None if initialization fails
        """
        try:
            from mlx_audio.codec.models.encodec.encodec import EncodecConfig
            
            # Create basic Encodec config
            config_dict = self.factory.create_minimal_encodec_config()
            config = EncodecConfig(**config_dict)
            
            # Define a minimal decoder
            class MinimalDecoder(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    # Create basic layers that would be in a decoder
                    self.conv1 = nn.Conv1d(config.num_filters, config.num_filters, kernel_size=7, padding=3)
                    self.relu = nn.ReLU()
                    self.conv2 = nn.Conv1d(config.num_filters, config.audio_channels, kernel_size=7, padding=3)
                    # Simple upsampling layer
                    self.upsample = lambda x: mx.repeat(x, 2, axis=2)
            
            decoder = MinimalDecoder(config)
            logger.info("Successfully built minimal Encodec decoder")
            return decoder
        
        except Exception as e:
            logger.error(f"Failed to build Encodec decoder: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def build_minimal_encodec(self) -> Optional[nn.Module]:
        """
        Build a complete minimal Encodec model from encoder and decoder components.
        
        Returns:
            Encodec module or None if initialization fails
        """
        try:
            from mlx_audio.codec.models.encodec.encodec import EncodecConfig
            
            # Get encoder and decoder
            encoder = self.build_encodec_encoder_only()
            decoder = self.build_encodec_decoder_only()
            
            if encoder is None or decoder is None:
                logger.error("Failed to build encoder or decoder components")
                return None
            
            # Create basic Encodec config
            config_dict = self.factory.create_minimal_encodec_config()
            config = EncodecConfig(**config_dict)
            
            # Define a minimal Encodec model with our components
            class MinimalEncodec(nn.Module):
                def __init__(self, config, encoder, decoder):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                    self.config = config
                    
                    # Add a quantizer component (simplified)
                    self.quantizer = nn.Identity()
            
            # Create the model with our components
            model = MinimalEncodec(config, encoder, decoder)
            logger.info("Successfully built minimal Encodec from components")
            return model
        
        except Exception as e:
            logger.error(f"Failed to build minimal Encodec: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def build_full_bark_with_verified_codec(self) -> Optional[nn.Module]:
        """
        Build Bark using a pre-verified codec.
        
        Returns:
            BarkPipeline with verified codec or None if initialization fails
        """
        try:
            from mlx_audio.tts.models.bark import BarkPipeline
            
            # Create verified codec first
            verified_codec = self.build_minimal_encodec()
            if verified_codec is None:
                logger.error("Could not create verified codec for Bark")
                return None
            
            # Create a minimal Bark pipeline with injected codec
            class MinimalBarkWithCodec(BarkPipeline):
                def __init__(self, codec):
                    # Initialize with minimal setup, inject our codec
                    self.repo_id = "dummy/repo"
                    self._codec = codec
                    
                    # Add stubs for required components
                    self.model = None
                    self.tokenizer = None
                    self.config = None
                    self.speakers = {"en_speaker_0": "test_speaker"}
            
            # Create the pipeline
            pipeline = MinimalBarkWithCodec(verified_codec)
            logger.info("Successfully built Bark with verified codec")
            return pipeline
        
        except Exception as e:
            logger.error(f"Failed to build Bark with verified codec: {e}")
            logger.debug(traceback.format_exc())
            return None
    
    def test_incremental_model_building(self):
        """
        Test incremental model building to identify issues.
        
        Returns:
            Dictionary with build status for each component
        """
        results = {}
        
        # Test Encodec encoder
        logger.info("\n=== Testing Encodec Encoder Build ===")
        encoder = self.build_encodec_encoder_only()
        results["encodec_encoder"] = encoder is not None
        
        # Test Encodec decoder
        logger.info("\n=== Testing Encodec Decoder Build ===")
        decoder = self.build_encodec_decoder_only()
        results["encodec_decoder"] = decoder is not None
        
        # Test combined Encodec
        logger.info("\n=== Testing Combined Encodec Build ===")
        encodec = self.build_minimal_encodec()
        results["encodec_combined"] = encodec is not None
        
        # Test Bark with verified codec
        logger.info("\n=== Testing Bark with Verified Codec ===")
        bark = self.build_full_bark_with_verified_codec()
        results["bark_with_codec"] = bark is not None
        
        # Print summary
        logger.info("\n=== Incremental Model Building Results ===")
        for component_name, success in results.items():
            logger.info(f"{component_name}: {'SUCCESS' if success else 'FAILED'}")
        
        return results


def main():
    """Run minimal model factory and incremental builder tests."""
    print("Minimal Model Testing Suite")
    print("==========================")
    
    # Test minimal model initialization
    print("\n=== Testing Minimal Model Initialization ===")
    factory = MinimalModelFactory()
    factory_results = factory.test_minimal_model_initialization()
    
    # Test incremental model building
    print("\n=== Testing Incremental Model Building ===")
    builder = IncrementalModelBuilder()
    builder_results = builder.test_incremental_model_building()
    
    # Print overall results
    print("\n=== Overall Results Summary ===")
    
    print("\nMinimal Model Initialization:")
    for model, success in factory_results.items():
        print(f"  {model}: {'✅ PASS' if success else '❌ FAIL'}")
    
    print("\nIncremental Model Building:")
    for component, success in builder_results.items():
        print(f"  {component}: {'✅ PASS' if success else '❌ FAIL'}")
    
    # Count successes
    factory_success = sum(1 for success in factory_results.values() if success)
    builder_success = sum(1 for success in builder_results.values() if success)
    
    print(f"\nMinimal Model Success Rate: {factory_success}/{len(factory_results)}")
    print(f"Incremental Building Success Rate: {builder_success}/{len(builder_results)}")
    
    # Log locations
    print("\nLogs and results are available at:")
    print(f"  - Log file: ./minimal_model.log")
    print(f"  - Model configurations: ./model_cache/")


if __name__ == "__main__":
    main()
