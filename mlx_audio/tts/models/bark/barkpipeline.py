import os
import sys
import logging
from typing import Any, Generator, Optional, Dict, Tuple, Union
from dataclasses import dataclass
from huggingface_hub import hf_hub_download, snapshot_download
import mlx.core as mx
from transformers import BertTokenizer

from .pipeline import Pipeline
from .bark import (
    Model, 
    ModelConfig, 
    SemanticConfig, 
    CoarseAcousticsConfig,
    FineAcousticsConfig,
    CodecConfig,
    filter_dataclass_fields
)

# Setup logger
logger = logging.getLogger("bark_pipeline")
handler = logging.StreamHandler(sys.stderr)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Default model path (can be overridden)
DEFAULT_MODEL_REPO = "suno/bark-small"
DEFAULT_CODEC_REPO = "mlx-community/encodec-24khz-float32"

class BarkPipeline(Pipeline):
    """
    A specialized pipeline for Bark TTS.
    This is a wrapper around the base Pipeline class to provide compatibility 
    with the expected BarkPipeline import in app.py.
    """
    
    def __init__(self, repo_id: str = DEFAULT_MODEL_REPO, **kwargs):
        """
        Initialize a BarkPipeline.
        
        Args:
            repo_id: The Hugging Face repo ID for the model
            **kwargs: Additional arguments to pass to the model initialization
        """
        try:
            logger.info(f"Initializing BarkPipeline with repo_id: {repo_id}")
            
            # Get configuration parameters
            local_files_only = kwargs.get("local_files_only", False)
            codec_path = kwargs.get("codec_path", DEFAULT_CODEC_REPO)
            
            # Create individual configurations with default values
            semantic_config = SemanticConfig()
            coarse_acoustics_config = CoarseAcousticsConfig()
            fine_acoustics_config = FineAcousticsConfig()
            codec_config = CodecConfig()
            
            # Create the model configuration with all required arguments
            model_config = ModelConfig(
                semantic_config=semantic_config,
                coarse_acoustics_config=coarse_acoustics_config,
                fine_acoustics_config=fine_acoustics_config,
                codec_config=codec_config
            )
            
            # Set codec path in config
            model_config.codec_path = codec_path
            
            # Load the tokenizer
            logger.info("Loading BertTokenizer...")
            tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
            if tokenizer is None:
                raise ValueError("Failed to load BertTokenizer")
            
            # Initialize the model
            logger.info("Creating Bark model...")
            model = Model(model_config)
            
            # Download model weights if not already cached
            logger.info(f"Downloading or loading model weights from {repo_id}...")
            try:
                # Use snapshot_download to get the whole repo
                model_path = snapshot_download(
                    repo_id=repo_id,
                    local_files_only=local_files_only
                )
                logger.info(f"Model weights downloaded to {model_path}")
                
                # Here you would load the weights into the model
                # For now, this is a placeholder as the actual loading would depend
                # on the specific implementation details of the Bark model
                
                # In a real implementation, you would load weights like:
                # weights = mx.load(os.path.join(model_path, "weights.npz"))
                # model.update(weights)
                
            except Exception as e:
                logger.error(f"Failed to download model weights: {str(e)}")
                raise ValueError(f"Failed to download model weights: {str(e)}")
            
            # Initialize the base Pipeline with the model and tokenizer
            logger.info("Initializing base Pipeline...")
            super().__init__(model, tokenizer, model_config)
            logger.info("BarkPipeline initialization complete")
            
        except Exception as e:
            logger.error(f"Error initializing BarkPipeline: {str(e)}")
            # Re-raise the exception to propagate it up
            raise
