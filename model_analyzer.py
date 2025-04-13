#!/usr/bin/env python3
"""
Model Architecture Analyzer for TTS Models

This script provides tools to analyze model architectures, parameter requirements,
and dependencies to better understand initialization issues.
"""
import sys
import inspect
import json
import os
from typing import Dict, Any, Set, List, Tuple, Optional, Type, Callable
import importlib
from dataclasses import dataclass
import logging
import traceback

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_analysis.log')
    ]
)
logger = logging.getLogger('model_analyzer')


@dataclass
class ParameterInfo:
    """Information about a model parameter."""
    name: str
    type_hint: Optional[Type] = None
    default_value: Any = None
    is_required: bool = True
    description: str = ""
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "type": str(self.type_hint) if self.type_hint else "unknown",
            "default_value": (
                str(self.default_value) 
                if not isinstance(self.default_value, (dict, list)) 
                else self.default_value
            ),
            "is_required": self.is_required,
            "description": self.description
        }


@dataclass
class ModelAnalysis:
    """Analysis results for a model."""
    model_class: Type
    parameters: List[ParameterInfo]
    children: Dict[str, Any]
    module_parameters: Dict[str, List[Tuple[str, Tuple]]]
    initialization_trace: List[str]
    
    def to_dict(self):
        """Convert to dictionary representation."""
        return {
            "model_class": self.model_class.__name__,
            "module": self.model_class.__module__,
            "parameters": [p.to_dict() for p in self.parameters],
            "children": {k: (v.__class__.__name__ if v else None) for k, v in self.children.items()},
            "parameter_count": {k: len(v) for k, v in self.module_parameters.items()},
            "initialization_trace": self.initialization_trace
        }


class ModelArchitectureAnalyzer:
    """
    Analyzer for TTS model architectures to understand initialization requirements.
    """
    
    def __init__(self, output_dir: str = "model_analysis_results"):
        """
        Initialize the analyzer.
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def analyze_model_class(self, 
                           model_class: Type, 
                           config_class: Optional[Type] = None,
                           model_type: str = "unknown") -> ModelAnalysis:
        """
        Analyze a model class to extract parameter requirements.
        
        Args:
            model_class: The model class to analyze
            config_class: Optional config class to analyze
            model_type: Type identifier for this model
            
        Returns:
            ModelAnalysis object with analysis results
        """
        logger.info(f"Analyzing model class: {model_class.__name__}")
        
        # Extract parameter info from __init__ method
        parameters = self._extract_init_parameters(model_class)
        
        # If config class is provided, analyze it as well
        if config_class:
            config_params = self._extract_init_parameters(config_class)
            logger.info(f"Config class {config_class.__name__} requires {len(config_params)} parameters")
            
            # Add a parameter for the config itself
            parameters.append(ParameterInfo(
                name="config",
                type_hint=config_class,
                is_required=True,
                description=f"Instance of {config_class.__name__}"
            ))
        
        # Try to create a minimal instance for further analysis
        initialized_model = None
        initialization_trace = []
        children = {}
        module_parameters = {}
        
        try:
            # Create a tracer to capture initialization path
            def init_tracer(frame, event, arg):
                if event == 'call':
                    co = frame.f_code
                    func_name = co.co_name
                    if func_name == '__init__':
                        func_filename = co.co_filename
                        class_name = frame.f_locals.get('self', None).__class__.__name__
                        initialization_trace.append(f"{class_name}.__init__ in {os.path.basename(func_filename)}")
                return init_tracer
            
            # Try to create minimal instances for inspection
            sys.settrace(init_tracer)
            
            # If we have a config class, try to create a minimal config
            minimal_config = None
            if config_class:
                try:
                    # Get default values for all parameters
                    config_defaults = {
                        # Common config parameters with typical defaults
                        'dim_in': 64,
                        'dropout': 0.1,
                        'hidden_dim': 512,
                        'n_layer': 3,
                        'n_mels': 80,
                        'n_token': 178,
                        'style_dim': 128,
                        'max_dur': 50,
                        'multispeaker': True,
                        'text_encoder_kernel_size': 5,
                        'model_type': model_type,
                        'audio_channels': 1,
                        'num_filters': 32,
                    }
                    
                    # Create config instance with minimal defaults
                    minimal_config = config_class(**{
                        p.name: config_defaults.get(p.name, p.default_value)
                        for p in config_params
                        if p.name in config_defaults or p.default_value is not None
                    })
                    logger.info(f"Created minimal config instance for {config_class.__name__}")
                except Exception as e:
                    logger.warning(f"Could not create minimal config: {e}")
                    minimal_config = None
            
            # Try to create the model with minimal config
            try:
                if minimal_config:
                    initialized_model = model_class(minimal_config)
                    logger.info(f"Created minimal model instance with config")
                else:
                    # Try with minimal parameters
                    minimal_params = {
                        'repo_id': 'dummy/repo',
                        'lang_code': 'a',
                    }
                    initialized_model = model_class(**minimal_params)
                    logger.info(f"Created minimal model instance")
            except Exception as e:
                logger.warning(f"Could not create minimal model instance: {e}")
                logger.debug(traceback.format_exc())
                initialized_model = None
                
            # Always restore system trace function
            sys.settrace(None)
            
            # If model was initialized, analyze its structure
            if initialized_model:
                # Get child modules
                for name, child in initialized_model.named_children():
                    children[name] = child
                
                # Get parameters
                for name, module in initialized_model.named_modules():
                    params = [(param_name, param.shape) 
                             for param_name, param in module.parameters().items()]
                    if params:
                        module_parameters[f"{name} ({module.__class__.__name__})"] = params
        
        except Exception as e:
            logger.error(f"Error during model initialization analysis: {e}")
            logger.debug(traceback.format_exc())
            sys.settrace(None)  # Ensure trace function is reset
        
        # Create and return analysis object
        analysis = ModelAnalysis(
            model_class=model_class,
            parameters=parameters,
            children=children,
            module_parameters=module_parameters,
            initialization_trace=initialization_trace
        )
        
        # Save analysis results
        self._save_analysis(analysis, model_type)
        
        return analysis
    
    def _extract_init_parameters(self, cls: Type) -> List[ParameterInfo]:
        """Extract parameter information from a class's __init__ method."""
        parameters = []
        
        # Get the __init__ method
        init_method = cls.__init__
        
        # Get signature
        try:
            signature = inspect.signature(init_method)
            
            # Get docstring to extract descriptions
            docstring = init_method.__doc__ or ""
            param_descriptions = self._parse_docstring_params(docstring)
            
            # Analyze each parameter
            for name, param in signature.parameters.items():
                # Skip 'self'
                if name == 'self':
                    continue
                
                # Determine if parameter is required
                is_required = param.default is inspect.Parameter.empty
                
                # Create parameter info
                param_info = ParameterInfo(
                    name=name,
                    type_hint=param.annotation if param.annotation is not inspect.Parameter.empty else None,
                    default_value=None if is_required else param.default,
                    is_required=is_required,
                    description=param_descriptions.get(name, "")
                )
                
                parameters.append(param_info)
        
        except Exception as e:
            logger.error(f"Error extracting parameters from {cls.__name__}: {e}")
        
        return parameters
    
    def _parse_docstring_params(self, docstring: str) -> Dict[str, str]:
        """Extract parameter descriptions from docstring."""
        param_descriptions = {}
        
        # Simple parser for docstring parameters
        param_section = False
        current_param = None
        current_desc = []
        
        for line in docstring.split('\n'):
            line = line.strip()
            
            # Check for parameter section
            if line.lower().startswith('parameters:') or line.lower().startswith('args:'):
                param_section = True
                continue
            
            # Look for a new parameter definition
            if param_section and line.startswith((':', '-', '*')):
                # Save the previous parameter if any
                if current_param and current_desc:
                    param_descriptions[current_param] = ' '.join(current_desc)
                
                # Parse the new parameter
                parts = line.lstrip(':- *').split(':', 1)
                if len(parts) > 1:
                    current_param = parts[0].strip()
                    current_desc = [parts[1].strip()]
                else:
                    # Try alternative format (param_name - description)
                    parts = line.lstrip(':- *').split(' - ', 1)
                    if len(parts) > 1:
                        current_param = parts[0].strip()
                        current_desc = [parts[1].strip()]
                    else:
                        current_param = parts[0].strip()
                        current_desc = []
            
            # Add to current description
            elif param_section and current_param and line and not line.startswith(('Returns:', 'Raises:')):
                current_desc.append(line)
            
            # End of parameter section
            elif param_section and (not line or line.startswith(('Returns:', 'Raises:'))):
                # Save the previous parameter if any
                if current_param and current_desc:
                    param_descriptions[current_param] = ' '.join(current_desc)
                
                # End parameter section if we hit a new section
                if line.startswith(('Returns:', 'Raises:')):
                    param_section = False
        
        # Save the last parameter if any
        if current_param and current_desc:
            param_descriptions[current_param] = ' '.join(current_desc)
        
        return param_descriptions
    
    def _save_analysis(self, analysis: ModelAnalysis, model_type: str):
        """Save analysis results to file."""
        filename = os.path.join(self.output_dir, f"{model_type}_{analysis.model_class.__name__}.json")
        
        with open(filename, 'w') as f:
            json.dump(analysis.to_dict(), f, indent=2)
        
        logger.info(f"Saved analysis to {filename}")


def analyze_kokoro_model():
    """Analyze the Kokoro TTS model architecture."""
    try:
        from mlx_audio.tts.models.kokoro import Model, ModelConfig, KokoroPipeline
        
        analyzer = ModelArchitectureAnalyzer()
        
        # Analyze the model class
        analyzer.analyze_model_class(Model, ModelConfig, model_type="kokoro")
        
        # Analyze the pipeline class
        analyzer.analyze_model_class(KokoroPipeline, model_type="kokoro_pipeline")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing Kokoro model: {e}")
        logger.debug(traceback.format_exc())
        return False


def analyze_bark_model():
    """Analyze the Bark TTS model architecture."""
    try:
        from mlx_audio.tts.models.bark import BarkPipeline
        from mlx_audio.tts.models.bark.bark import Model as BarkModel, ModelConfig as BarkConfig
        
        analyzer = ModelArchitectureAnalyzer()
        
        # Analyze the model class
        analyzer.analyze_model_class(BarkModel, BarkConfig, model_type="bark")
        
        # Analyze the pipeline class
        analyzer.analyze_model_class(BarkPipeline, model_type="bark_pipeline")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing Bark model: {e}")
        logger.debug(traceback.format_exc())
        return False


def analyze_encodec_model():
    """Analyze the Encodec model used by Bark."""
    try:
        from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig
        
        analyzer = ModelArchitectureAnalyzer()
        
        # Analyze the model class
        analyzer.analyze_model_class(Encodec, EncodecConfig, model_type="encodec")
        
        return True
    except Exception as e:
        logger.error(f"Error analyzing Encodec model: {e}")
        logger.debug(traceback.format_exc())
        return False


def main():
    """Run model architecture analysis."""
    print("Model Architecture Analysis")
    print("==========================")
    
    success_count = 0
    
    print("\nAnalyzing Kokoro model architecture...")
    if analyze_kokoro_model():
        print("✅ Kokoro model analysis complete")
        success_count += 1
    else:
        print("❌ Kokoro model analysis failed")
    
    print("\nAnalyzing Bark model architecture...")
    if analyze_bark_model():
        print("✅ Bark model analysis complete")
        success_count += 1
    else:
        print("❌ Bark model analysis failed")
    
    print("\nAnalyzing Encodec model architecture...")
    if analyze_encodec_model():
        print("✅ Encodec model analysis complete")
        success_count += 1
    else:
        print("❌ Encodec model analysis failed")
    
    print("\nAnalysis Summary:")
    print(f"- Completed {success_count}/3 model analyses")
    print(f"- Results saved to ./model_analysis_results/")
    print(f"- Log saved to ./model_analysis.log")


if __name__ == "__main__":
    main()
