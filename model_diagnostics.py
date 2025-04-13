#!/usr/bin/env python3
"""
Comprehensive Diagnostics Framework for TTS Model Initialization

This script provides detailed diagnostics, instrumentation, and monitoring
tools to identify the exact causes of model initialization failures.
"""
import sys
import os
import json
import time
import inspect
import traceback
import logging
from typing import Dict, Any, Set, List, Tuple, Optional, Type, Callable, Union
from pathlib import Path
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, field
import threading
from datetime import datetime

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('model_diagnostics.log')
    ]
)
logger = logging.getLogger('model_diagnostics')


@dataclass
class DiagnosticResult:
    """Diagnostic result information."""
    success: bool
    model_class: str
    start_time: float
    end_time: float
    duration: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    parameter_values: Dict[str, Any] = field(default_factory=dict)
    memory_usage: Dict[str, Any] = field(default_factory=dict)
    initialization_trace: List[str] = field(default_factory=list)
    validation_errors: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "model_class": self.model_class,
            "duration": self.duration,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.end_time).isoformat(),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
            "parameter_values": self.parameter_values,
            "memory_usage": self.memory_usage,
            "initialization_trace": self.initialization_trace,
            "validation_errors": self.validation_errors
        }


@dataclass
class ParameterSchema:
    """Schema definition for parameter validation."""
    name: str
    param_type: Union[Type, List[Type]]
    required: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    nested_schema: Optional[Dict[str, 'ParameterSchema']] = None
    validator: Optional[Callable[[Any], Tuple[bool, str]]] = None
    
    def validate(self, value: Any) -> Tuple[bool, List[str]]:
        """
        Validate a parameter value against this schema.
        
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check if required
        if self.required and value is None:
            errors.append(f"Parameter '{self.name}' is required but was None")
            return False, errors
        
        # Skip further validation if value is None and not required
        if value is None:
            return True, errors
        
        # Type checking
        if isinstance(self.param_type, list):
            valid_type = any(isinstance(value, t) for t in self.param_type)
            type_names = ", ".join(t.__name__ for t in self.param_type)
        else:
            valid_type = isinstance(value, self.param_type)
            type_names = self.param_type.__name__
        
        if not valid_type:
            errors.append(f"Parameter '{self.name}' should be of type {type_names}, got {type(value).__name__}")
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            if self.min_value is not None and value < self.min_value:
                errors.append(f"Parameter '{self.name}' value {value} is less than minimum {self.min_value}")
            if self.max_value is not None and value > self.max_value:
                errors.append(f"Parameter '{self.name}' value {value} is greater than maximum {self.max_value}")
        
        # Allowed values validation
        if self.allowed_values is not None and value not in self.allowed_values:
            errors.append(f"Parameter '{self.name}' value {value} not in allowed values: {self.allowed_values}")
        
        # Nested schema validation for dictionaries
        if self.nested_schema is not None and isinstance(value, dict):
            for key, schema in self.nested_schema.items():
                if key in value:
                    nested_valid, nested_errors = schema.validate(value[key])
                    if not nested_valid:
                        for error in nested_errors:
                            errors.append(f"{self.name}.{error}")
                elif schema.required:
                    errors.append(f"Required nested parameter '{self.name}.{key}' is missing")
        
        # Custom validator
        if self.validator is not None:
            valid, error = self.validator(value)
            if not valid:
                errors.append(f"Parameter '{self.name}': {error}")
        
        return len(errors) == 0, errors


class ModelSchemaRegistry:
    """Registry of parameter schemas for model validation."""
    
    def __init__(self):
        """Initialize the registry."""
        self.schemas = {}
    
    def register_schema(self, model_class: str, schema: Dict[str, ParameterSchema]):
        """Register a schema for a model class."""
        self.schemas[model_class] = schema
    
    def get_schema(self, model_class: str) -> Optional[Dict[str, ParameterSchema]]:
        """Get the schema for a model class if it exists."""
        return self.schemas.get(model_class)
    
    def validate_parameters(self, model_class: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate parameters against the registered schema.
        
        Returns:
            Tuple of (is_valid, error_dict)
        """
        schema = self.get_schema(model_class)
        if not schema:
            return True, {}
        
        all_valid = True
        errors = {}
        
        # Validate each parameter
        for name, param_schema in schema.items():
            value = params.get(name)
            valid, param_errors = param_schema.validate(value)
            if not valid:
                all_valid = False
                errors[name] = param_errors
        
        # Check for unknown parameters
        for name in params:
            if name not in schema:
                logger.warning(f"Unknown parameter '{name}' for model '{model_class}'")
        
        return all_valid, errors


# Create a global schema registry
schema_registry = ModelSchemaRegistry()


# Define Kokoro model schema
kokoro_schema = {
    "dim_in": ParameterSchema("dim_in", int, min_value=1),
    "dropout": ParameterSchema("dropout", float, min_value=0.0, max_value=1.0),
    "hidden_dim": ParameterSchema("hidden_dim", int, min_value=1),
    "max_dur": ParameterSchema("max_dur", int, min_value=1),
    "max_conv_dim": ParameterSchema("max_conv_dim", int, min_value=1, required=False),
    "multispeaker": ParameterSchema("multispeaker", bool),
    "n_layer": ParameterSchema("n_layer", int, min_value=1),
    "n_mels": ParameterSchema("n_mels", int, min_value=1),
    "n_token": ParameterSchema("n_token", int, min_value=1),
    "style_dim": ParameterSchema("style_dim", int, min_value=1),
    "text_encoder_kernel_size": ParameterSchema("text_encoder_kernel_size", int, min_value=1),
    "istftnet": ParameterSchema(
        "istftnet", 
        dict,
        nested_schema={
            "upsample_kernel_sizes": ParameterSchema("upsample_kernel_sizes", list),
            "upsample_rates": ParameterSchema("upsample_rates", list),
            "gen_istft_hop_size": ParameterSchema("gen_istft_hop_size", int, min_value=1),
            "gen_istft_n_fft": ParameterSchema("gen_istft_n_fft", int, min_value=1),
            "resblock_dilation_sizes": ParameterSchema("resblock_dilation_sizes", list),
            "resblock_kernel_sizes": ParameterSchema("resblock_kernel_sizes", list),
            "upsample_initial_channel": ParameterSchema("upsample_initial_channel", int, min_value=1)
        }
    ),
    "plbert": ParameterSchema(
        "plbert", 
        dict,
        nested_schema={
            "hidden_size": ParameterSchema("hidden_size", int, min_value=1),
            "num_attention_heads": ParameterSchema("num_attention_heads", int, min_value=1),
            "intermediate_size": ParameterSchema("intermediate_size", int, min_value=1),
            "max_position_embeddings": ParameterSchema("max_position_embeddings", int, min_value=1),
            "num_hidden_layers": ParameterSchema("num_hidden_layers", int, min_value=1),
            "dropout": ParameterSchema("dropout", float, min_value=0.0, max_value=1.0)
        }
    ),
    "vocab": ParameterSchema("vocab", dict)
}
schema_registry.register_schema("mlx_audio.tts.models.kokoro.Model", kokoro_schema)
schema_registry.register_schema("mlx_audio.tts.models.kokoro.ModelConfig", kokoro_schema)

# Define Encodec model schema
encodec_schema = {
    "model_type": ParameterSchema("model_type", str, allowed_values=["encodec"]),
    "audio_channels": ParameterSchema("audio_channels", int, min_value=1),
    "num_filters": ParameterSchema("num_filters", int, min_value=1),
    "hidden_size": ParameterSchema("hidden_size", int, min_value=1, required=False),
    "num_residual_layers": ParameterSchema("num_residual_layers", int, min_value=1),
    "sampling_rate": ParameterSchema("sampling_rate", int, min_value=1),
    "target_bandwidths": ParameterSchema("target_bandwidths", list),
    "upsampling_ratios": ParameterSchema("upsampling_ratios", list)
}
schema_registry.register_schema("mlx_audio.codec.models.encodec.encodec.Encodec", encodec_schema)
schema_registry.register_schema("mlx_audio.codec.models.encodec.encodec.EncodecConfig", encodec_schema)


class ModelInitializationDiagnostics:
    """
    Comprehensive diagnostics for model initialization.
    """
    
    def __init__(self, output_dir: str = "diagnostics_output"):
        """
        Initialize the diagnostics framework.
        
        Args:
            output_dir: Directory to save diagnostic results
        """
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self.results = []
        self.memory_tracker_interval = 0.5  # seconds
        self.schema_registry = schema_registry
    
    def instrument_model_class(self, model_class: Type) -> Type:
        """
        Instrument a model class to add diagnostic capabilities.
        
        Args:
            model_class: The model class to instrument
            
        Returns:
            The instrumented class
        """
        original_init = model_class.__init__
        diagnostics = self
        
        @wraps(original_init)
        def instrumented_init(self, *args, **kwargs):
            # Start tracking initialization
            diagnostics_result = diagnostics.start_diagnostic_session(
                model_class.__module__ + "." + model_class.__name__
            )
            
            # Keep track of parameter values
            params = {}
            sig = inspect.signature(original_init)
            arg_names = list(sig.parameters.keys())[1:]  # Skip 'self'
            
            # Add positional args to params
            for i, arg in enumerate(args):
                if i < len(arg_names):
                    params[arg_names[i]] = arg
            
            # Add keyword args
            params.update(kwargs)
            
            # Validate parameters if schema exists
            diagnostics.validate_parameters(model_class.__module__ + "." + model_class.__name__, params)
            
            # Store parameter values
            diagnostics_result.parameter_values = {
                k: v if not isinstance(v, (dict, list)) else str(v)
                for k, v in params.items()
            }
            
            # Start memory tracking thread
            stop_event = threading.Event()
            memory_thread = threading.Thread(
                target=diagnostics.track_memory_usage,
                args=(diagnostics_result, stop_event)
            )
            memory_thread.daemon = True
            memory_thread.start()
            
            try:
                # Run original initialization
                original_init(self, *args, **kwargs)
                # Set success state
                diagnostics_result.success = True
            except Exception as e:
                # Capture error information
                diagnostics_result.success = False
                diagnostics_result.error_type = type(e).__name__
                diagnostics_result.error_message = str(e)
                diagnostics_result.traceback = traceback.format_exc()
                
                # Re-raise the exception
                raise
            finally:
                # Stop memory tracking
                stop_event.set()
                # Complete the diagnostic session
                diagnostics.complete_diagnostic_session(diagnostics_result)
        
        # Use the instrumented init method
        model_class.__init__ = instrumented_init
        
        return model_class
    
    def validate_parameters(self, model_class: str, params: Dict[str, Any]) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate parameters against a schema.
        
        Args:
            model_class: Fully qualified model class name
            params: Parameters to validate
            
        Returns:
            Tuple of (is_valid, error_dict)
        """
        is_valid, errors = self.schema_registry.validate_parameters(model_class, params)
        return is_valid, errors
    
    def track_memory_usage(self, result: DiagnosticResult, stop_event: threading.Event):
        """
        Track memory usage during model initialization.
        
        Args:
            result: Diagnostic result to update
            stop_event: Event to signal when to stop tracking
        """
        try:
            import psutil
            process = psutil.Process()
            
            max_memory = 0
            while not stop_event.is_set():
                # Get current memory usage
                memory_info = process.memory_info()
                max_memory = max(max_memory, memory_info.rss)
                
                # Update result
                result.memory_usage = {
                    "current_mb": memory_info.rss / (1024 * 1024),
                    "max_mb": max_memory / (1024 * 1024)
                }
                
                # Sleep for a bit
                stop_event.wait(self.memory_tracker_interval)
        except Exception as e:
            logger.warning(f"Error tracking memory usage: {e}")
            result.memory_usage = {"error": str(e)}
    
    def start_diagnostic_session(self, model_class: str) -> DiagnosticResult:
        """
        Start a diagnostic session for model initialization.
        
        Args:
            model_class: Fully qualified model class name
            
        Returns:
            DiagnosticResult instance for tracking
        """
        start_time = time.time()
        
        # Create result object
        result = DiagnosticResult(
            success=False,
            model_class=model_class,
            start_time=start_time,
            end_time=0,
            duration=0
        )
        
        # Setup trace function to track initialization path
        def init_tracer(frame, event, arg):
            if event == 'call':
                func_name = frame.f_code.co_name
                if func_name == '__init__':
                    try:
                        class_name = frame.f_locals.get('self').__class__.__name__
                        module_name = frame.f_locals.get('self').__class__.__module__
                        filename = os.path.basename(frame.f_code.co_filename)
                        result.initialization_trace.append(
                            f"{module_name}.{class_name}.__init__ in {filename}"
                        )
                    except (AttributeError, KeyError):
                        pass
            return init_tracer
        
        # Set trace function
        sys.settrace(init_tracer)
        
        return result
    
    def complete_diagnostic_session(self, result: DiagnosticResult):
        """
        Complete a diagnostic session and save results.
        
        Args:
            result: DiagnosticResult to finalize and save
        """
        # Clear trace function
        sys.settrace(None)
        
        # Calculate duration
        end_time = time.time()
        result.end_time = end_time
        result.duration = end_time - result.start_time
        
        # Save result
        self.results.append(result)
        self._save_result(result)
        
        # Log summary
        if result.success:
            logger.info(f"Model initialization successful: {result.model_class} in {result.duration:.2f}s")
        else:
            logger.error(
                f"Model initialization failed: {result.model_class} - {result.error_type}: {result.error_message}"
            )
    
    def _save_result(self, result: DiagnosticResult):
        """Save diagnostic result to file."""
        # Create filename with timestamp and model class
        timestamp = datetime.fromtimestamp(result.start_time).strftime("%Y%m%d_%H%M%S")
        model_name = result.model_class.split(".")[-1]
        status = "success" if result.success else "failure"
        filename = f"{timestamp}_{model_name}_{status}.json"
        
        # Save to file
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved diagnostic result to {filepath}")
    
    def generate_diagnostic_report(self, all_results: bool = False):
        """
        Generate a comprehensive diagnostic report.
        
        Args:
            all_results: Whether to include all results or just the latest ones
            
        Returns:
            Path to the generated report
        """
        # Use all results or just the ones from this session
        results_to_report = self._load_all_results() if all_results else self.results
        
        # Generate report filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"diagnostic_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write("# TTS Model Initialization Diagnostic Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            total = len(results_to_report)
            successful = sum(1 for r in results_to_report if r.success)
            f.write(f"## Summary Statistics\n\n")
            f.write(f"- Total initialization attempts: {total}\n")
            f.write(f"- Successful: {successful} ({successful/total*100:.1f}%)\n")
            f.write(f"- Failed: {total-successful} ({(total-successful)/total*100:.1f}%)\n\n")
            
            # Group by model class
            by_model = {}
            for result in results_to_report:
                model_class = result.model_class
                if model_class not in by_model:
                    by_model[model_class] = []
                by_model[model_class].append(result)
            
            # Report by model class
            f.write(f"## Results by Model\n\n")
            for model_class, model_results in by_model.items():
                total_model = len(model_results)
                successful_model = sum(1 for r in model_results if r.success)
                f.write(f"### {model_class}\n\n")
                f.write(f"- Total attempts: {total_model}\n")
                f.write(f"- Success rate: {successful_model/total_model*100:.1f}%\n")
                
                # Average initialization time
                if successful_model > 0:
                    avg_time = sum(r.duration for r in model_results if r.success) / successful_model
                    f.write(f"- Average initialization time: {avg_time:.2f}s\n")
                
                # Common errors
                if total_model > successful_model:
                    f.write("\n#### Common Errors\n\n")
                    error_types = {}
                    for result in model_results:
                        if not result.success and result.error_type:
                            error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
                    
                    for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                        f.write(f"- {error_type}: {count} occurrences ({count/(total_model-successful_model)*100:.1f}%)\n")
                
                f.write("\n")
            
            # Detailed failure analysis
            f.write("## Detailed Failure Analysis\n\n")
            for result in results_to_report:
                if not result.success:
                    f.write(f"### {result.model_class} - {datetime.fromtimestamp(result.start_time).strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(f"**Error**: {result.error_type}: {result.error_message}\n\n")
                    
                    # Parameter values
                    if result.parameter_values:
                        f.write("**Parameters**:\n```json\n")
                        f.write(json.dumps(result.parameter_values, indent=2))
                        f.write("\n```\n\n")
                    
                    # Validation errors
                    if result.validation_errors:
                        f.write("**Validation Errors**:\n")
                        for param, errors in result.validation_errors.items():
                            for error in errors:
                                f.write(f"- {error}\n")
                        f.write("\n")
                    
                    # Initialization trace
                    if result.initialization_trace:
                        f.write("**Initialization Trace**:\n")
                        for i, trace in enumerate(result.initialization_trace):
                            f.write(f"{i+1}. {trace}\n")
                        f.write("\n")
                    
                    # Traceback
                    if result.traceback:
                        f.write("**Traceback**:\n```\n")
                        f.write(result.traceback)
                        f.write("\n```\n\n")
                    
                    f.write("---\n\n")
        
        logger.info(f"Diagnostic report generated: {report_file}")
        return report_file
    
    def _load_all_results(self) -> List[DiagnosticResult]:
        """Load all saved diagnostic results."""
        results = []
        
        for filepath in self.output_dir.glob("*.json"):
            if filepath.name.startswith("diagnostic_report"):
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                # Convert back to DiagnosticResult
                result = DiagnosticResult(
                    success=data["success"],
                    model_class=data["model_class"],
                    start_time=datetime.fromisoformat(data["start_time"]).timestamp(),
                    end_time=datetime.fromisoformat(data["end_time"]).timestamp(),
                    duration=data["duration"],
                    error_type=data.get("error_type"),
                    error_message=data.get("error_message"),
                    traceback=data.get("traceback"),
                    parameter_values=data.get("parameter_values", {}),
                    memory_usage=data.get("memory_usage", {}),
                    initialization_trace=data.get("initialization_trace", []),
                    validation_errors=data.get("validation_errors", {})
                )
                
                results.append(result)
            except Exception as e:
                logger.warning(f"Error loading diagnostic result from {filepath}: {e}")
        
        return results


@contextmanager
def diagnostic_mode(model_class, diagnostics=None):
    """
    Context manager to temporarily instrument a model class with diagnostics.
    
    Args:
        model_class: The model class to instrument
        diagnostics: Optional diagnostics instance to use
    
    Yields:
        The instrumented model class
    """
    if diagnostics is None:
        diagnostics = ModelInitializationDiagnostics()
    
    # Save the original __init__
    original_init = model_class.__init__
    
    try:
        # Instrument the class
        instrumented_class = diagnostics.instrument_model_class(model_class)
        yield instrumented_class
    finally:
        # Restore the original __init__
        model_class.__init__ = original_init


def test_kokoro_with_diagnostics():
    """Test Kokoro model initialization with diagnostics."""
    try:
        from mlx_audio.tts.models.kokoro import Model, ModelConfig
        
        diagnostics = ModelInitializationDiagnostics()
        
        # Test proper initialization
        logger.info("\n=== Testing Kokoro with Diagnostics - Proper Config ===")
        config_dict = {
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
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
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
            "vocab": {" ": 16, "a": 43, "b": 44, "c": 45}  # Simplified vocab
        }
        
        # Instrument ModelConfig first
        with diagnostic_mode(ModelConfig, diagnostics):
            model_config = ModelConfig(**config_dict)
        
        # Then test model initialization with diagnostics
        with diagnostic_mode(Model, diagnostics):
            try:
                model = Model(model_config, repo_id=None)
                logger.info("Successfully created Model with diagnostics")
            except Exception as e:
                logger.error(f"Error creating Model with diagnostics: {e}")
        
        # Test with deliberately incorrect config
        logger.info("\n=== Testing Kokoro with Diagnostics - Bad Config ===")
        bad_config_dict = config_dict.copy()
        bad_config_dict["dim_in"] = -1  # Invalid value
        del bad_config_dict["max_dur"]  # Missing required field
        
        with diagnostic_mode(ModelConfig, diagnostics):
            try:
                bad_config = ModelConfig(**bad_config_dict)
                logger.info("Successfully created bad ModelConfig (unexpected)")
            except Exception as e:
                logger.info(f"Expected error creating bad ModelConfig: {e}")
        
        # Generate diagnostic report
        report_path = diagnostics.generate_diagnostic_report()
        logger.info(f"Diagnostic report generated at: {report_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_kokoro_with_diagnostics: {e}")
        logger.debug(traceback.format_exc())
        return False


def test_encodec_with_diagnostics():
    """Test Encodec model initialization with diagnostics."""
    try:
        from mlx_audio.codec.models.encodec.encodec import Encodec, EncodecConfig
        
        diagnostics = ModelInitializationDiagnostics()
        
        # Test proper initialization
        logger.info("\n=== Testing Encodec with Diagnostics - Proper Config ===")
        config_dict = {
            "model_type": "encodec",
            "audio_channels": 1,
            "num_filters": 32,
            "hidden_size": 128,
            "num_residual_layers": 1,
            "sampling_rate": 24000,
            "target_bandwidths": [1.5, 3.0, 6.0, 12.0],
            "upsampling_ratios": [8, 5, 4, 2]
        }
        
        # Instrument EncodecConfig
        with diagnostic_mode(EncodecConfig, diagnostics):
            encodec_config = EncodecConfig(**config_dict)
        
        # Test Encodec initialization with diagnostics
        with diagnostic_mode(Encodec, diagnostics):
            try:
                model = Encodec(encodec_config)
                logger.info("Successfully created Encodec with diagnostics")
            except Exception as e:
                logger.error(f"Error creating Encodec with diagnostics: {e}")
        
        # Test with deliberately incorrect config
        logger.info("\n=== Testing Encodec with Diagnostics - Bad Config ===")
        bad_config_dict = config_dict.copy()
        bad_config_dict["model_type"] = "invalid_model_type"  # Invalid value
        bad_config_dict["num_filters"] = -10  # Invalid value
        
        with diagnostic_mode(EncodecConfig, diagnostics):
            try:
                bad_config = EncodecConfig(**bad_config_dict)
                logger.info("Successfully created bad EncodecConfig (unexpected)")
            except Exception as e:
                logger.info(f"Expected error creating bad EncodecConfig: {e}")
        
        # Generate diagnostic report
        report_path = diagnostics.generate_diagnostic_report()
        logger.info(f"Diagnostic report generated at: {report_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_encodec_with_diagnostics: {e}")
        logger.debug(traceback.format_exc())
        return False


def main():
    """Run model diagnostics tests."""
    print("Model Initialization Diagnostics")
    print("===============================")
    
    success_count = 0
    
    print("\nTesting Kokoro model with diagnostics...")
    if test_kokoro_with_diagnostics():
        print("✅ Kokoro diagnostics test complete")
        success_count += 1
    else:
        print("❌ Kokoro diagnostics test failed")
    
    print("\nTesting Encodec model with diagnostics...")
    if test_encodec_with_diagnostics():
        print("✅ Encodec diagnostics test complete")
        success_count += 1
    else:
        print("❌ Encodec diagnostics test failed")
    
    print("\nDiagnostics Test Summary:")
    print(f"- Completed {success_count}/2 tests")
    print(f"- Results saved to ./diagnostics_output/")
    print(f"- Log saved to ./model_diagnostics.log")


if __name__ == "__main__":
    main()
