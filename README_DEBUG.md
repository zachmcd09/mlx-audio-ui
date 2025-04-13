# TTS Model Debugging Framework

This repository contains a comprehensive framework for debugging TTS model initialization issues in MLX Audio projects. It provides a systematic approach to identifying and resolving complex initialization problems with Kokoro, Bark, and Encodec models.

## Quick Start

To run the full debugging suite:

```bash
./debug_tts.sh
```

This will:
1. Set up an isolated test environment
2. Analyze model architectures
3. Test minimal models
4. Run detailed diagnostics
5. Generate a comprehensive HTML report

## Options

```bash
./debug_tts.sh --output-dir custom_dir --skip-environment
```

- `--output-dir`: Specify a custom output directory (default: `debug_report`)
- `--skip-environment`: Skip environment setup (useful for subsequent runs)

## Framework Components

| Component | Description |
|-----------|-------------|
| `setup_test_env.py` | Creates an isolated testing environment |
| `model_analyzer.py` | Analyzes model architectures and parameters |
| `minimal_model_factory.py` | Creates minimal model reproductions |
| `model_diagnostics.py` | Provides detailed initialization diagnostics |
| `debug_tts_models.py` | Orchestrates all debugging tools |

## Documentation

For a comprehensive guide to using this debugging framework, see:

- [TTS Debugging Guide](TTS_DEBUGGING_GUIDE.md) - Detailed guide with explanations and strategies

## Contributing

To extend this framework:

1. Add new model schemas in `model_diagnostics.py`
2. Implement custom parameter adapters in `minimal_model_factory.py`
3. Add test fixtures for known-good configurations
4. Enhance the reporting in `debug_tts_models.py`

## Outputs

The debugging process generates the following outputs:

- **Model Analysis**: JSON files in `model_analysis_results/`
- **Diagnostics**: Reports in `diagnostics_output/`
- **Unified Report**: HTML report in `debug_report/`
- **Logs**: Comprehensive logs in `*.log` files

## Troubleshooting

If the debugging process itself encounters issues:

1. Check the `tts_debug.log` file for detailed error messages
2. Run individual tools manually to isolate the issue
3. Ensure Python dependencies are properly installed
4. Check file permissions for writing logs and reports

## License

Same as the main MLX Audio project.
