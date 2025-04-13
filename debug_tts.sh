#!/bin/bash
# Main script to run the comprehensive TTS debugging system

echo "======================================================="
echo "TTS Model Debugging System"
echo "======================================================="

# Make scripts executable
chmod +x setup_test_env.py
chmod +x model_analyzer.py
chmod +x minimal_model_factory.py
chmod +x model_diagnostics.py
chmod +x debug_tts_models.py

# Check if output directory was specified
OUTPUT_DIR="debug_report"
SKIP_ENV=""

# Parse command line args
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output-dir) OUTPUT_DIR="$2"; shift ;;
        --skip-environment) SKIP_ENV="--skip-environment" ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

echo "Output directory: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

# Run the orchestrator script
python debug_tts_models.py --output-dir "$OUTPUT_DIR" $SKIP_ENV

# Check if successful
if [ $? -eq 0 ]; then
    echo "Debug process completed successfully."
    echo "Report available in $OUTPUT_DIR directory."
    
    # Open the HTML report if it exists
    REPORT=$(find "$OUTPUT_DIR" -name "tts_debug_report_*.html" | sort | tail -n 1)
    if [ ! -z "$REPORT" ]; then
        echo "Opening debug report: $REPORT"
        if [[ "$OSTYPE" == "darwin"* ]]; then
            open "$REPORT"  # macOS
        elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
            xdg-open "$REPORT"  # Linux
        elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
            start "$REPORT"  # Windows
        else
            echo "Unable to automatically open the report. Please open it manually."
        fi
    fi
else
    echo "Debug process failed. Check logs for details."
fi

echo "======================================================="
