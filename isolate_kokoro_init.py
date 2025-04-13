import sys
import traceback

# --- Configuration ---
KOKORO_REPO_ID = "prince-canuma/Kokoro-82M"

# --- Attempt to import necessary components ---
try:
    # Import our custom model initialization utility for Kokoro
    from model_init_utils import initialize_kokoro_model
    print("Successfully imported initialize_kokoro_model from model_init_utils.")
except ImportError as e:
    print(f"Error: Failed to import 'initialize_kokoro_model' from 'model_init_utils'.", file=sys.stderr)
    print(f"Make sure 'model_init_utils.py' exists and is accessible.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# --- Main Initialization Logic ---
if __name__ == "__main__":
    print(f"\n--- Attempting Isolated Kokoro Initialization ---")
    print(f"Using repo_id: {KOKORO_REPO_ID}")

    initialized_model = None
    try:
        # Call the initialization utility function
        print("Calling initialize_kokoro_model...")
        initialized_model, _ = initialize_kokoro_model(KOKORO_REPO_ID)

        # Check if the model was successfully initialized
        if initialized_model:
            print("\n✅ SUCCESS: Kokoro model initialized successfully!")
            print(f"   Initialized model type: {type(initialized_model)}")
        else:
            # This case might indicate an issue within the utility function itself
            print("\n⚠️ WARNING: initialize_kokoro_model returned None without raising an exception.")

    except Exception as e:
        print(f"\n❌ FAILURE: An error occurred during Kokoro model initialization.", file=sys.stderr)
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"   Error Type: {error_type}", file=sys.stderr)
        print(f"   Error Message: {error_msg}", file=sys.stderr)
        print(f"\n--- Traceback ---", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"--- End Traceback ---", file=sys.stderr)
        sys.exit(1) # Exit with error code if initialization fails

    print("\n--- Kokoro Initialization Script Finished ---")
