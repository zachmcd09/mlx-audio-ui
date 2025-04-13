import sys
import traceback

# --- Configuration ---
BARK_REPO_ID = "suno/bark-small" # Or "suno/bark" for the larger model

# --- Attempt to import necessary components ---
try:
    # Import our custom model initialization utility for Bark
    from model_init_utils import initialize_bark_model
    print("Successfully imported initialize_bark_model from model_init_utils.")
except ImportError as e:
    print(f"Error: Failed to import 'initialize_bark_model' from 'model_init_utils'.", file=sys.stderr)
    print(f"Make sure 'model_init_utils.py' exists and is accessible.", file=sys.stderr)
    print(f"Details: {e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during import:", file=sys.stderr)
    print(traceback.format_exc(), file=sys.stderr)
    sys.exit(1)

# --- Main Initialization Logic ---
if __name__ == "__main__":
    print(f"\n--- Attempting Isolated Bark Initialization ---")
    print(f"Using repo_id: {BARK_REPO_ID}")

    initialized_pipeline = None
    try:
        # Call the initialization utility function
        print("Calling initialize_bark_model...")
        initialized_pipeline = initialize_bark_model(BARK_REPO_ID)

        # Check if the pipeline was successfully initialized
        if initialized_pipeline:
            print("\n✅ SUCCESS: Bark pipeline initialized successfully!")
            print(f"   Initialized pipeline type: {type(initialized_pipeline)}")
        else:
            # This case might indicate an issue within the utility function itself
            print("\n⚠️ WARNING: initialize_bark_model returned None without raising an exception.")

    except Exception as e:
        print(f"\n❌ FAILURE: An error occurred during Bark pipeline initialization.", file=sys.stderr)
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"   Error Type: {error_type}", file=sys.stderr)
        print(f"   Error Message: {error_msg}", file=sys.stderr)
        print(f"\n--- Traceback ---", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        print(f"--- End Traceback ---", file=sys.stderr)
        sys.exit(1) # Exit with error code if initialization fails

    print("\n--- Bark Initialization Script Finished ---")
