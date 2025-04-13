# app.py
import sys
import re
import time
import threading
from queue import Queue, Empty
from flask import Flask, request, Response, jsonify, stream_with_context
from flask_cors import CORS  # type: ignore # Import CORS
import numpy as np
import mlx.core as mx

# --- Assume TTS related imports are available ---
# You might put the TTS functions in a separate tts_engine.py and import them
try:
    from mlx_audio.tts.models.kokoro import KokoroPipeline, Model, ModelConfig
    from mlx_audio.tts.models.bark import BarkPipeline # Added Bark
    from mlx_audio.tts.utils import load_model
    # Import our custom model initialization utilities
    from model_init_utils import (
        initialize_kokoro_model,
        initialize_bark_model,
        get_kokoro_model_config
    )
except ImportError as e:
    print(f"Error: mlx-audio related modules not found. Make sure it's installed. Details: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
# Define available TTS models and their configurations
AVAILABLE_TTS_MODELS = {
    "kokoro": {
        "repo_id": "prince-canuma/Kokoro-82M",
        "pipeline_class": KokoroPipeline,
        "default_voice": "af_heart", # Kokoro specific voice/speaker
        "sample_rate": 24000,
        "init_args": {"lang_code": "a"}, # Extra args for KokoroPipeline constructor
        "name": "Kokoro (Default)",
        "description": "Standard neural voice, fast generation."
    },
    "bark": {
        "repo_id": "suno/bark-small", # Or suno/bark for larger model
        "pipeline_class": BarkPipeline,
        "default_voice": "en_speaker_6", # Example Bark speaker
        "sample_rate": 24000,
        "init_args": {}, # No extra args needed for BarkPipeline
        "name": "Bark",
        "description": "Expressive generative model, slower generation."
    }
    # Add more models here if needed
}
DEFAULT_MODEL_ID = "kokoro" # The key to use if none is specified in request

# General Audio Config (should ideally match model output)
AUDIO_CHANNELS = 1
AUDIO_BIT_DEPTH = 16 # 16-bit PCM

# --- Global TTS Pipelines ---
# Initialize expensive resources once on startup
tts_pipelines = {} # Dictionary to hold multiple loaded pipelines

def initialize_tts_pipelines():
    """Loads all configured TTS models and creates pipeline instances."""
    global tts_pipelines
    if not tts_pipelines: # Only initialize if empty
        print("Initializing TTS pipelines...")
        init_success_count = 0
        init_failure_count = 0
        
        # Track at least one successful model initialization
        any_success = False
        
        for model_id, config in AVAILABLE_TTS_MODELS.items():
            print(f"  Loading model '{model_id}' ({config['repo_id']})...")
            try:
                # Different initialization approach for each model type using our utilities
                if model_id == "kokoro":
                    # Use our utility to initialize the Kokoro model
                    print("  Using enhanced Kokoro model initialization...")
                    # <<< NEW LOG (Before Kokoro) >>>
                    print(f"    Attempting Kokoro initialization with repo_id: {config['repo_id']}")
                    model, _ = initialize_kokoro_model(config['repo_id'])
                    # <<< NEW LOG (After Kokoro Success) >>>
                    print(f"    Kokoro model object initialized successfully. Type: {type(model)}")
                    
                    # Create pipeline with the model
                    pipeline_instance = config['pipeline_class'](
                        lang_code=config['init_args']['lang_code'],
                        model=model,
                        repo_id=config['repo_id'],
                    )
                elif model_id == "bark":
                    # Use our utility to initialize the Bark model
                    print("  Using enhanced Bark model initialization...")
                    # <<< NEW LOG (Before Bark) >>>
                    print(f"    Attempting Bark initialization with repo_id: {config['repo_id']}")
                    pipeline_instance = initialize_bark_model(config['repo_id'])
                    # <<< NEW LOG (After Bark Success) >>>
                    print(f"    Bark pipeline object initialized successfully. Type: {type(pipeline_instance)}")
                else:
                    # For other pipeline types - fallback to default initialization
                    pipeline_instance = config['pipeline_class'](
                        repo_id=config['repo_id'], 
                        **config['init_args']
                    )
                
                # Basic validation of pipeline instance
                if pipeline_instance is None:
                    raise ValueError(f"Pipeline class returned None for {model_id}")
                
                # Store the successfully initialized pipeline
                tts_pipelines[model_id] = pipeline_instance
                init_success_count += 1
                any_success = True
                print(f"  ✓ Model '{model_id}' loaded successfully.")
                
            except Exception as e:
                init_failure_count += 1
                error_type = type(e).__name__
                error_msg = str(e)
                # <<< ENHANCED LOG (Inside Exception) >>>
                print(f"\n  ❌ ERROR during initialization for model_id='{model_id}', repo_id='{config['repo_id']}'", file=sys.stderr)
                print(f"  Error type: {error_type}", file=sys.stderr)
                print(f"  Details: {error_msg}", file=sys.stderr)
                
                # Print traceback for more detailed debugging
                import traceback
                print(f"  Traceback:\n{traceback.format_exc()}", file=sys.stderr)
                
                # Continue trying to load other models
        
        print(f"\nTTS pipeline initialization summary:")
        print(f"  - Successfully loaded: {init_success_count} model(s)")
        print(f"  - Failed to load: {init_failure_count} model(s)")
        
        if not any_success:
            error_msg = "FATAL: No TTS pipelines could be initialized. Check logs for details."
            print(f"\n{error_msg}", file=sys.stderr)
            raise RuntimeError(error_msg)
        
        # Update DEFAULT_MODEL_ID if necessary
        global DEFAULT_MODEL_ID
        if DEFAULT_MODEL_ID not in tts_pipelines and tts_pipelines:
            old_default = DEFAULT_MODEL_ID
            DEFAULT_MODEL_ID = next(iter(tts_pipelines.keys()))
            print(f"Note: Default model '{old_default}' is not available. Using '{DEFAULT_MODEL_ID}' as default instead.")
            
        print("TTS pipelines initialization complete.")
    else:
        print("TTS pipelines already initialized.")

# --- TTS Generation Logic (Adapted for Server) ---

def _generate_tts_for_chunk_thread(pipeline, text_chunk, voice, speed, result_queue): # Accepts pipeline instance
    """
    Worker function for TTS generation (runs in thread). Uses the provided pipeline.
    Puts raw audio bytes (int16) or None into the result_queue.
    """
    # print(f"  [Thread] Generating TTS for: \"{text_chunk[:30]}...\"") # Debug
    try:
        audio_data_bytes = None
        chunk_audio_segments = []
        cleaned_chunk = text_chunk.strip()

        if not cleaned_chunk:
             # --- DEBUG THREAD ---
             # print(f"  [Thread] Chunk '{cleaned_chunk[:20]}...' is empty, putting None in queue.")
             # --- END DEBUG ---
             result_queue.put(None)
             return

        # --- DEBUG THREAD ---
        print(f"  [Thread] Starting pipeline call for chunk '{cleaned_chunk[:20]}...' using {type(pipeline).__name__}")
        # --- END DEBUG ---
        # Use the passed pipeline instance
        for _, _, audio_segment in pipeline(cleaned_chunk, voice=voice, speed=speed):
             # --- DEBUG THREAD ---
             # print(f"  [Thread] Pipeline yielded segment type: {type(audio_segment)}")
             # --- END DEBUG ---
             if audio_segment is not None and audio_segment.size > 0:
                if audio_segment.ndim > 1:
                     audio_segment = audio_segment.flatten()
                chunk_audio_segments.append(audio_segment)

        if chunk_audio_segments:
            concatenated_mx_array = mx.concatenate(chunk_audio_segments)
            # Convert to NumPy float32
            np_array = np.array(concatenated_mx_array, copy=False).astype(np.float32, copy=False)
            # Replace NaN/Inf with 0 before clipping and casting
            np_array = np.nan_to_num(np_array, nan=0.0, posinf=1.0, neginf=-1.0) # Replace NaN with 0, clamp Inf
            # Scale float32 array (-1.0 to 1.0) to int16 range (-32768 to 32767)
            int16_array = np.clip(np_array * 32767, -32768, 32767).astype(np.int16)
            audio_data_bytes = int16_array.tobytes()
            # print(f"  [Thread] Generated {len(audio_data_bytes)} bytes.") # Debug
        # --- DEBUG THREAD ---
        # print(f"  [Thread] Finished processing chunk '{cleaned_chunk[:20]}...'. Putting result in queue (Type: {type(audio_data_bytes)}, Size: {len(audio_data_bytes) if audio_data_bytes else 'None'}).")
        # --- END DEBUG ---
        result_queue.put(audio_data_bytes)

    except Exception as e:
        print(f"\nError in TTS generation thread for chunk: \"{text_chunk[:50]}...\"", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        result_queue.put(None) # Signal error

# --- New Function for Full Audio Generation ---
def generate_full_tts_audio(pipeline, text_to_speak, voice, speed): # Accepts pipeline instance
    """Generates the full TTS audio and returns raw PCM bytes using the specified pipeline."""
    if pipeline is None:
        print("Error: Invalid TTS pipeline provided.", file=sys.stderr)
        return None # Indicate error

    print(f"Generating full audio for: \"{text_to_speak[:50]}...\" using {type(pipeline).__name__}")
    start_gen_time = time.time()
    try:
        all_audio_segments = []
        cleaned_text = text_to_speak.strip()
        if not cleaned_text:
            return b'' # Return empty bytes for empty text

        # Generate all segments from the provided pipeline
        for _, _, audio_segment in pipeline(cleaned_text, voice=voice, speed=speed):
            if audio_segment is not None and audio_segment.size > 0:
                if audio_segment.ndim > 1: # Ensure 1D array
                    audio_segment = audio_segment.flatten()
                all_audio_segments.append(audio_segment)

        if not all_audio_segments:
            print("Warning: No audio segments generated.")
            return b''

        # Concatenate, convert to NumPy float32, then int16 bytes
        concatenated_mx_array = mx.concatenate(all_audio_segments)
        np_array = np.array(concatenated_mx_array, copy=False).astype(np.float32, copy=False)
        # Replace NaN/Inf with 0 before clipping and casting
        np_array = np.nan_to_num(np_array, nan=0.0, posinf=1.0, neginf=-1.0) # Replace NaN with 0, clamp Inf
        int16_array = np.clip(np_array * 32767, -32768, 32767).astype(np.int16)
        audio_data_bytes = int16_array.tobytes()

        end_gen_time = time.time()
        print(f"Full audio generation took {end_gen_time - start_gen_time:.2f} seconds. Total bytes: {len(audio_data_bytes)}")
        return audio_data_bytes

    except Exception as e:
        print(f"\nError during full TTS generation: {e}", file=sys.stderr)
        return None # Indicate error


def stream_tts_audio(pipeline, text_to_speak, voice, speed): # Accepts pipeline instance
    """
    Generator function that yields raw PCM audio chunks using threaded lookahead
    with the specified pipeline.
    Returns a tuple: (generator, total_chunks)
    """
    if pipeline is None:
        print("Error: Invalid TTS pipeline provided for streaming.", file=sys.stderr)
        # Return a generator that yields nothing and 0 chunks
        def empty_generator():
            if False: yield b'' # Make it a generator
        return empty_generator(), 0

    print(f"Starting audio stream generation using {type(pipeline).__name__}...")
    cleaned_text = text_to_speak.strip()
    if not cleaned_text:
        # Return an empty generator and 0 chunks for empty input
        def empty_generator():
            if False: yield b'' # Make it a generator
        return empty_generator(), 0

    # Split text into non-empty chunks
    chunks = [chunk for chunk in re.split(r'(?<=[.?!])\s+|\n{2,}', cleaned_text) if chunk and chunk.strip()]

    if not chunks:
        print("Could not split text into speakable chunks.")
        # Return a generator that yields nothing and 0 chunks
        def empty_generator():
            if False: yield b'' # Make it a generator
        return empty_generator(), 0

    total_chunks = len(chunks)
    print(f"Processing {total_chunks} chunk(s) for streaming...")

    # Define the actual generator function internally
    def audio_chunk_generator():
        current_audio_bytes = None
        tts_thread = None
        result_queue = Queue(maxsize=1)

        for i in range(total_chunks):
            chunk = chunks[i].strip() # Should already be stripped, but be safe
            if not chunk: continue

            # --- Get audio for CURRENT chunk (i) ---
        if i == 0:
            # First chunk generated in main thread to ensure something is yielded quickly
            _generate_tts_for_chunk_thread(pipeline, chunk, voice, speed, result_queue) # Pass pipeline
            current_audio_bytes = result_queue.get()
        else:
            # Wait for the previous background thread to finish
            if tts_thread:
                tts_thread.join() # Wait for completion
                try:
                    # Use timeout in case thread hangs? Queue get should block appropriately.
                    current_audio_bytes = result_queue.get(block=True, timeout=30) # Timeout after 30s
                except Empty:
                     print(f"Timeout waiting for TTS result from thread for chunk {i}", file=sys.stderr)
                     current_audio_bytes = None # Treat as error/no data
                except Exception as e:
                     print(f"Error getting result from queue for chunk {i}: {e}", file=sys.stderr)
                     current_audio_bytes = None


        # --- Start TTS for NEXT chunk (i+1) in background ---
        next_chunk_index = i + 1
        if next_chunk_index < total_chunks:
            next_chunk = chunks[next_chunk_index].strip()
            if next_chunk:
                tts_thread = None # Clear previous thread object
                thread_args = (pipeline, next_chunk, voice, speed, result_queue) # Pass pipeline
                tts_thread = threading.Thread(target=_generate_tts_for_chunk_thread, args=thread_args)
                tts_thread.start()

        # --- Yield the CURRENT chunk's audio bytes ---
        if current_audio_bytes:
            # --- DEBUG STREAM ---
            # print(f"  [Stream] Yielding {len(current_audio_bytes)} bytes for chunk {i+1}/{total_chunks}...")
            # --- END DEBUG ---
            yield current_audio_bytes
        else:
                # --- DEBUG STREAM ---
                # print(f"  [Stream] No audio data to yield for chunk {i+1}/{total_chunks}.")
                # --- END DEBUG ---
                pass # Don't yield anything if bytes are None

        # The loop correctly handles joining the thread and getting the result for the final chunk in its last iteration.
        # No extra handling is needed after the loop.

        print("Finished audio stream generation.")

    # Return the generator function and the total chunk count
    return audio_chunk_generator(), total_chunks


# --- Flask App Setup ---
app = Flask(__name__)
# Initialize CORS, explicitly allowing the frontend origin
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}}, supports_credentials=True)

@app.before_request # Changed from before_first_request (deprecated)
def ensure_tts_initialized():
    app.logger.info(f"--- Flask Request Received: {request.method} {request.path} ---") # Use Flask logger
    # This ensures TTS pipelines are loaded before handling the first real request
    initialize_tts_pipelines() # Call the updated initialization function

@app.route('/')
def index():
     # Simple status page or basic UI placeholder
    return "MLX-Audio TTS Server is running."

@app.route('/voices', methods=['GET'])
def get_available_voices():
    """Returns a list of available TTS models/voices."""
    app.logger.info(">>> /voices endpoint hit <<<")
    voice_list = []
    # Ensure pipelines are loaded before listing
    if not tts_pipelines:
         initialize_tts_pipelines() # Should be called by before_request, but belt-and-suspenders

    for model_id, config in AVAILABLE_TTS_MODELS.items():
         # Only list models that were successfully loaded
         if model_id in tts_pipelines:
            voice_list.append({
                "id": model_id,
                "name": config.get("name", model_id), # Use configured name or model_id
                "description": config.get("description", "No description available."),
                # Add other relevant info if needed, e.g., language support
            })
    return jsonify(voice_list)

@app.route('/synthesize_pcm', methods=['POST'])
def synthesize_pcm_audio():
    app.logger.info(">>> /synthesize_pcm endpoint hit <<<") # Use Flask logger
    """
    Endpoint to synthesize speech from text and stream raw PCM audio.
    Accepts JSON: {"text": "...", "voice": "...", "speed": ...}
    Returns raw PCM audio stream with custom headers for parameters.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text = data.get('text')
    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    # --- Get parameters and select pipeline ---
    model_id_from_request = data.get('voice') # Frontend sends selected model ID here
    model_id = model_id_from_request if model_id_from_request in tts_pipelines else DEFAULT_MODEL_ID

    selected_pipeline = tts_pipelines.get(model_id)
    if selected_pipeline is None:
         # This case should ideally not happen if initialization succeeded and default exists
         app.logger.error(f"Pipeline for requested model ID '{model_id_from_request}' not found, and default '{DEFAULT_MODEL_ID}' also unavailable.")
         # Attempt to re-initialize just in case? Or return error.
         initialize_tts_pipelines()
         selected_pipeline = tts_pipelines.get(model_id) # Try getting default again
         if selected_pipeline is None:
             return jsonify({"error": f"Requested voice model '{model_id}' is unavailable and server failed to initialize TTS."}), 503 # Service Unavailable

    # Get model-specific config
    model_config = AVAILABLE_TTS_MODELS.get(model_id, {})
    # Use model-specific default voice/speaker if needed, or a generic default
    # Note: The 'voice' parameter might now refer to speaker within the model
    speaker_voice = data.get('speaker', model_config.get('default_voice', None)) # Allow overriding speaker via 'speaker' param
    sample_rate = model_config.get('sample_rate', 24000) # Use model's sample rate

    speed = float(data.get('speed', 1.0)) # Use a generic default speed
    speed = max(0.5, min(speed, 2.0)) # Clamp speed

    app.logger.info(f"Synthesizing using model: '{model_id}', speaker: '{speaker_voice}', speed: {speed}")

    # Get the generator and total chunks count
    audio_generator, total_chunks = stream_tts_audio(selected_pipeline, text, speaker_voice, speed)

    # Create the response object with the generator and mimetype
    response = Response(stream_with_context(audio_generator), mimetype='audio/pcm')

    # Add custom headers based on the selected model's config
    response.headers['X-Audio-Sample-Rate'] = str(sample_rate)
    response.headers['X-Audio-Channels'] = str(AUDIO_CHANNELS)
    response.headers['X-Audio-Bit-Depth'] = str(AUDIO_BIT_DEPTH)
    response.headers['X-Audio-Total-Chunks'] = str(total_chunks) # Add total chunks header
    # Expose custom headers including the new one
    response.headers['Access-Control-Expose-Headers'] = 'X-Audio-Sample-Rate, X-Audio-Channels, X-Audio-Bit-Depth, X-Audio-Total-Chunks'

    return response

# ---- Script Entry Point ----
# Removed the app.run() block.
# Use 'flask run' or a WSGI server (like gunicorn) to start the app.
# Example: flask --app app run --debug
# Example: gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 app:app
