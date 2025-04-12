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
    from mlx_audio.tts.models.kokoro import KokoroPipeline
    from mlx_audio.tts.utils import load_model
except ImportError:
    print("Error: mlx-audio related modules not found. Make sure it's installed.", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
# MLX-Audio TTS Configuration (Can be overridden by request)
DEFAULT_TTS_MODEL_REPO_ID = "prince-canuma/Kokoro-82M"
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_TTS_SPEED = 1.0
DEFAULT_TTS_LANG_CODE = "a"
TTS_SAMPLE_RATE = 24000 # Kokoro's sample rate
AUDIO_CHANNELS = 1
AUDIO_BIT_DEPTH = 16 # 16-bit PCM

# --- Global TTS Pipeline ---
# Initialize expensive resources once on startup
tts_pipeline = None

def initialize_global_tts():
    """Loads the TTS model and creates the pipeline globally."""
    global tts_pipeline
    if tts_pipeline is None:
        print(f"Initializing global TTS model: {DEFAULT_TTS_MODEL_REPO_ID}...")
        try:
            model = load_model(DEFAULT_TTS_MODEL_REPO_ID)
            tts_pipeline = KokoroPipeline(lang_code=DEFAULT_TTS_LANG_CODE, model=model, repo_id=DEFAULT_TTS_MODEL_REPO_ID)
            print("Global TTS model loaded successfully.")
        except Exception as e:
            print(f"\nFATAL Error: Failed to load global TTS model '{DEFAULT_TTS_MODEL_REPO_ID}'.", file=sys.stderr)
            print(f"Details: {e}", file=sys.stderr)
            # Optionally exit or let Flask handle the error state
            raise RuntimeError("TTS Engine failed to initialize") from e
    else:
         print("Global TTS model already initialized.")

# --- TTS Generation Logic (Adapted for Server) ---

def _generate_tts_for_chunk_thread(pipeline, text_chunk, voice, speed, result_queue):
    """
    Worker function for TTS generation (runs in thread).
    Puts raw audio bytes (int16) or None into the result_queue.
    """
    # print(f"  [Thread] Generating TTS for: \"{text_chunk[:30]}...\"") # Debug
    try:
        audio_data_bytes = None
        chunk_audio_segments = []
        cleaned_chunk = text_chunk.strip()

        if not cleaned_chunk:
             # --- DEBUG THREAD ---
             print(f"  [Thread] Chunk '{cleaned_chunk[:20]}...' is empty, putting None in queue.")
             # --- END DEBUG ---
             result_queue.put(None)
             return

        # --- DEBUG THREAD ---
        print(f"  [Thread] Starting pipeline call for chunk '{cleaned_chunk[:20]}...'")
        # --- END DEBUG ---
        for _, _, audio_segment in pipeline(cleaned_chunk, voice=voice, speed=speed):
             # --- DEBUG THREAD ---
             # print(f"  [Thread] Pipeline yielded segment type: {type(audio_segment)}") # Potentially too verbose
             # --- END DEBUG ---
             if audio_segment is not None and audio_segment.size > 0:
                if audio_segment.ndim > 1:
                     audio_segment = audio_segment.flatten()
                chunk_audio_segments.append(audio_segment)

        if chunk_audio_segments:
            concatenated_mx_array = mx.concatenate(chunk_audio_segments)
            # Convert to NumPy, then to int16
            np_array = np.array(concatenated_mx_array, copy=False).astype(np.float32, copy=False)
            # Scale float32 array (-1.0 to 1.0) to int16 range (-32768 to 32767)
            int16_array = np.clip(np_array * 32767, -32768, 32767).astype(np.int16)
            audio_data_bytes = int16_array.tobytes()
            # print(f"  [Thread] Generated {len(audio_data_bytes)} bytes.") # Debug
        # --- DEBUG THREAD ---
        print(f"  [Thread] Finished processing chunk '{cleaned_chunk[:20]}...'. Putting result in queue (Type: {type(audio_data_bytes)}, Size: {len(audio_data_bytes) if audio_data_bytes else 'None'}).")
        # --- END DEBUG ---
        result_queue.put(audio_data_bytes)

    except Exception as e:
        print(f"\nError in TTS generation thread for chunk: \"{text_chunk[:50]}...\"", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        result_queue.put(None) # Signal error

# --- New Function for Full Audio Generation ---
def generate_full_tts_audio(text_to_speak, voice, speed):
    """Generates the full TTS audio and returns raw PCM bytes."""
    if tts_pipeline is None:
        print("Error: TTS pipeline not initialized.", file=sys.stderr)
        return None # Indicate error

    print(f"Generating full audio for: \"{text_to_speak[:50]}...\"")
    start_gen_time = time.time()
    try:
        all_audio_segments = []
        cleaned_text = text_to_speak.strip()
        if not cleaned_text:
            return b'' # Return empty bytes for empty text

        # Generate all segments from the pipeline
        for _, _, audio_segment in tts_pipeline(cleaned_text, voice=voice, speed=speed):
            if audio_segment is not None and audio_segment.size > 0:
                if audio_segment.ndim > 1:
                    audio_segment = audio_segment.flatten()
                all_audio_segments.append(audio_segment)

        if not all_audio_segments:
            print("Warning: No audio segments generated.")
            return b''

        # Concatenate, convert to NumPy float32, then int16 bytes
        concatenated_mx_array = mx.concatenate(all_audio_segments)
        np_array = np.array(concatenated_mx_array, copy=False).astype(np.float32, copy=False)
        int16_array = np.clip(np_array * 32767, -32768, 32767).astype(np.int16)
        audio_data_bytes = int16_array.tobytes()

        end_gen_time = time.time()
        print(f"Full audio generation took {end_gen_time - start_gen_time:.2f} seconds. Total bytes: {len(audio_data_bytes)}")
        return audio_data_bytes

    except Exception as e:
        print(f"\nError during full TTS generation: {e}", file=sys.stderr)
        return None # Indicate error


def stream_tts_audio(text_to_speak, voice, speed):
    """
    Generator function that yields raw PCM audio chunks using threaded lookahead.
    """
    # global tts_pipeline # Removed unnecessary global
    if tts_pipeline is None:
        print("Error: TTS pipeline not initialized.", file=sys.stderr)
        yield b'' # Yield empty bytes on error
        return

    print("Starting audio stream generation...")
    cleaned_text = text_to_speak.strip()
    if not cleaned_text:
        yield b''
        return

    # Split text into non-empty chunks
    chunks = [chunk for chunk in re.split(r'(?<=[.?!])\s+|\n{2,}', cleaned_text) if chunk and chunk.strip()]

    if not chunks:
        print("Could not split text into speakable chunks.")
        yield b''
        return

    total_chunks = len(chunks)
    print(f"Processing {total_chunks} chunk(s) for streaming...")

    current_audio_bytes = None
    tts_thread = None
    result_queue = Queue(maxsize=1)

    for i in range(total_chunks):
        chunk = chunks[i].strip() # Should already be stripped, but be safe
        if not chunk: continue

        # --- Get audio for CURRENT chunk (i) ---
        if i == 0:
            # First chunk generated in main thread to ensure something is yielded quickly
            _generate_tts_for_chunk_thread(tts_pipeline, chunk, voice, speed, result_queue)
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
                thread_args = (tts_pipeline, next_chunk, voice, speed, result_queue)
                tts_thread = threading.Thread(target=_generate_tts_for_chunk_thread, args=thread_args)
                tts_thread.start()


        # --- Yield the CURRENT chunk's audio bytes ---
        if current_audio_bytes:
            # --- DEBUG STREAM ---
            print(f"  [Stream] Yielding {len(current_audio_bytes)} bytes for chunk {i+1}/{total_chunks}...")
            # --- END DEBUG ---
            yield current_audio_bytes
        else:
            # --- DEBUG STREAM ---
            print(f"  [Stream] No audio data to yield for chunk {i+1}/{total_chunks}.")
            # --- END DEBUG ---

    # The loop correctly handles joining the thread and getting the result for the final chunk in its last iteration.
    # No extra handling is needed after the loop.

    print("Finished audio stream generation.")


# --- Flask App Setup ---
app = Flask(__name__)
# Initialize CORS globally (more permissive, often easier for dev)
CORS(app) # Apply default CORS settings to all routes

@app.before_request # Changed from before_first_request (deprecated)
def ensure_tts_initialized():
    app.logger.info(f"--- Flask Request Received: {request.method} {request.path} ---") # Use Flask logger
    # This ensures TTS is loaded before handling the first real request
    # In production with multiple workers, each worker might initialize it.
    # Consider more robust initialization patterns for production scale.
    initialize_global_tts()

@app.route('/')
def index():
     # Simple status page or basic UI placeholder
    return "MLX-Audio TTS Server is running."

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

    # Get optional parameters or use defaults
    voice_from_request = data.get('voice') # Get value, could be None
    voice = voice_from_request if voice_from_request is not None else DEFAULT_TTS_VOICE # Use default if None or key missing
    speed = float(data.get('speed', DEFAULT_TTS_SPEED))
    # Clamp speed to reasonable values
    speed = max(0.5, min(speed, 2.0))

    # # --- DIAGNOSTIC: Generate full audio instead of streaming ---
    # try:
    #     full_audio_bytes = generate_full_tts_audio(text, voice, speed)
    #     if full_audio_bytes is None:
    #         # Error occurred during generation
    #         return jsonify({"error": "TTS generation failed internally."}), 500
    #     elif not full_audio_bytes:
    #          # No audio generated (e.g., empty text after cleaning)
    #          return Response(b'', mimetype='audio/pcm') # Return empty response

    #     # Create the response object with the full audio data
    #     response = Response(full_audio_bytes, mimetype='audio/pcm')
    #     # No need for custom headers or stream_with_context for non-streaming
    #     return response

    # except Exception as e:
    #     app.logger.error(f"Error in /synthesize_pcm (non-streaming): {e}", exc_info=True)
    #     return jsonify({"error": "An unexpected error occurred during audio synthesis."}), 500
    # # --- END DIAGNOSTIC ---

    # Create the generator for the streaming response (Original Code)
    audio_generator = stream_tts_audio(text, voice, speed)
    # Create the response object with the generator and mimetype
    response = Response(stream_with_context(audio_generator), mimetype='audio/pcm')
    # Add custom headers so the client knows how to play the PCM stream
    response.headers['X-Audio-Sample-Rate'] = str(TTS_SAMPLE_RATE)
    response.headers['X-Audio-Channels'] = str(AUDIO_CHANNELS)
    response.headers['X-Audio-Bit-Depth'] = str(AUDIO_BIT_DEPTH)
    # Expose custom headers (Flask-CORS handles Access-Control-Allow-Origin)
    response.headers['Access-Control-Expose-Headers'] = 'X-Audio-Sample-Rate, X-Audio-Channels, X-Audio-Bit-Depth, X-Audio-Total-Chunks' # Added Total-Chunks
    # Removed manual Access-Control-Allow-Origin header
    # TODO: Add X-Audio-Total-Chunks header based on segmentation
    # total_chunks = len(segment_text(text)) # Assuming segment_text exists
    # response.headers['X-Audio-Total-Chunks'] = str(total_chunks)
    return response

# ---- Script Entry Point ----
# Removed the app.run() block.
# Use 'flask run' or a WSGI server (like gunicorn) to start the app.
# Example: flask --app app run --debug
# Example: gunicorn --workers 1 --threads 4 --bind 0.0.0.0:5000 app:app
