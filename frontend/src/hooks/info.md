# Hooks (`frontend/src/hooks/info.md`)

This directory contains custom React hooks used throughout the frontend application to encapsulate reusable stateful logic and side effects.

## `useAudioStreamer.ts`

This is the core hook responsible for managing the entire audio playback lifecycle.

**Responsibilities:**

*   **Audio Context Management:** Initializes and manages the browser's `AudioContext`.
*   **API Interaction:** Calls the `fetchTTSAudioStream` service to request audio synthesis from the backend.
*   **Header Parsing:** Reads custom headers (`X-Audio-Sample-Rate`, `X-Audio-Total-Chunks`, etc.) from the backend response to configure playback and progress indication. Updates relevant Zustand state (`_setTotalChunks`) and internal state (`sampleRate`).
*   **Stream Processing:** Reads the `ReadableStream` of raw PCM audio data returned by the backend.
*   **Audio Decoding:** Decodes incoming audio chunks (expected to be 16-bit PCM) into `AudioBuffer` objects suitable for the Web Audio API.
*   **Buffering/Queueing:** Manages a queue of decoded `AudioBuffer`s to ensure smooth playback.
*   **Playback Scheduling:** Uses `AudioBufferSourceNode`s to schedule and play the decoded audio chunks sequentially. Relies on the `onended` event to trigger the next chunk.
*   **State Synchronization:** Updates the global Zustand store (`useAppStore`) with the current playback state (`_setPlaybackState`), progress (`_setCurrentChunkIndex`), and any errors (`_setErrorMessage`).
*   **Playback Controls:** Exposes functions (`play`, `pause`, `resume`, `stop`, `adjustSpeed`) to be called by UI components (like `PlayerControls`) to manage the audio stream.
*   **Resource Cleanup:** Handles stopping playback and releasing resources (AudioContext, stream reader, source nodes) when stopped or encountering errors.

This hook isolates the complexities of the Web Audio API and streaming logic from the UI components.
