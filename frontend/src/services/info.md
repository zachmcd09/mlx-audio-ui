# Services (`frontend/src/services/info.md`)

This directory contains modules responsible for interacting with external services or APIs.

## `ttsApi.ts`

This module handles communication with the Flask backend API for text-to-speech functionality.

**Responsibilities:**

*   **`fetchTTSAudioStream` Function:**
    *   Constructs the request URL for the `/synthesize_pcm` endpoint using a base URL (currently hardcoded to `http://127.0.0.1:5001`, should be made configurable).
    *   Sends a `POST` request with the text to be synthesized in the JSON body.
    *   Sets the appropriate `Content-Type: application/json` header.
    *   Uses the browser's `fetch` API to make the request.
    *   Performs basic error checking on the response status (`response.ok`).
    *   Throws an error if the fetch fails or the server returns a non-successful status code.
    *   Includes basic checks/warnings for expected custom headers (`X-Audio-*`) in the response.
    *   Returns the raw `Response` object to the caller (the `useAudioStreamer` hook), which is then responsible for processing the response body (the `ReadableStream`) and headers.

*(Placeholder TODOs exist in the code to make the API base URL configurable and potentially add functions for file/URL processing later.)*
