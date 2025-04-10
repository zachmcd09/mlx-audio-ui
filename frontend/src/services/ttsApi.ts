// TODO: Make the API base URL configurable via environment variables
const API_BASE_URL = 'http://127.0.0.1:5001'; // Use IP address instead of localhost

/**
 * Fetches the TTS audio stream from the backend.
 *
 * @param text The text to synthesize.
 * @param voice The selected voice ID (optional).
 * @param speed The desired playback speed.
 * @returns A Promise that resolves to the Response object from the fetch call.
 *          The caller is responsible for handling the response body (ReadableStream).
 * @throws An error if the fetch request fails.
 */
export async function fetchTTSAudioStream(
  text: string,
  voice: string | null,
  speed: number
): Promise<Response> {
  const endpoint = `${API_BASE_URL}/synthesize_pcm`; // Matches backend endpoint

  try {
    const response = await fetch(endpoint, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: text,
        // Include voice and speed if the backend expects them
        // voice: voice,
        // speed: speed, // Speed might be handled client-side via playbackRate
      }),
    });

    if (!response.ok) {
      // Attempt to read error message from backend response body
      let errorBody = 'Unknown error';
      try {
        const errorData = await response.json();
        errorBody = errorData.error || JSON.stringify(errorData);
      } catch (e) {
        // Ignore if response body isn't JSON or is empty
        errorBody = `Server responded with status ${response.status}`;
      }
      throw new Error(`Failed to fetch audio stream: ${errorBody}`);
    }

    // Check for necessary headers (caller will use these)
    if (!response.headers.get('X-Audio-Total-Chunks')) {
       console.warn('Response missing X-Audio-Total-Chunks header.');
       // Decide if this should be a hard error or just a warning
       // throw new Error('Response missing X-Audio-Total-Chunks header.');
    }
     if (!response.headers.get('X-Audio-Sample-Rate')) {
       console.warn('Response missing X-Audio-Sample-Rate header.');
       // throw new Error('Response missing X-Audio-Sample-Rate header.');
    }
    // Add checks for other required audio param headers if needed

    return response; // Return the whole response object
  } catch (error) {
    console.error('Error fetching TTS audio stream:', error);
    // Re-throw the error so the caller (useAudioStreamer hook) can handle it
    throw error;
  }
}

// TODO: Add functions for file upload and URL parsing if needed later
// export async function parseFileContent(file: File): Promise<string> { ... }
// export async function fetchArticleText(url: string): Promise<string> { ... }
