// frontend/src/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

// Define your API request handlers here.
// Example:
// export const handlers = [
//   http.post('/api/tts', async ({ request }) => {
//     const { text } = await request.json();
//     if (!text) {
//       return new HttpResponse(null, { status: 400 });
//     }
//     // Simulate successful audio stream
//     const mockAudio = new ArrayBuffer(1024); // Placeholder
//     return new HttpResponse(mockAudio, {
//       status: 200,
//       headers: { 'Content-Type': 'audio/wav' },
//     });
//   }),
//   // Add handlers for error cases, other endpoints, etc.
// ];

const mockAudioChunk = new ArrayBuffer(1024); // Simulate some audio data bytes

export const handlers = [
  // Mock the TTS endpoint
  http.post('/api/tts', async ({ request }) => {
    // Optionally check request body if needed
    // const { text } = await request.json();
    // console.log('MSW intercepted /api/tts with text:', text);

    // Simulate a successful response with PCM audio data
    return new HttpResponse(mockAudioChunk, {
      status: 200,
      headers: {
        'Content-Type': 'audio/pcm',
        // Include the custom headers the frontend expects
        'X-Audio-Sample-Rate': '24000',
        'X-Audio-Channels': '1',
        'X-Audio-Bit-Depth': '16',
      },
    });
  }),

  // Add other handlers here if needed (e.g., for error simulation)
];
