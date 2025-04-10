# Frontend Architecture (`frontend/info.md`)

This document details the architecture and key components of the React frontend application for the MLX Audio UI project.

## Overview

The frontend is a single-page application (SPA) built using Vite, React, and TypeScript. It provides the user interface for interacting with the text-to-speech functionality provided by the Flask backend. Its primary responsibilities include:

*   Accepting user text input (initially via pasting).
*   Displaying playback controls (Play, Pause, Stop, Speed).
*   Communicating with the backend API (`/synthesize_pcm`) to request audio synthesis.
*   Receiving the streamed raw PCM audio data from the backend.
*   Decoding and playing the audio stream seamlessly using the Web Audio API.
*   Displaying playback status and progress information.
*   Managing application state (input text, playback status, configuration).

## Technology Stack

*   **Build Tool:** Vite
*   **Framework/Library:** React 18+
*   **Language:** TypeScript
*   **Styling:** Tailwind CSS
*   **State Management:** Zustand
*   **Audio Playback:** Web Audio API

## Directory Structure (`src/`)

```
src/
├── assets/             # Static assets (images, icons)
├── components/         # Reusable React UI components
│   ├── common/         # General-purpose components (Button, Slider, etc. - TBD)
│   ├── Input/          # Components for text input (TextInput.tsx)
│   ├── Layout/         # Layout structure components (TBD)
│   ├── Player/         # Playback-related UI (PlayerControls.tsx, ProgressIndicator.tsx)
│   └── StatusBar/      # Status display component (StatusBar.tsx)
├── hooks/              # Custom React hooks
│   └── useAudioStreamer.ts # Core audio streaming and playback logic
├── services/           # Modules for interacting with external APIs
│   └── ttsApi.ts       # Functions for calling the Flask backend
├── store/              # Global state management setup
│   └── index.ts        # Zustand store definition (useAppStore)
├── styles/             # Global CSS or styling utilities (TBD)
├── types/              # Shared TypeScript type definitions (TBD)
├── utils/              # Utility functions (TBD)
├── App.css             # Component-specific styles for App.tsx (minimal)
├── App.tsx             # Main application component, layout assembly
├── index.css           # Global styles, Tailwind directives
├── main.tsx            # Application entry point (renders App)
└── vite-env.d.ts       # Vite TypeScript environment types
```

## Core Components & Logic

*   **`App.tsx`:** The root component. Sets up the main layout, instantiates the `useAudioStreamer` hook, and renders the primary UI components (`StatusBar`, `TextInput`, `PlayerControls`), passing down necessary controls.
*   **`main.tsx`:** The entry point that renders the `App` component into the DOM.
*   **`TextInput.tsx`:** Provides a `<textarea>` for users to paste text. Updates the `inputText` state in the Zustand store on change.
*   **`PlayerControls.tsx`:** Displays Play/Pause/Stop buttons and a speed control slider. Reads relevant state (`playbackState`, `speed`, `displayText`) from Zustand. Dispatches actions (`requestPlayback`, `requestPause`, etc.) to Zustand and calls corresponding control functions provided by the `useAudioStreamer` hook (passed via props). Includes the `ProgressIndicator`.
*   **`ProgressIndicator.tsx`:** Displays the playback progress (e.g., "Chunk 5 / 25") based on `currentChunkIndex` and `totalChunks` state from Zustand. Uses a `<progress>` element for visual feedback. Renders conditionally based on playback state.
*   **`StatusBar.tsx`:** Displays the current `playbackState` (Idle, Buffering, Playing, Paused, Error) and any `errorMessage` from the Zustand store. Provides a button to clear errors.
*   **`useAppStore` (`store/index.ts`):** The Zustand store. Acts as the single source of truth for shared application state. Defines state variables (`inputText`, `playbackState`, `speed`, `totalChunks`, `currentChunkIndex`, `errorMessage`, etc.) and actions for updating state. Internal actions (`_set*`) are called by the `useAudioStreamer` hook to reflect audio processing status. UI components trigger request actions (`request*`, `set*`).
*   **`ttsApi.ts` (`services/`):** Contains the `fetchTTSAudioStream` function, responsible for making the `POST` request to the Flask backend's `/synthesize_pcm` endpoint. It handles sending the text and retrieving the `Response` object containing the audio stream and headers.
*   **`useAudioStreamer.ts` (`hooks/`):** The most complex piece. This custom hook encapsulates all logic related to audio streaming and playback:
    *   Manages the `AudioContext`.
    *   Calls `fetchTTSAudioStream` to initiate the backend request.
    *   Reads necessary headers from the response (`X-Audio-Sample-Rate`, `X-Audio-Total-Chunks`, etc.) and updates Zustand state (`_setTotalChunks`) and internal state (`sampleRate`).
    *   Reads the `ReadableStream` from the response body chunk by chunk.
    *   Decodes each chunk (assuming 16-bit PCM based on backend config) into an `AudioBuffer` using `_decodeSentenceData`.
    *   Manages a queue (`decodedQueueRef`) of decoded `AudioBuffer`s.
    *   Schedules playback of queued buffers using `AudioBufferSourceNode`s.
    *   Uses the `onended` event of audio nodes to trigger playback of the next queued chunk (`_tryPlayNextFromQueue`).
    *   Updates Zustand state (`_setPlaybackState`, `_setCurrentChunkIndex`, `_setErrorMessage`) based on playback events and errors.
    *   Provides control functions (`play`, `pause`, `resume`, `stop`, `adjustSpeed`) that are returned by the hook and used by `PlayerControls.tsx`.
    *   Handles cleanup of audio resources (`AudioContext`, source nodes, stream reader).

## Data Flow Summary

1.  User pastes text into `TextInput`.
2.  `TextInput` calls `setInputText` action in Zustand store.
3.  User clicks "Play" in `PlayerControls`.
4.  `PlayerControls` calls `requestPlayback` action (updating Zustand state to 'Buffering') AND calls the `play` function from the `useAudioStreamer` hook instance.
5.  `useAudioStreamer` hook's `play` function calls `fetchTTSAudioStream` in `ttsApi.ts`.
6.  `ttsApi.ts` sends `POST` request to Flask `/synthesize_pcm`.
7.  Flask backend generates audio chunks and streams the response.
8.  `useAudioStreamer` receives the response, reads headers (updating `totalChunks` in Zustand), and starts reading/decoding the stream body.
9.  Decoded audio chunks (`AudioBuffer`s) are queued.
10. The hook schedules playback of the first chunk. `_setPlaybackState('Playing')` and `_setCurrentChunkIndex(0)` are called.
11. `StatusBar`, `PlayerControls`, `ProgressIndicator` re-render based on updated Zustand state.
12. When an audio chunk finishes (`onended`), the hook attempts to play the next queued chunk, updating `_setCurrentChunkIndex`.
13. User interactions (Pause, Stop, Speed) call corresponding hook functions via `PlayerControls`, which update the audio playback and Zustand state.
