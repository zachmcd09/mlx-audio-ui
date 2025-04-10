# Player Components (`frontend/src/components/Player/info.md`)

This directory contains components related to controlling and displaying the audio playback.

## Components

*   **`PlayerControls.tsx`**:
    *   **Purpose:** Provides the main user interface for controlling audio playback (Play, Pause, Stop) and adjusting settings like speed.
    *   **Implementation:** Renders buttons and a range slider (`<input type="range">`). Button appearance and disabled states change based on the current `playbackState` read from the Zustand store (`useAppStore`).
    *   **State Interaction:**
        *   Reads `playbackState`, `speed`, `displayText`, `voice` from `useAppStore`.
        *   Calls `requestPlayback`, `requestPause`, `requestStop`, `setSpeed` actions from `useAppStore` to signal intent.
        *   Receives control functions (`play`, `pause`, `stop`, `adjustSpeed`) from the `useAudioStreamer` hook via props (passed down from `App.tsx`). It calls these hook functions directly to trigger the actual audio operations.
    *   **Composition:** Renders the `ProgressIndicator` component internally.

*   **`ProgressIndicator.tsx`**:
    *   **Purpose:** Displays the current playback progress based on the audio chunks processed by the backend.
    *   **Implementation:** Shows text like "Chunk X / Y" and uses an HTML `<progress>` element for a visual bar. Renders conditionally based on playback state.
    *   **State Interaction:** Reads `currentChunkIndex` and `totalChunks` from `useAppStore` to calculate and display the progress. Reads `playbackState` to determine visibility.

*(Future components like a voice selector dropdown would also reside here.)*
