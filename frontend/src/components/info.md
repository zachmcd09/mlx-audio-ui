# Components (`frontend/src/components/info.md`)

This directory contains all the React UI components used to build the frontend application.

## Structure

Components are organized into subdirectories based on their primary function or feature area:

*   **`Input/`**: Components related to user input methods (e.g., pasting text, uploading files, entering URLs).
    *   `TextInput.tsx`: A textarea for pasting text.
*   **`Player/`**: Components related to audio playback control and display.
    *   `PlayerControls.tsx`: Buttons (Play/Pause/Stop), speed slider.
    *   `ProgressIndicator.tsx`: Displays playback progress based on audio chunks.
*   **`StatusBar/`**: Components for displaying application status or feedback.
    *   `StatusBar.tsx`: Shows current playback state (Idle, Playing, Error, etc.) and error messages.
*   **`common/`**: (Placeholder) Intended for general-purpose, reusable UI elements (e.g., Button, Modal, Slider) that might be used across different feature areas.
*   **`Layout/`**: (Placeholder) Intended for components that define the overall page structure or layout (e.g., Header, Footer, Main content area wrapper).

## State Management Interaction

Most components interact with the global application state managed by Zustand (`useAppStore` defined in `src/store/index.ts`). They typically:

1.  **Select State:** Use the `useAppStore` hook to select only the specific pieces of state they need to render (e.g., `playbackState`, `inputText`, `currentChunkIndex`). This optimizes re-renders.
2.  **Dispatch Actions:** Get action functions from the store (`useAppStore.getState().someAction`) to signal user intent or update simple state (e.g., `setInputText`, `setSpeed`, `requestPlayback`).
3.  **Use Hooks:** Components like `PlayerControls` receive control functions (e.g., `play`, `pause`) from the `useAudioStreamer` hook (instantiated in `App.tsx` and passed down) to trigger complex audio operations.

This structure promotes separation of concerns, keeping UI components focused on rendering and user interaction, while state logic resides in the Zustand store and complex side effects (like audio handling) are encapsulated in custom hooks.
