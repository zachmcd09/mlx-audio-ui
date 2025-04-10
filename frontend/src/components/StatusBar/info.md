# StatusBar Component (`frontend/src/components/StatusBar/info.md`)

This directory contains the component responsible for displaying the overall application status and error messages.

## Components

*   **`StatusBar.tsx`**:
    *   **Purpose:** Provides visual feedback to the user about the current state of the audio playback process (e.g., Ready, Buffering, Playing, Paused, Stopped, Error). It also displays specific error messages when issues occur.
    *   **Implementation:** Renders a `div` whose background color and text content change based on the application state. Includes a "Clear" button when an error message is present. Uses `role="status"` and `aria-live="polite"` for accessibility, ensuring screen readers announce status changes.
    *   **State Interaction:**
        *   Reads `playbackState` and `errorMessage` from the `useAppStore` (Zustand).
        *   Calls the `_clearError` action from `useAppStore` when the "Clear" button is clicked.
