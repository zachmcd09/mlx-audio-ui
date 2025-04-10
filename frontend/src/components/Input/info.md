# Input Components (`frontend/src/components/Input/info.md`)

This directory contains components responsible for handling user input, primarily the text that needs to be synthesized.

## Components

*   **`TextInput.tsx`**:
    *   **Purpose:** Provides the primary interface for users to paste or type text for TTS processing.
    *   **Implementation:** Renders a standard HTML `<textarea>`.
    *   **State Interaction:**
        *   Reads the current `inputText` value from the `useAppStore` (Zustand) to display it.
        *   On change (`onChange` event), it calls the `setInputText` action from `useAppStore` to update the global state with the new text content.

*(Future components for file upload or URL input would also reside here.)*
