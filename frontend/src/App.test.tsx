// frontend/src/App.test.tsx
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event'; // Use userEvent for more realistic interactions
import App from './App';
import { useAppStore } from './store'; // Import store to check state

// Mock the useAudioStreamer hook as its internal logic (Web Audio) is not needed for this integration test
// We only care that the component calls the hook's functions correctly based on UI interaction.
const mockPlay = vi.fn();
const mockPause = vi.fn();
const mockResume = vi.fn();
const mockStop = vi.fn();

vi.mock('./hooks/useAudioStreamer', () => ({
  useAudioStreamer: () => ({
    play: mockPlay,
    pause: mockPause,
    resume: mockResume,
    stop: mockStop,
    // Provide dummy state values if needed by components
    // isPlaying: false,
    // isBuffering: false,
    // error: null,
    // duration: 0,
    // currentTime: 0,
  }),
}));

// Reset store and mocks before each test
beforeEach(() => {
  useAppStore.setState(useAppStore.getState(), true); // Reset store state
  vi.clearAllMocks(); // Clear mock call history
});

describe('<App /> Integration Tests', () => {
  it('should update input text when user types', async () => {
    render(<App />);
    const user = userEvent.setup();
    const textInput = screen.getByRole('textbox'); // Assuming TextInput uses <textarea> or <input type="text">

    await user.type(textInput, 'Hello test');

    expect(textInput).toHaveValue('Hello test');
    // Optionally check if Zustand state was updated (though maybe redundant if input works)
    // expect(useAppStore.getState().inputText).toBe('Hello test');
  });

  it('should call requestPlayback and enter Buffering state when Play is clicked with text', async () => {
    render(<App />);
    const user = userEvent.setup();
    const textInput = screen.getByRole('textbox');
    const playButton = screen.getByRole('button', { name: /play/i }); // Find button by accessible name (adjust regex if needed)

    const testText = 'Synthesize this audio.';
    await user.type(textInput, testText);

    // Act: Click the play button
    await user.click(playButton);

    // Assert: Check if the store state was updated correctly by requestPlayback
    await waitFor(() => {
      const state = useAppStore.getState();
      expect(state.playbackState).toBe('Buffering');
      expect(state.displayText).toBe(testText);
      expect(state.errorMessage).toBeNull();
    });

    // Assert: Check if the mocked play function from useAudioStreamer was called
    // This confirms the UI interaction triggered the hook interaction.
    // We need to wait because the state update might trigger the effect calling play
    await waitFor(() => {
       expect(mockPlay).toHaveBeenCalledTimes(1);
       // Optionally check arguments if needed:
       // expect(mockPlay).toHaveBeenCalledWith(testText, expect.any(String), expect.any(Number));
    });

    // We can also check if the UI reflects the buffering state, e.g., button disabled or status text changes
    // Example: Check if status bar shows "Buffering..."
    // expect(screen.getByText(/buffering/i)).toBeInTheDocument();
  });

   it('should NOT call requestPlayback if Play is clicked with empty text', async () => {
    render(<App />);
    const user = userEvent.setup();
    const playButton = screen.getByRole('button', { name: /play/i });
    const initialPlaybackState = useAppStore.getState().playbackState;

    // Act: Click play with empty input
    await user.click(playButton);

    // Assert: Store state should remain unchanged (or handle validation state)
    // Using waitFor to ensure no async state change happens unexpectedly
    await waitFor(() => {
       expect(useAppStore.getState().playbackState).toBe(initialPlaybackState); // Should still be 'Idle'
    });

    // Assert: Mock play function should NOT have been called
    expect(mockPlay).not.toHaveBeenCalled();

    // Optional: Check for a validation message if implemented
    // expect(screen.queryByText(/please enter some text/i)).toBeInTheDocument();
  });

  // TODO: Add tests for Pause, Resume, Stop button interactions
  // TODO: Add tests for error states (e.g., if mockPlay simulated an error via the store)
});
