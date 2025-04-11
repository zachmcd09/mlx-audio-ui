// frontend/src/store/index.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react'; // Import act
import { useAppStore, PlaybackState } from './index'; // Import the hook and types

// Define the initial state values only
const initialStateValues = {
  inputText: '',
  displayText: '',
  playbackState: 'Idle' as PlaybackState,
  errorMessage: null as string | null,
  speed: 1.0,
  voice: null as string | null,
  totalChunks: 0,
  currentChunkIndex: -1,
};

describe('useAppStore', () => {
  // Reset the store state before each test by merging initial values
  beforeEach(() => {
    // Get the original actions to avoid overwriting them
    const originalActions = {
        setInputText: useAppStore.getState().setInputText,
        setSpeed: useAppStore.getState().setSpeed,
        setVoice: useAppStore.getState().setVoice,
        requestPlayback: useAppStore.getState().requestPlayback,
        requestPause: useAppStore.getState().requestPause,
        requestResume: useAppStore.getState().requestResume,
        requestStop: useAppStore.getState().requestStop,
        _setPlaybackState: useAppStore.getState()._setPlaybackState,
        _setErrorMessage: useAppStore.getState()._setErrorMessage,
        _clearError: useAppStore.getState()._clearError,
        _setTotalChunks: useAppStore.getState()._setTotalChunks,
        _setCurrentChunkIndex: useAppStore.getState()._setCurrentChunkIndex,
    };
    useAppStore.setState({ ...initialStateValues, ...originalActions }, true); // Replace state including actions
  });

  it('should have correct initial state', () => {
    const state = useAppStore.getState();
    expect(state.inputText).toBe(initialStateValues.inputText); // Use initialStateValues
    expect(state.displayText).toBe(initialStateValues.displayText); // Use initialStateValues
    expect(state.playbackState).toBe(initialStateValues.playbackState); // Use initialStateValues
    expect(state.errorMessage).toBe(initialStateValues.errorMessage); // Use initialStateValues
    expect(state.speed).toBe(initialStateValues.speed); // Use initialStateValues
    expect(state.voice).toBe(initialStateValues.voice); // Use initialStateValues
    expect(state.totalChunks).toBe(initialStateValues.totalChunks); // Use initialStateValues
    expect(state.currentChunkIndex).toBe(initialStateValues.currentChunkIndex); // Use initialStateValues
  });

  // --- Test UI-triggered Actions ---

  it('setInputText should update inputText, displayText and reset relevant state', () => {
    // Arrange: Set some non-initial state first
    useAppStore.setState({
      playbackState: 'Playing',
      currentChunkIndex: 5,
      totalChunks: 10,
      errorMessage: 'Old error',
    });

    const newText = 'This is new input text.';

    // Act: Wrap state update in act
    act(() => {
      useAppStore.getState().setInputText(newText);
    });

    // Assert
    const state = useAppStore.getState();
    expect(state.inputText).toBe(newText);
    expect(state.displayText).toBe(newText); // Assumes direct mapping for now
    expect(state.playbackState).toBe('Idle'); // Should reset state
    expect(state.currentChunkIndex).toBe(-1); // Should reset progress
    expect(state.totalChunks).toBe(0); // Should reset progress
    expect(state.errorMessage).toBeNull(); // Should clear error
  });

  it('setSpeed should update speed', () => {
    const newSpeed = 1.5;
    act(() => {
      useAppStore.getState().setSpeed(newSpeed);
    });
    expect(useAppStore.getState().speed).toBe(newSpeed);
  });

  it('setVoice should update voice', () => {
    const newVoice = 'test-voice-id';
    act(() => {
      useAppStore.getState().setVoice(newVoice);
    });
    expect(useAppStore.getState().voice).toBe(newVoice);
  });

  it('requestPlayback should update state to Buffering and store parameters', () => {
    const text = 'Text for playback';
    const voice = 'voice-1';
    const speed = 1.2;

    act(() => {
      useAppStore.getState().requestPlayback(text, voice, speed);
    });

    const state = useAppStore.getState();
    expect(state.playbackState).toBe('Buffering');
    expect(state.displayText).toBe(text);
    expect(state.voice).toBe(voice);
    expect(state.speed).toBe(speed);
    expect(state.errorMessage).toBeNull();
    expect(state.currentChunkIndex).toBe(-1);
    expect(state.totalChunks).toBe(0);
  });

  // --- Test Internal Actions (called by hook) ---

  it('_setPlaybackState should update playbackState', () => {
    act(() => {
      useAppStore.getState()._setPlaybackState('Playing');
    });
    expect(useAppStore.getState().playbackState).toBe('Playing');
    act(() => {
      useAppStore.getState()._setPlaybackState('Paused');
    });
    expect(useAppStore.getState().playbackState).toBe('Paused');
  });

  it('_setErrorMessage should update errorMessage and set state to Error', () => {
    const errorMsg = 'TTS failed';
    act(() => {
      useAppStore.getState()._setErrorMessage(errorMsg);
    });
    const state = useAppStore.getState();
    expect(state.errorMessage).toBe(errorMsg);
    expect(state.playbackState).toBe('Error');
  });

  it('_clearError should clear errorMessage and reset state if in Error state', () => {
    // Arrange: Set error state
    act(() => {
      useAppStore.setState({ errorMessage: 'Some error', playbackState: 'Error' });
    });

    // Act
    act(() => {
      useAppStore.getState()._clearError();
    });

    // Assert
    const state = useAppStore.getState();
    expect(state.inputText).toBe(initialStateValues.inputText);
    expect(state.displayText).toBe(initialStateValues.displayText);
    expect(state.playbackState).toBe(initialStateValues.playbackState);
    expect(state.errorMessage).toBe(initialStateValues.errorMessage);
    expect(state.speed).toBe(initialStateValues.speed);
    expect(state.voice).toBe(initialStateValues.voice);
    expect(state.totalChunks).toBe(initialStateValues.totalChunks);
    expect(state.currentChunkIndex).toBe(initialStateValues.currentChunkIndex);
  });

   it('_clearError should only clear errorMessage if not in Error state', () => {
    // Arrange: Set non-error state with an old error message (unlikely scenario, but test)
    act(() => {
      useAppStore.setState({ errorMessage: 'Old error', playbackState: 'Playing' });
    });

    // Act
    act(() => {
      useAppStore.getState()._clearError();
    });

    // Assert
    const state = useAppStore.getState();
    expect(state.errorMessage).toBeNull();
    expect(state.playbackState).toBe('Playing'); // State should not change
  });

  it('_setTotalChunks should update totalChunks', () => {
    act(() => {
      useAppStore.getState()._setTotalChunks(15);
    });
    expect(useAppStore.getState().totalChunks).toBe(15);
  });

  it('_setCurrentChunkIndex should update currentChunkIndex', () => {
    act(() => {
      useAppStore.getState()._setCurrentChunkIndex(3);
    });
    expect(useAppStore.getState().currentChunkIndex).toBe(3);
  });

  // Note: Tests for requestPause, requestResume, requestStop are omitted here
  // as their primary role is signaling intent, and they don't directly modify
  // the state in a way that isn't covered by _setPlaybackState tests.
  // Their effect depends on the hook calling the internal actions.
});
