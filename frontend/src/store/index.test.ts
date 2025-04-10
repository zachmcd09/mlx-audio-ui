// frontend/src/store/index.test.ts
import { describe, it, expect, beforeEach, vi } from 'vitest'; // Merged imports
import { useAppStore, PlaybackState } from './index'; // Import the hook and types

// Define the initial state explicitly for resetting, including dummy actions
const initialState = {
  // State values
  inputText: '',
  displayText: '',
  playbackState: 'Idle' as PlaybackState,
  errorMessage: null as string | null,
  speed: 1.0,
  voice: null as string | null,
  totalChunks: 0,
  currentChunkIndex: -1,
  // Dummy action implementations (required for replace: true)
  setInputText: vi.fn(),
  setSpeed: vi.fn(),
  setVoice: vi.fn(),
  requestPlayback: vi.fn(),
  requestPause: vi.fn(),
  requestResume: vi.fn(),
  requestStop: vi.fn(),
  _setPlaybackState: vi.fn(),
  _setErrorMessage: vi.fn(),
  _clearError: vi.fn(),
  _setTotalChunks: vi.fn(),
  _setCurrentChunkIndex: vi.fn(),
};

describe('useAppStore', () => {
  // Reset the store to initial state before each test
  beforeEach(() => {
    useAppStore.setState(initialState, true); // true replaces the entire state
  });

  it('should have correct initial state', () => {
    const state = useAppStore.getState();
    expect(state.inputText).toBe(initialState.inputText);
    expect(state.displayText).toBe(initialState.displayText);
    expect(state.playbackState).toBe(initialState.playbackState);
    expect(state.errorMessage).toBe(initialState.errorMessage);
    expect(state.speed).toBe(initialState.speed);
    expect(state.voice).toBe(initialState.voice);
    expect(state.totalChunks).toBe(initialState.totalChunks);
    expect(state.currentChunkIndex).toBe(initialState.currentChunkIndex);
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

    // Act
    useAppStore.getState().setInputText(newText);

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
    useAppStore.getState().setSpeed(newSpeed);
    expect(useAppStore.getState().speed).toBe(newSpeed);
  });

  it('setVoice should update voice', () => {
    const newVoice = 'test-voice-id';
    useAppStore.getState().setVoice(newVoice);
    expect(useAppStore.getState().voice).toBe(newVoice);
  });

  it('requestPlayback should update state to Buffering and store parameters', () => {
    const text = 'Text for playback';
    const voice = 'voice-1';
    const speed = 1.2;

    useAppStore.getState().requestPlayback(text, voice, speed);

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
    useAppStore.getState()._setPlaybackState('Playing');
    expect(useAppStore.getState().playbackState).toBe('Playing');
    useAppStore.getState()._setPlaybackState('Paused');
    expect(useAppStore.getState().playbackState).toBe('Paused');
  });

  it('_setErrorMessage should update errorMessage and set state to Error', () => {
    const errorMsg = 'TTS failed';
    useAppStore.getState()._setErrorMessage(errorMsg);
    const state = useAppStore.getState();
    expect(state.errorMessage).toBe(errorMsg);
    expect(state.playbackState).toBe('Error');
  });

  it('_clearError should clear errorMessage and reset state if in Error state', () => {
    // Arrange: Set error state
    useAppStore.setState({ errorMessage: 'Some error', playbackState: 'Error' });

    // Act
    useAppStore.getState()._clearError();

    // Assert
    const state = useAppStore.getState();
    expect(state.errorMessage).toBeNull();
    expect(state.playbackState).toBe('Idle'); // Assumes reset to Idle
  });

   it('_clearError should only clear errorMessage if not in Error state', () => {
    // Arrange: Set non-error state with an old error message (unlikely scenario, but test)
    useAppStore.setState({ errorMessage: 'Old error', playbackState: 'Playing' });

    // Act
    useAppStore.getState()._clearError();

    // Assert
    const state = useAppStore.getState();
    expect(state.errorMessage).toBeNull();
    expect(state.playbackState).toBe('Playing'); // State should not change
  });

  it('_setTotalChunks should update totalChunks', () => {
    useAppStore.getState()._setTotalChunks(15);
    expect(useAppStore.getState().totalChunks).toBe(15);
  });

  it('_setCurrentChunkIndex should update currentChunkIndex', () => {
    useAppStore.getState()._setCurrentChunkIndex(3);
    expect(useAppStore.getState().currentChunkIndex).toBe(3);
  });

  // Note: Tests for requestPause, requestResume, requestStop are omitted here
  // as their primary role is signaling intent, and they don't directly modify
  // the state in a way that isn't covered by _setPlaybackState tests.
  // Their effect depends on the hook calling the internal actions.
});
