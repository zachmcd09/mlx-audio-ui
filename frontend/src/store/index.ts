import { create } from 'zustand';

// Define the possible states for playback
export type PlaybackState =
  | 'Idle'
  | 'Buffering'
  | 'Playing'
  | 'Paused'
  | 'Stopped'
  | 'Error';

// Define the shape of our state
interface AppState {
  // Input and Content
  inputText: string; // Raw text pasted by the user
  displayText: string; // Text being processed (could be from paste, file, URL)

  // Playback Status
  playbackState: PlaybackState;
  errorMessage: string | null; // Any error message to display

  // Configuration
  speed: number; // Playback speed (e.g., 1.0)
  voice: string | null; // Selected voice ID (null if only one option or none selected)
  showAdvancedPanel: boolean; // Whether the advanced panel is shown

  // Progress Indication (driven by backend)
  totalChunks: number; // Total audio chunks (sentences/segments) expected
  currentChunkIndex: number; // Index of the chunk currently playing or last played (-1 if none)
  
  // Text highlighting during playback
  highlightedTextRange: {start: number, end: number} | null; // Range of text being highlighted during playback

  // --- Actions ---
  // Actions triggered by UI components
  setInputText: (text: string) => void;
  setSpeed: (speed: number) => void;
  setVoice: (voice: string | null) => void;
  toggleAdvancedPanel: () => void;
  // Actions that will likely trigger the audio hook (implementation details TBD in hook/component)
  requestPlayback: (text: string, voice: string | null, speed: number) => void;
  requestPause: () => void;
  requestResume: () => void;
  requestStop: () => void;

  // Internal actions (called by the audio hook to update shared state)
  _setPlaybackState: (state: PlaybackState) => void;
  _setErrorMessage: (message: string | null) => void;
  _clearError: () => void;
  _setTotalChunks: (count: number) => void;
  _setCurrentChunkIndex: (index: number) => void;
  _setHighlightedTextRange: (range: {start: number, end: number} | null) => void;
}

// Create the store hook
export const useAppStore = create<AppState>((set, get) => ({
  // --- Initial State Values ---
  inputText: '',
  displayText: '',
  playbackState: 'Idle',
  errorMessage: null,
  speed: 1.0,
  voice: null, // TODO: Set a default voice if available from backend later
  totalChunks: 0,
  currentChunkIndex: -1,
  highlightedTextRange: null,
  showAdvancedPanel: false,

  // --- Action Implementations ---

  // Actions triggered by UI
  setInputText: (text) => {
    // Reset progress/state when new text is input
    set({
      inputText: text,
      displayText: text, // For now, assume paste = display
      playbackState: 'Idle',
      currentChunkIndex: -1,
      totalChunks: 0,
      errorMessage: null,
    });
  },

  setSpeed: (newSpeed) => set({ speed: newSpeed }), // Hook will read this value

  setVoice: (newVoice) => set({ voice: newVoice }), // Hook will read this value
  
  toggleAdvancedPanel: () => set((state) => ({ showAdvancedPanel: !state.showAdvancedPanel })),

  // Actions to signal intent to the audio hook (hook needs to be called separately)
  requestPlayback: (text, voice, speed) => {
    // Set initial state before hook is called by the component
    set({
      playbackState: 'Buffering',
      displayText: text, // Store the text that is intended for playback
      currentChunkIndex: -1,
      totalChunks: 0, // Will be updated by hook after fetch
      errorMessage: null,
      voice: voice, // Store selected config
      speed: speed,
    });
    // Note: The component calling this action is responsible for also calling the hook's play function
  },

  requestPause: () => {
    // Optimistic UI update (optional) - hook will confirm via _setPlaybackState
    // if (get().playbackState === 'Playing') {
    //   set({ playbackState: 'Paused' });
    // }
    // Note: Component calls hook's pause function
  },

  requestResume: () => {
    // Optimistic UI update (optional) - hook will confirm
    // if (get().playbackState === 'Paused') {
    //   set({ playbackState: 'Buffering' });
    // }
    // Note: Component calls hook's resume function
  },

  requestStop: () => {
    // Optimistic UI update (optional) - hook will confirm
    // set({ playbackState: 'Stopped', currentChunkIndex: -1 });
    // Note: Component calls hook's stop function
  },

  // --- Internal Actions (Called by useAudioStreamer Hook) ---

  _setPlaybackState: (state) => set({ playbackState: state }),

  _setErrorMessage: (message) => {
    set({ errorMessage: message, playbackState: 'Error' });
  },

  _clearError: () => {
    // Clear error usually means returning to Idle or Stopped state
    if (get().playbackState === 'Error') {
      set({ errorMessage: null, playbackState: 'Idle' }); // Or 'Stopped' depending on context
    } else {
      set({ errorMessage: null });
    }
  },

  _setTotalChunks: (count) => set({ totalChunks: count }),

  _setCurrentChunkIndex: (index) => set({ currentChunkIndex: index }),
  
  // Helper to update text highlighting
  _setHighlightedTextRange: (range: {start: number, end: number} | null) => set({ 
    highlightedTextRange: range 
  }),
}));
