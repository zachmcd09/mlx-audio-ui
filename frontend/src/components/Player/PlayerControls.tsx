import React from 'react';
import { useAppStore } from '../../store';
import ProgressIndicator from './ProgressIndicator';
import { AudioStreamerControls } from '../../hooks/useAudioStreamer'; // Import the controls type

// Define props for the component
interface PlayerControlsProps {
  controls: AudioStreamerControls;
}

function PlayerControls({ controls }: PlayerControlsProps) { // Destructure controls from props
  // Select state needed for controls
  const playbackState = useAppStore((state) => state.playbackState);
  const speed = useAppStore((state) => state.speed);
  const displayText = useAppStore((state) => state.displayText); // Text to be played
  const voice = useAppStore((state) => state.voice); // Currently selected voice

  // Get actions to request playback changes
  const { requestPlayback, requestPause, requestStop, setSpeed } = useAppStore.getState();

  const handlePlay = async () => { // Make async if play is async
    if (displayText.trim()) {
      requestPlayback(displayText, voice, speed); // Update state first
      try {
        await controls.play({ text: displayText, voice, speed }); // Call hook's play function
      } catch (error) {
        console.error("Playback initiation failed:", error);
        // Error state is likely set within the hook, but could add specific UI feedback here
      }
    } else {
      console.warn('No text provided to play.');
    }
  };

  const handlePause = () => {
    requestPause(); // Update state
    controls.pause(); // Call hook's pause function
  };

  const handleStop = () => {
    requestStop(); // Update state
    controls.stop(); // Call hook's stop function
  };

  const handleSpeedChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newSpeed = parseFloat(event.target.value);
    setSpeed(newSpeed); // Update state
    // If playing, immediately adjust speed on the current node
    if (playbackState === 'Playing') {
      controls.adjustSpeed(newSpeed); // Call hook's adjustSpeed function
    }
  };

  const isPlaying = playbackState === 'Playing';
  const isPaused = playbackState === 'Paused';
  const isBuffering = playbackState === 'Buffering';
  const isIdle = playbackState === 'Idle' || playbackState === 'Stopped';
  const canPlay = !isBuffering && !isPlaying && !!displayText.trim();
  const canPause = isPlaying;
  const canStop = isPlaying || isPaused || isBuffering;

  return (
    <div className="player-controls p-4 bg-gray-200 rounded-md shadow sticky bottom-4">
      <div className="flex items-center justify-center space-x-4 mb-4">
        {/* Play/Pause Button */}
        {isPlaying ? (
          <button
            onClick={handlePause}
            disabled={!canPause}
            className="px-4 py-2 bg-yellow-500 text-white rounded-md hover:bg-yellow-600 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Pause"
          >
            Pause
          </button>
        ) : (
          <button
            onClick={handlePlay}
            disabled={!canPlay}
            className="px-4 py-2 bg-green-500 text-white rounded-md hover:bg-green-600 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label={isPaused ? "Resume" : "Play"}
          >
            {isPaused ? 'Resume' : 'Play'} {/* TODO: Implement Resume button action */}
          </button>
        )}

        {/* Stop Button */}
        <button
          onClick={handleStop}
          disabled={!canStop}
          className="px-4 py-2 bg-red-500 text-white rounded-md hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Stop"
        >
          Stop
        </button>
      </div>

      {/* Speed Control */}
      <div className="flex items-center space-x-2">
        <label htmlFor="speed-control" className="text-sm font-medium text-gray-700">
          Speed: {speed.toFixed(1)}x
        </label>
        <input
          id="speed-control"
          type="range"
          min="0.5"
          max="2.0"
          step="0.1"
          value={speed}
          onChange={handleSpeedChange}
          className="w-full h-2 bg-gray-300 rounded-lg appearance-none cursor-pointer"
        />
      </div>

      {/* TODO: Add Voice Selection Dropdown */}

      {/* Add Progress Indicator Here */}
      <ProgressIndicator />
    </div>
  );
}

export default PlayerControls;
