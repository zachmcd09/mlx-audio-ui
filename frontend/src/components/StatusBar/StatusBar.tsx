import React from 'react';
import { useAppStore } from '../../store';

function StatusBar() {
  const playbackState = useAppStore((state) => state.playbackState);
  const errorMessage = useAppStore((state) => state.errorMessage);
  const { _clearError } = useAppStore.getState(); // Action to clear error

  let statusMessage = '';
  let bgColor = 'bg-gray-500'; // Default background
  let textColor = 'text-white';

  switch (playbackState) {
    case 'Idle':
      statusMessage = 'Ready';
      bgColor = 'bg-blue-500';
      break;
    case 'Buffering':
      statusMessage = 'Buffering audio...';
      bgColor = 'bg-yellow-500';
      textColor = 'text-black';
      break;
    case 'Playing':
      statusMessage = 'Playing';
      bgColor = 'bg-green-500';
      break;
    case 'Paused':
      statusMessage = 'Paused';
      bgColor = 'bg-yellow-600';
      break;
    case 'Stopped':
      statusMessage = 'Stopped';
      bgColor = 'bg-gray-600';
      break;
    case 'Error':
      statusMessage = `Error: ${errorMessage || 'Unknown error'}`;
      bgColor = 'bg-red-600';
      break;
    default:
      statusMessage = 'Unknown State';
  }

  // If there's an error message, prioritize showing it
  const displayMessage = errorMessage ? `Error: ${errorMessage}` : statusMessage;
  const displayBgColor = errorMessage ? 'bg-red-600' : bgColor;
  const displayTextColor = errorMessage ? 'text-white' : textColor;

  return (
    <div
      className={`status-bar p-2 text-center text-sm font-medium ${displayBgColor} ${displayTextColor} transition-colors duration-300 ease-in-out`}
      role="status"
      aria-live="polite" // Announce changes to screen readers
    >
      {displayMessage}
      {/* Add a button to clear the error message */}
      {errorMessage && (
        <button
          onClick={_clearError}
          className="ml-4 px-2 py-0.5 bg-white text-red-600 rounded text-xs hover:bg-gray-200"
          aria-label="Clear error message"
        >
          Clear
        </button>
      )}
    </div>
  );
}

export default StatusBar;
