import React from 'react';
import { useAppStore } from '../../store';

function ProgressIndicator() {
  const currentChunkIndex = useAppStore((state) => state.currentChunkIndex);
  const totalChunks = useAppStore((state) => state.totalChunks);
  const playbackState = useAppStore((state) => state.playbackState);

  // Determine if progress should be shown
  const shouldShowProgress =
    (playbackState === 'Playing' || playbackState === 'Paused') && totalChunks > 0;

  if (!shouldShowProgress) {
    return null; // Don't render anything if not playing/paused or no chunks
  }

  // User-friendly 1-based index for display
  const displayIndex = currentChunkIndex + 1;

  // Ensure displayIndex doesn't exceed totalChunks visually
  const clampedDisplayIndex = Math.min(displayIndex, totalChunks);

  return (
    <div className="progress-indicator mt-4">
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>
          Chunk {clampedDisplayIndex} / {totalChunks}
        </span>
        {/* Optional: Add percentage */}
        {/* <span>{Math.round((clampedDisplayIndex / totalChunks) * 100)}%</span> */}
      </div>
      <progress
        value={clampedDisplayIndex}
        max={totalChunks}
        className="w-full h-2 [&::-webkit-progress-bar]:rounded-lg [&::-webkit-progress-value]:rounded-lg [&::-webkit-progress-bar]:bg-gray-300 [&::-webkit-progress-value]:bg-blue-500 [&::-moz-progress-bar]:bg-blue-500"
        aria-label="Playback progress"
      ></progress>
    </div>
  );
}

export default ProgressIndicator;
