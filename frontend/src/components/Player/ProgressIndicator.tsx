import React, { useEffect, useRef } from 'react';
import { useAppStore } from '../../store';
import { FaHeadphones, FaWaveSquare, FaStream, FaClock, FaRegLightbulb } from 'react-icons/fa';

function ProgressIndicator() {
  const currentChunkIndex = useAppStore((state) => state.currentChunkIndex);
  const totalChunks = useAppStore((state) => state.totalChunks);
  const playbackState = useAppStore((state) => state.playbackState);
  const progressBarRef = useRef<HTMLDivElement>(null);
  const progressContainerRef = useRef<HTMLDivElement>(null);

  // Determine if progress should be shown
  const shouldShowProgress =
    (playbackState === 'Playing' || playbackState === 'Paused' || playbackState === 'Buffering') && totalChunks > 0;

  // Ensure displayIndex doesn't exceed totalChunks visually
  const displayIndex = Math.min(currentChunkIndex + 1, totalChunks);
  const progressPercentage = totalChunks > 0 ? (displayIndex / totalChunks) * 100 : 0;
  
  // Add pulse animation effect when buffering
  useEffect(() => {
    if (progressBarRef.current) {
      if (playbackState === 'Buffering') {
        progressBarRef.current.classList.add('animate-pulse');
      } else {
        progressBarRef.current.classList.remove('animate-pulse');
      }
    }
  }, [playbackState]);

  // Generate chunk markers for improved visualization
  const renderChunkMarkers = () => {
    if (!shouldShowProgress || totalChunks <= 1) return null;
    
    const markers = [];
    // Only show markers if we have a reasonable number of chunks
    if (totalChunks <= 20) {
      for (let i = 1; i < totalChunks; i++) {
        const position = (i / totalChunks) * 100;
        markers.push(
          <div 
            key={i}
            style={{ left: `${position}%` }}
            className={`absolute top-0 bottom-0 w-0.5 ${
              i <= currentChunkIndex 
                ? 'bg-primary/40' 
                : 'bg-secondary/40'
            } transform-gpu opacity-60`}
          />
        );
      }
    }
    return markers;
  };

  // Empty state UI when no audio is playing
  if (!shouldShowProgress) {
    return (
      <div className="flex flex-col items-center justify-center py-6 px-3 space-y-3 bg-secondary/5 rounded-lg border border-border/30 transition-all duration-300 group hover:border-primary/20 hover:bg-secondary/10">
        <div className="flex items-center justify-center gap-2 text-sm text-muted-foreground font-medium">
          <FaHeadphones className="text-primary/70" size={15} />
          <span>Ready for audio playback</span>
        </div>
        
        <div className="w-full max-w-md h-2 bg-secondary/20 rounded-full overflow-hidden relative">
          {/* Subtle animation to indicate readiness */}
          <div className="absolute inset-0 bg-gradient-to-r from-transparent via-secondary/30 to-transparent bg-[length:200%_100%] animate-shimmer opacity-70"></div>
        </div>
        
        <div className="text-xs text-muted-foreground/70 flex items-center gap-1.5 px-3 py-1.5 bg-secondary/5 rounded-full">
          <FaRegLightbulb size={10} className="text-primary/50" />
          <span>Click Play to start audio synthesis</span>
        </div>
      </div>
    );
  }

  // Status message based on playback state
  const statusMessage = {
    'Playing': 'Processing audio in chunks',
    'Paused': 'Playback paused',
    'Buffering': 'Processing next chunk...'
  }[playbackState] || '';

  return (
    <div className="py-4 animate-in fade-in duration-300">
      {/* Top info section */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center">
          <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium rounded-full transition-all duration-300 ${
            playbackState === 'Buffering' 
              ? 'text-accent bg-accent/10 animate-[pulse_1.5s_ease-in-out_infinite]' 
              : playbackState === 'Paused'
                ? 'text-amber-500 bg-amber-500/10'
                : 'text-primary bg-primary/10'
          }`}>
            <FaStream size={10} className="opacity-70" />
            {playbackState === 'Buffering' ? 'Buffering...' : `Chunk ${displayIndex} of ${totalChunks}`}
          </span>
        </div>
        
        <div className="text-xs font-medium text-muted-foreground">
          {statusMessage}
        </div>
      </div>
      
      {/* Custom styled progress bar */}
      <div 
        ref={progressContainerRef}
        className="relative h-4 bg-secondary/20 rounded-full overflow-hidden mt-1.5 mb-3 group shadow-inner border border-border/20"
      >
        {/* Progress fill */}
        <div 
          ref={progressBarRef}
          style={{ width: `${progressPercentage}%` }}
          className={`h-full transition-all duration-300 ease-out ${
            playbackState === 'Paused'
              ? 'bg-amber-500'
              : playbackState === 'Buffering'
                ? 'bg-accent animate-[pulse_1.5s_ease-in-out_infinite]'
                : 'bg-gradient-to-r from-primary to-accent bg-[length:200%_100%] animate-gradient'
          }`}
          role="progressbar"
          aria-valuenow={displayIndex}
          aria-valuemin={0}
          aria-valuemax={totalChunks}
        />
        
        {/* Chunk markers */}
        {renderChunkMarkers()}
        
        {/* Progress marker - subtle dot showing exact position */}
        <div 
          style={{ left: `${progressPercentage}%` }}
          className={`absolute top-1/2 -translate-x-1/2 -translate-y-1/2 w-3.5 h-3.5 rounded-full bg-background border-2 shadow-sm transition-all duration-150 ${
            playbackState === 'Paused'
              ? 'border-amber-500 scale-100'
              : playbackState === 'Buffering'
                ? 'border-accent scale-100 animate-[pulse_1.5s_ease-in-out_infinite]'
                : 'border-primary scale-100 group-hover:scale-125'
          }`}
        />
        
        {/* Subtle shimmer effect on progress bar */}
        <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/10 to-white/0 opacity-50 pointer-events-none"></div>
      </div>
      
      {/* Progress details */}
      <div className="flex justify-between items-center text-xs">
        <div className="text-muted-foreground">
          {playbackState === 'Buffering' ? (
            <span className="inline-flex items-center gap-1.5 text-accent animate-[pulse_2s_ease-in-out_infinite]">
              <svg className="animate-spin h-3 w-3" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing audio data
            </span>
          ) : (
            <span className="flex items-center gap-1.5">
              <FaClock className="text-primary/60" size={11} />
              <span>{`Processing ${displayIndex} of ${totalChunks} chunks`}</span>
            </span>
          )}
        </div>
        
        <div className="font-semibold px-2 py-0.5 bg-primary/10 rounded-full text-primary text-xs">
          {Math.round(progressPercentage)}%
        </div>
      </div>
      
      {/* Add subtle info about chunking for first-time users */}
      {displayIndex < totalChunks && (
        <div className="mt-3 text-xs text-muted-foreground/80 flex items-center gap-1.5 justify-center bg-background/80 py-1.5 px-3 rounded-full border border-border/30 shadow-sm transition-all duration-300 hover:border-primary/20 hover:bg-primary/5">
          <svg width="12" height="12" viewBox="0 0 15 15" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-primary/70">
            <path d="M7.5 11C7.22386 11 7 11.2239 7 11.5C7 11.7761 7.22386 12 7.5 12C7.77614 12 8 11.7761 8 11.5C8 11.2239 7.77614 11 7.5 11ZM8 10V7H7V10H8Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path>
            <path d="M7.5 1C3.91015 1 1 3.91015 1 7.5C1 11.0899 3.91015 14 7.5 14C11.0899 14 14 11.0899 14 7.5C14 3.91015 11.0899 1 7.5 1ZM2 7.5C2 4.46243 4.46243 2 7.5 2C10.5376 2 13 4.46243 13 7.5C13 10.5376 10.5376 13 7.5 13C4.46243 13 2 10.5376 2 7.5Z" fill="currentColor" fillRule="evenodd" clipRule="evenodd"></path>
          </svg>
          <span>Audio is processed and streamed in sequential chunks</span>
        </div>
      )}
      
      <style>{`
        @keyframes shimmer {
          0% { background-position: -200% 0; }
          100% { background-position: 200% 0; }
        }
        
        .animate-shimmer {
          animation: shimmer 2s infinite linear;
        }
        
        @keyframes gradient {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        
        .animate-gradient {
          animation: gradient 3s ease infinite;
        }
      `}</style>
    </div>
  );
}

export default ProgressIndicator;
