import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../../store';
import ProgressIndicator from './ProgressIndicator';
// Assuming API base URL is defined elsewhere or use full URL
// TODO: Refactor API calls into a dedicated service file if not already done
const API_BASE_URL = 'http://127.0.0.1:5000';
import AudioVisualizer, { VisualizationMode } from './AudioVisualizer';
import VoiceSelector from '../Input/VoiceSelector';
import { AudioStreamerControls } from '../../hooks/useAudioStreamer';
import { 
  FaPlay, FaPause, FaStop, FaVolumeUp, 
  FaVolumeDown, FaRedo, FaDownload, FaUser,
  FaWaveSquare, FaRobot, FaUserAlt, FaInfoCircle,
  FaMicrophone, FaCheck
} from 'react-icons/fa';

interface PlayerControlsProps {
  controls: AudioStreamerControls;
}

function PlayerControls({ controls }: PlayerControlsProps) {
  // Visualization state
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>('frequency');
  const showAdvancedPanel = useAppStore((state) => state.showAdvancedPanel);
  const playbackState = useAppStore((state) => state.playbackState);
  const speed = useAppStore((state) => state.speed);
  const displayText = useAppStore((state) => state.displayText);
  const voice = useAppStore((state) => state.voice); // This now holds the model ID ('kokoro', 'bark', etc.)
  const setVoice = useAppStore((state) => state.setVoice);
  const [availableVoices, setAvailableVoices] = useState<{ id: string; name: string; description: string; icon?: JSX.Element }[]>([]);
  const [isLoadingVoices, setIsLoadingVoices] = useState(true);
  const [activeButton, setActiveButton] = useState<string | null>(null);
  const [activeTooltip, setActiveTooltip] = useState<string | null>(null);
  const [downloadStatus, setDownloadStatus] = useState<string | null>(null);
  const speedControlRef = useRef<HTMLInputElement>(null);

  // State for error details when voice fetching fails
  const [voiceLoadError, setVoiceLoadError] = useState<{
    message: string;
    details?: string;
    retry?: () => void;
  } | null>(null);
  
  // Function to fetch voices with retry capability
  const fetchVoices = async () => {
    setIsLoadingVoices(true);
    setVoiceLoadError(null);
    
    try {
      console.log("Fetching voices from server...");
      const response = await fetch(`${API_BASE_URL}/voices`);
      
      if (!response.ok) {
        const errorText = await response.text().catch(() => "Unknown error");
        throw new Error(`Server responded with ${response.status}: ${errorText}`);
      }
      
      const data = await response.json();
      
      // Check if we got empty data
      if (!Array.isArray(data) || data.length === 0) {
        throw new Error("Server returned empty voice list. TTS models may not be initialized properly.");
      }
      
      console.log(`Loaded ${data.length} voices from server`);
      
      // Add icons based on ID or name
      const voicesWithIcons = data.map((v: any) => ({
        ...v,
        icon: v.id === 'kokoro' ? <FaWaveSquare /> : 
              v.id === 'bark' ? <FaRobot /> : 
              <FaMicrophone />
      }));
      
      setAvailableVoices(voicesWithIcons);
      
      // Set default voice in store if not already set or if current voice is invalid
      if (!voice || !voicesWithIcons.some((v: any) => v.id === voice)) {
        if (voicesWithIcons.length > 0) {
          setVoice(voicesWithIcons[0].id);
        }
      }
    } catch (error) {
      console.error("Failed to fetch voices:", error);
      
      // Set detailed error information
      setVoiceLoadError({
        message: "Failed to load voice models",
        details: error instanceof Error ? error.message : "Unknown error",
        retry: fetchVoices  // Function to retry loading
      });
      
      // Set fallback error state in the voices dropdown
      setAvailableVoices([{ 
        id: 'default_error', 
        name: 'Error Loading Voices', 
        description: 'The server failed to initialize TTS models. Check server logs for details.' 
      }]);
    } finally {
      setIsLoadingVoices(false);
    }
  };

  // Fetch available voices from the backend on component mount
  useEffect(() => {
    fetchVoices();
  }, [voice, setVoice]); // Include voice and setVoice in dependency array

  // Calculate progress value for range input styling
  const progressPercent = ((speed - 0.5) / 1.5) * 100;

  const { requestPlayback, requestPause, requestStop, setSpeed } = useAppStore.getState();

  // Function to show tooltip for a specific amount of time
  const showTooltip = (id: string) => {
    setActiveTooltip(id);
    setTimeout(() => {
      if (activeTooltip === id) {
        setActiveTooltip(null);
      }
    }, 2000);
  };

  const handlePlay = async () => {
    console.log("handlePlay called!"); // <<< ADDED LOG >>>
    if (displayText.trim()) {
      setActiveButton('play');
      requestPlayback(displayText, voice, speed);
      try {
        await controls.play({ text: displayText, voice, speed });
      } catch (error) {
        console.error("Playback initiation failed:", error);
      } finally {
        setActiveButton(null);
      }
    } else {
      console.warn('No text provided to play.');
      showTooltip('play-error');
    }
  };

  const handlePause = () => {
    setActiveButton('pause');
    requestPause();
    controls.pause();
    setTimeout(() => setActiveButton(null), 200);
  };

  const handleStop = () => {
    setActiveButton('stop');
    requestStop();
    controls.stop();
    setTimeout(() => setActiveButton(null), 200);
  };

  const handleSpeedChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newSpeed = parseFloat(event.target.value);
    setSpeed(newSpeed);
    if (playbackState === 'Playing') {
      controls.adjustSpeed(newSpeed);
    }
  };

  const handleDownload = () => {
    // This would be replaced with actual download functionality
    setActiveButton('save');
    setDownloadStatus('downloading');
    
    setTimeout(() => {
      setDownloadStatus('complete');
      setTimeout(() => {
        setDownloadStatus(null);
        setActiveButton(null);
      }, 2000);
    }, 1500);
  };

  // Update range input background on mount and when speed changes
  useEffect(() => {
    if (speedControlRef.current) {
      speedControlRef.current.style.background = `linear-gradient(to right, 
        hsl(var(--primary)) 0%, 
        hsl(var(--primary)) ${progressPercent}%, 
        rgba(0, 0, 0, 0.1) ${progressPercent}%, 
        rgba(0, 0, 0, 0.1) 100%)`;
    }
  }, [speed, progressPercent]);

  // Removed hardcoded voices array

  const isPlaying = playbackState === 'Playing';
  const isPaused = playbackState === 'Paused';
  const isBuffering = playbackState === 'Buffering';
  const canPlay = !isBuffering && !isPlaying && !!displayText.trim();
  const canPause = isPlaying;
  const canStop = isPlaying || isPaused || isBuffering;

  // Get voice details from the fetched state
  const selectedVoice = availableVoices.find(v => v.id === voice) || availableVoices[0] || { id: '', name: 'Loading...', description: '' };

  console.log("canPlay:", canPlay); // <<< ADDED LOG >>>
  return (
    <div className="rounded-xl bg-card text-card-foreground shadow-lg border border-border/60 p-6 mt-6 relative overflow-hidden transition-all duration-200 group hover:-translate-y-0.5 hover:shadow-xl">
      {/* Animated accent at the top */}
      <div className={`absolute top-0 left-0 right-0 h-1 ${
        isPlaying 
          ? 'bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] animate-gradient'
          : isPaused
            ? 'bg-amber-500'
            : isBuffering
              ? 'bg-accent'
              : 'bg-muted'
      } transition-all duration-500`} />
      
      {/* Add animation keyframes */}
      <style>
        {`
          @keyframes gradient {
            0% { background-position: 0% 50% }
            50% { background-position: 100% 50% }
            100% { background-position: 0% 50% }
          }
          
          .animate-gradient {
            animation: gradient 6s ease infinite;
          }
          
          @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
          }

          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(5px); }
            to { opacity: 1; transform: translateY(0); }
          }

          @keyframes shimmer {
            0% { opacity: 0.7; }
            50% { opacity: 1; }
            100% { opacity: 0.7; }
          }
          
          .glass-effect {
            background-color: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(4px);
            -webkit-backdrop-filter: blur(4px);
          }
          
          .animate-slide-up {
            animation: slideUp 0.3s ease-out forwards;
          }
          
          @keyframes slideUp {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          
          @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-5px); }
          }
        `}
      </style>
      
      {/* Section title */}
      <div className="flex items-center justify-between mb-5">
        <h3 className="m-0 text-base font-semibold text-foreground flex items-center tracking-tight">
          <FaMicrophone className="mr-2 text-primary" size={14} />
          <span>Voice Controls</span>
          {isPlaying && (
            <span className="inline-flex items-center ml-3 text-xs font-medium text-primary px-2 py-1 rounded-full bg-primary/10 animate-[shimmer_2s_ease-in-out_infinite]">
              <span className="w-1.5 h-1.5 rounded-full bg-primary mr-1.5"></span>
              Active
            </span>
          )}
        </h3>
        
        {/* Help tooltip */}
        <div className="relative">
          <button
            onClick={() => setActiveTooltip(activeTooltip === 'help' ? null : 'help')}
            className="w-7 h-7 flex items-center justify-center rounded-full border-none bg-muted/20 text-muted-foreground cursor-pointer transition-all duration-150 hover:bg-muted/30 hover:text-foreground"
            aria-label="Help"
          >
            <FaInfoCircle size={12} />
          </button>
          
          {activeTooltip === 'help' && (
            <div className="absolute top-full mt-2 right-0 w-56 p-3 bg-popover text-popover-foreground rounded-lg shadow-lg text-xs leading-relaxed z-10 border border-border/80 animate-[fadeIn_0.2s_ease-out_forwards]">
              <p className="m-0 mb-2 font-medium">Audio Playback Tips:</p>
              <ul className="m-0 pl-4">
                <li className="mb-1">Adjust speed before or during playback.</li>
                <li className="mb-1">Long texts are processed in chunks.</li>
                <li className="mb-1">Different voices have varying characteristics.</li>
                <li>Click Save to download the current audio.</li>
              </ul>
            </div>
          )}
          
          {activeTooltip === 'play-error' && (
            <div className="absolute bottom-full mb-2 right-0 w-48 p-3 bg-destructive/10 text-destructive rounded-lg shadow-lg text-xs z-10 animate-[fadeIn_0.2s_ease-out_forwards]">
              <div className="flex items-center">
                <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" className="mr-2">
                  <path d="M8 5V9M8 11H8.01M15 8C15 11.866 11.866 15 8 15C4.13401 15 1 11.866 1 8C1 4.13401 4.13401 1 8 1C11.866 1 15 4.13401 15 8Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                </svg>
                <span className="font-medium">Please enter text first</span>
              </div>
            </div>
          )}
        </div>
      </div>
      
      {/* Error message if voice loading failed */}
      {voiceLoadError && (
        <div className="mb-4 p-4 bg-destructive/10 text-destructive rounded-lg border border-destructive/30 animate-[fadeIn_0.3s_ease-out_forwards]">
          <div className="flex items-start">
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg" className="mr-3 mt-0.5 shrink-0">
              <path d="M10 6V10M10 14H10.01M19 10C19 14.9706 14.9706 19 10 19C5.02944 19 1 14.9706 1 10C1 5.02944 5.02944 1 10 1C14.9706 1 19 5.02944 19 10Z" 
                stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
            </svg>
            <div className="flex-1">
              <h4 className="font-semibold text-sm mb-1">{voiceLoadError.message}</h4>
              <p className="text-xs mb-3">{voiceLoadError.details}</p>
              {voiceLoadError.retry && (
                <button 
                  onClick={voiceLoadError.retry}
                  className="inline-flex items-center justify-center rounded-md py-1 px-3 text-xs font-medium bg-background/30 hover:bg-background/50 text-destructive transition-colors"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" 
                    strokeWidth="2" className="mr-1.5" xmlns="http://www.w3.org/2000/svg">
                    <path d="M3 12C3 16.9706 7.02944 21 12 21C16.9706 21 21 16.9706 21 12C21 7.02944 16.9706 3 12 3M12 3L16 7M12 3L8 7" 
                      strokeLinecap="round" strokeLinejoin="round"/>
                  </svg>
                  <span>Retry Loading Voices</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Voice & Speed Controls Row */}
      <div className="flex flex-wrap gap-5 justify-between items-stretch mb-6 bg-muted/5 p-5 rounded-xl border border-border/40">
        {/* Voice selection with the new VoiceSelector component */}
        <div className="flex-1 min-w-[200px] flex flex-col">
          <label 
            className="flex items-center text-sm font-medium text-foreground mb-2"
          >
            <FaVolumeUp className="mr-2 text-primary" size={14} />
            <span>Voice</span>
          </label>
          
          <VoiceSelector 
            disabled={isPlaying || isBuffering} 
          />
        </div>
        
        {/* Speed control with enhanced styling */}
        <div className="flex-1 min-w-[220px] flex flex-col">
          <label 
            htmlFor="speed-control" 
            className="flex items-center text-sm font-medium text-foreground mb-2"
          >
            {speed < 1 ? (
              <FaVolumeDown className="mr-2 text-primary" size={14} />
            ) : (
              <FaVolumeUp className="mr-2 text-primary" size={14} />
            )}
            <span>Playback Speed:</span>
            <span className="font-semibold ml-1.5 text-primary">{speed.toFixed(1)}x</span>
          </label>
          
          <div className="flex items-center gap-3 mt-1">
            <span className="text-xs font-medium text-muted-foreground">0.5x</span>
            
            <div className="relative flex-1 h-6 flex items-center">
              <input
                id="speed-control"
                ref={speedControlRef}
                type="range"
                min="0.5"
                max="2.0"
                step="0.1"
                value={speed}
                onChange={handleSpeedChange}
                className="w-full h-1.5 rounded-full outline-none cursor-pointer appearance-none m-0"
              />
              {/* Custom range track styling */}
              <style>
                {`
                  input[type='range'] {
                    -webkit-appearance: none;
                    appearance: none;
                    background: transparent;
                  }
                  
                  input[type='range']::-webkit-slider-runnable-track {
                    height: 4px;
                    border-radius: 2px;
                  }
                  
                  input[type='range']::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    appearance: none;
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: hsl(var(--primary));
                    cursor: pointer;
                    border: 2px solid white;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
                    margin-top: -6px;
                    transition: all 100ms ease;
                  }
                  
                  input[type='range']::-webkit-slider-thumb:hover {
                    transform: scale(1.1);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                  }
                  
                  input[type='range']::-webkit-slider-thumb:active {
                    transform: scale(0.95);
                  }
                  
                  input[type='range']::-moz-range-track {
                    height: 4px;
                    border-radius: 2px;
                  }
                  
                  input[type='range']::-moz-range-thumb {
                    width: 16px;
                    height: 16px;
                    border-radius: 50%;
                    background: hsl(var(--primary));
                    cursor: pointer;
                    border: 2px solid white;
                    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
                    transition: all 100ms ease;
                  }
                  
                  input[type='range']::-moz-range-thumb:hover {
                    transform: scale(1.1);
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
                  }
                  
                  input[type='range']::-moz-range-thumb:active {
                    transform: scale(0.95);
                  }
                `}
              </style>
            </div>
            
            <span className="text-xs font-medium text-muted-foreground">2.0x</span>
          </div>
          
          {/* Speed description */}
          <p className="mt-2 mb-0 text-xs text-muted-foreground leading-relaxed">
            Adjust playback speed from slower (0.5x) to faster (2.0x).
          </p>
        </div>
      </div>

      {/* Audio visualizer */}
      <div className="mb-5">
        <AudioVisualizer 
          audioContext={controls.getAudioContext()} 
          sourceNode={controls.getCurrentSourceNode()}
          mode={visualizationMode}
          showControls={showAdvancedPanel}
        />
      </div>
      
      {/* Progress Indicator with improved styling */}
      <ProgressIndicator />
      
      {/* Playback Controls - enhanced styling */}
      <div className="flex justify-center items-center gap-4 mt-5 flex-wrap">
        {/* Play/Pause Button */}
        {isPlaying ? (
          <button
            onClick={handlePause}
            disabled={!canPause}
            className={`relative inline-flex items-center justify-center rounded-full py-0 px-5 h-11 text-sm font-semibold tracking-tight transition-all duration-180 shadow-sm min-w-28 
              ${activeButton === 'pause' ? 'scale-[0.98] shadow-inner' : ''}
              ${canPause 
                ? 'bg-amber-100 text-amber-800 hover:-translate-y-0.5 hover:shadow active:scale-[0.98]' 
                : 'opacity-60 cursor-not-allowed pointer-events-none bg-muted/80 text-muted-foreground border border-border/60'
              }`}
            aria-label="Pause"
          >
            <FaPause className="mr-2" size={14} />
            <span>Pause</span>
            
            {/* Subtle glow effect for active button */}
            {canPause && (
              <span className="absolute inset-0 rounded-full bg-amber-500/20 blur-md -z-10"></span>
            )}
          </button>
        ) : (
          <button
            onClick={() => { console.log("Play button onClick triggered!"); handlePlay(); }} // <<< MODIFIED onClick >>>
            disabled={!canPlay}
            className={`relative inline-flex items-center justify-center rounded-full py-0 px-5 h-11 text-sm font-semibold tracking-tight transition-all duration-180 shadow-sm min-w-28 
              ${activeButton === 'play' ? 'scale-[0.98] shadow-inner' : ''}
              ${canPlay 
                ? 'bg-gradient-to-r from-primary to-accent text-primary-foreground hover:-translate-y-0.5 hover:shadow active:scale-[0.98]' 
                : 'opacity-60 cursor-not-allowed pointer-events-none bg-muted/80 text-muted-foreground border border-border/60'
              }`}
            aria-label={isPaused ? 'Resume' : 'Play'}
          >
            {isPaused ? <FaRedo className="mr-2" size={14} /> : <FaPlay className="mr-2" size={14} />}
            <span>{isPaused ? 'Resume' : 'Play'}</span>
            
            {/* Subtle glow effect for active button */}
            {canPlay && (
              <span className="absolute inset-0 rounded-full bg-primary/20 blur-md -z-10"></span>
            )}
          </button>
        )}
        
        {/* Stop Button */}
        <button
          onClick={handleStop}
          disabled={!canStop}
          className={`relative inline-flex items-center justify-center rounded-full py-0 px-5 h-11 text-sm font-semibold tracking-tight transition-all duration-180 shadow-sm min-w-28 
            ${activeButton === 'stop' ? 'scale-[0.98] shadow-inner' : ''}
            ${canStop 
              ? 'bg-destructive text-destructive-foreground hover:-translate-y-0.5 hover:shadow active:scale-[0.98]' 
              : 'opacity-60 cursor-not-allowed pointer-events-none bg-muted/80 text-muted-foreground border border-border/60'
            }`}
          aria-label="Stop"
        >
          <FaStop className="mr-2" size={14} />
          <span>Stop</span>
          
          {/* Subtle glow effect for active button */}
          {canStop && (
            <span className="absolute inset-0 rounded-full bg-destructive/20 blur-md -z-10"></span>
          )}
        </button>
        
        {/* Download Button (mocked functionality) */}
        <button
          onClick={handleDownload}
          disabled={!displayText.trim() || downloadStatus === 'downloading'}
          className={`relative inline-flex items-center justify-center rounded-full py-0 px-5 h-11 text-sm font-semibold tracking-tight transition-all duration-180 shadow-sm min-w-28 
            ${activeButton === 'save' ? 'scale-[0.98] shadow-inner' : ''}
            ${displayText.trim() && downloadStatus !== 'downloading'
              ? 'bg-accent/90 text-accent-foreground hover:-translate-y-0.5 hover:shadow active:scale-[0.98]' 
              : 'opacity-60 cursor-not-allowed pointer-events-none bg-muted/80 text-muted-foreground border border-border/60'
            }`}
          aria-label="Download audio"
        >
          {downloadStatus === 'complete' ? (
            <>
              <FaCheck className="mr-2" size={14} />
              <span>Saved</span>
            </>
          ) : downloadStatus === 'downloading' ? (
            <>
              <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <span>Saving...</span>
            </>
          ) : (
            <>
              <FaDownload className="mr-2" size={14} />
              <span>Save</span>
            </>
          )}
          
          {/* Subtle glow effect for active button */}
          {displayText.trim() && downloadStatus !== 'downloading' && (
            <span className="absolute inset-0 rounded-full bg-accent/20 blur-md -z-10"></span>
          )}
        </button>
      </div>
    </div>
  );
}

export default PlayerControls;
