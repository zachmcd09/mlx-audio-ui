import React, { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../../store';
import { 
  FaCheck, FaPlay, FaPause, FaStop, FaExclamationTriangle, 
  FaTimes, FaMusic, FaBroadcastTower, FaMicrophone, 
  FaVolumeUp, FaBell, FaInfoCircle, FaCog, FaCheckCircle
} from 'react-icons/fa';

interface ToastMessage {
  id: number;
  type: 'success' | 'info' | 'warning' | 'error';
  message: string;
}

function StatusBar() {
  const playbackState = useAppStore((state) => state.playbackState);
  const errorMessage = useAppStore((state) => state.errorMessage);
  const { _clearError } = useAppStore.getState();
  const statusBarRef = useRef<HTMLDivElement>(null);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);
  const [prevPlaybackState, setPrevPlaybackState] = useState<string>(playbackState);
  const [isStatusExpanded, setIsStatusExpanded] = useState(false);

  // Define status configurations with icons, classes and messages
  const statusConfig = {
    Idle: {
      message: 'Ready',
      icon: <FaCheck className="text-green-500" />,
      bgClass: 'bg-background',
      textClass: 'text-foreground',
      accentClass: 'bg-gradient-to-r from-green-400 to-green-500',
      pulseClass: ''
    },
    Buffering: {
      message: 'Generating audio...',
      icon: <FaBroadcastTower className="text-accent" />,
      bgClass: 'bg-background',
      textClass: 'text-foreground',
      accentClass: 'bg-gradient-to-r from-primary via-accent to-primary bg-[length:200%_100%] animate-gradient',
      pulseClass: 'animate-pulse'
    },
    Playing: {
      message: 'Playing audio',
      icon: <FaPlay className="text-primary" />,
      bgClass: 'bg-background',
      textClass: 'text-foreground',
      accentClass: 'bg-gradient-to-r from-primary to-accent',
      pulseClass: ''
    },
    Paused: {
      message: 'Playback paused',
      icon: <FaPause className="text-amber-500" />,
      bgClass: 'bg-background',
      textClass: 'text-foreground',
      accentClass: 'bg-amber-500',
      pulseClass: ''
    },
    Stopped: {
      message: 'Playback stopped',
      icon: <FaStop className="text-muted-foreground" />,
      bgClass: 'bg-background',
      textClass: 'text-foreground',
      accentClass: 'bg-muted-foreground',
      pulseClass: ''
    },
    Error: {
      message: errorMessage ? `Error: ${errorMessage}` : 'Error occurred',
      icon: <FaExclamationTriangle className="text-destructive" />,
      bgClass: 'bg-background',
      textClass: 'text-destructive',
      accentClass: 'bg-destructive',
      pulseClass: 'animate-pulse'
    }
  };

  // Get config for current state
  const currentStatus = statusConfig[playbackState as keyof typeof statusConfig] || statusConfig.Idle;
  
  // Show toast notification when states change
  useEffect(() => {
    if (prevPlaybackState !== playbackState) {
      // Only show toasts for important state changes
      if (playbackState === 'Playing') {
        showToast('success', 'Audio playback started');
      } else if (playbackState === 'Buffering') {
        showToast('info', 'Generating audio stream...');
      } else if (playbackState === 'Error') {
        showToast('error', `Error: ${errorMessage || 'An error occurred'}`);
      }
      
      setPrevPlaybackState(playbackState);
    }
    
    if (statusBarRef.current) {
      // Add flash effect
      statusBarRef.current.classList.add('status-change');
      
      const timer = setTimeout(() => {
        if (statusBarRef.current) {
          statusBarRef.current.classList.remove('status-change');
        }
      }, 500);
      
      return () => clearTimeout(timer);
    }
  }, [playbackState, errorMessage, prevPlaybackState]);
  
  // Function to show toast message
  const showToast = (type: 'success' | 'info' | 'warning' | 'error', message: string) => {
    const id = Date.now();
    setToasts((prev) => [...prev, { id, type, message }]);
    
    // Auto-remove toast after 5 seconds
    setTimeout(() => {
      setToasts((prev) => prev.filter((toast) => toast.id !== id));
    }, 5000);
  };
  
  // Function to toggle expanded status view
  const toggleStatusExpanded = () => {
    setIsStatusExpanded(!isStatusExpanded);
  };

  return (
    <div className="relative border-b border-border/60 bg-background/95 backdrop-blur-sm z-10 shadow-sm">
      {/* Animation keyframes */}
      <style>
        {`
          @keyframes statusChange {
            0% { background-color: rgba(var(--primary-rgb), 0.05); }
            100% { background-color: transparent; }
          }
          
          .status-change {
            animation: statusChange 500ms ease-out;
          }
          
          @keyframes gradient {
            0% { background-position: 0% 50% }
            50% { background-position: 100% 50% }
            100% { background-position: 0% 50% }
          }
          
          .animate-gradient {
            animation: gradient 6s ease infinite;
          }
          
          @keyframes slideInRight {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
          }
          
          @keyframes slideOutRight {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
          }
          
          @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-4px); }
            60% { transform: translateY(-2px); }
          }
          
          .slide-in-right {
            animation: slideInRight 0.3s forwards;
          }
          
          .slide-out-right {
            animation: slideOutRight 0.3s forwards;
          }
          
          .bounce {
            animation: bounce 1s ease;
          }
        `}
      </style>
      
      {/* Accent strip at top */}
      <div className={`absolute top-0 left-0 right-0 h-0.5 transition-all duration-300 ${currentStatus.accentClass} ${currentStatus.pulseClass}`} />
      
      {/* Toast notifications container */}
      <div className="fixed top-16 right-4 z-50 flex flex-col gap-2 items-end">
        {toasts.map((toast) => (
          <div 
            key={toast.id}
            className={`px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 slide-in-right backdrop-blur-sm border ${
              toast.type === 'error' 
                ? 'bg-destructive/90 text-destructive-foreground border-destructive/50' 
                : toast.type === 'warning'
                  ? 'bg-amber-500/90 text-white border-amber-600/50' 
                  : toast.type === 'success'
                    ? 'bg-green-500/90 text-white border-green-600/50'
                    : 'bg-primary/90 text-primary-foreground border-primary/50'
            }`}
            style={{maxWidth: '320px'}}
          >
            <span className="flex-shrink-0">
              {toast.type === 'error' && <FaExclamationTriangle size={16} />}
              {toast.type === 'warning' && <FaExclamationTriangle size={16} />}
              {toast.type === 'success' && <FaCheckCircle size={16} />}
              {toast.type === 'info' && <FaInfoCircle size={16} />}
            </span>
            <span className="text-sm font-medium">{toast.message}</span>
          </div>
        ))}
      </div>
      
      {/* Status bar content */}
      <div 
        ref={statusBarRef}
        className={`py-2.5 px-5 flex items-center justify-between ${currentStatus.bgClass} ${currentStatus.textClass} transition-all duration-300 ${errorMessage ? 'shadow-md' : ''}`}
        role="status"
        aria-live={errorMessage ? 'assertive' : 'polite'}
      >
        {/* App branding */}
        <div className="flex items-center gap-2">
          <div className="flex items-center justify-center w-6 h-6 rounded-full bg-primary/10">
            <FaMusic className="text-primary" size={12} />
          </div>
          <span className="font-semibold text-sm tracking-tight">MLX Audio</span>
          
          {/* Only show extra info on Playing/Buffering states */}
          {(playbackState === 'Playing' || playbackState === 'Buffering') && (
            <div className="text-xs font-medium ml-2 px-1.5 py-0.5 rounded-full bg-primary/10 text-primary flex items-center">
              <span className="inline-block w-1.5 h-1.5 rounded-full bg-primary mr-1.5 animate-[pulse_2s_ease-in-out_infinite]"></span>
              Active
            </div>
          )}
        </div>
        
        {/* Status indicator - centered with hover expansion */}
        <div 
          className="absolute left-1/2 -translate-x-1/2 flex items-center cursor-pointer"
          onClick={toggleStatusExpanded}
        >
          <div 
            className={`flex items-center gap-2 py-1.5 px-4 rounded-full ${
              playbackState === 'Error' 
                ? 'bg-destructive/10 text-destructive' 
                : playbackState === 'Playing'
                  ? 'bg-primary/10 text-primary'
                  : playbackState === 'Buffering'
                    ? 'bg-accent/10 text-accent animate-[pulse_2s_ease-in-out_infinite]'
                    : 'bg-muted/20 text-muted-foreground'
            } transition-all duration-300 border border-transparent hover:border-border/30 hover:shadow-sm`}
          >
            <span className="flex items-center justify-center w-4 h-4">
              {currentStatus.icon}
            </span>
            
            <span className="font-medium text-xs tracking-tight whitespace-nowrap">
              {currentStatus.message}
            </span>
            
            {/* Status details */}
            {isStatusExpanded && (
              <div className="ml-2 text-xs text-muted-foreground/80 overflow-hidden max-w-[120px] md:max-w-[200px] transition-all duration-200">
                {playbackState === 'Playing' && (
                  <span className="flex items-center">
                    <FaVolumeUp size={10} className="mr-1" />
                    Streaming audio
                  </span>
                )}
                {playbackState === 'Buffering' && (
                  <span className="flex items-center">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-accent mr-1.5 animate-[pulse_2s_ease-in-out_infinite]"></span>
                    Processing
                  </span>
                )}
                {playbackState === 'Paused' && (
                  <span className="flex items-center">
                    Click to resume
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
        
        {/* Right action area */}
        <div className="flex justify-end gap-2">
          {/* Notifications indicator */}
          <div 
            className={`group relative py-1.5 px-2 rounded-full hover:bg-muted/20 cursor-pointer flex items-center transition-all duration-150 ${
              toasts.length > 0 ? 'text-primary' : 'text-muted-foreground'
            }`}
            onClick={() => toasts.length > 0 && setToasts([])}
          >
            <FaBell size={14} />
            {toasts.length > 0 && (
              <span className="absolute -top-1 -right-1 w-4 h-4 flex items-center justify-center bg-primary text-[10px] text-primary-foreground rounded-full font-bold">
                {toasts.length}
              </span>
            )}
            
            {/* Tooltip */}
            <div className="absolute hidden group-hover:block bottom-full right-0 mb-2 p-2 bg-popover text-popover-foreground rounded-lg shadow-lg text-xs font-medium whitespace-nowrap">
              {toasts.length > 0 ? 'Click to clear notifications' : 'No notifications'}
            </div>
          </div>
          
          {errorMessage ? (
            <button
              onClick={_clearError}
              className="text-xs font-medium px-3 py-1.5 rounded-full text-destructive border border-destructive/20 hover:bg-destructive/10 transition-colors duration-150 flex items-center gap-1.5"
              aria-label="Clear error message"
            >
              <FaTimes size={10} />
              <span>Dismiss</span>
            </button>
          ) : (
            <div className="text-xs font-medium px-3 py-1.5 rounded-full bg-muted/10 text-muted-foreground flex items-center gap-1.5 hover:bg-muted/20 transition-colors duration-150">
              <FaMicrophone size={10} className="text-primary/70" />
              <span>MLX-powered</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default StatusBar;
