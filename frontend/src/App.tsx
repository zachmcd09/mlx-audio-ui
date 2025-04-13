import React, { useEffect } from 'react';
import './App.css';
import TextInput from './components/Input/TextInput';
import PlayerControls from './components/Player/PlayerControls';
import StatusBar from './components/StatusBar/StatusBar';
import ThemeToggle from './components/ThemeToggle/ThemeToggle';
import HistoryPanel from './components/History/HistoryPanel';
import { useAudioStreamer } from './hooks/useAudioStreamer';
import { useAppStore } from './store';
import { initializeTheme } from './utils/theme';
import { 
  FaMicrophone, FaGithub, 
  FaInfoCircle, FaWaveSquare, FaCog, FaHistory, 
  FaCode, FaApple, FaCheck
} from 'react-icons/fa';

function App() {
  // Instantiate the audio streamer hook
  const audioControls = useAudioStreamer();
  const showAdvancedPanel = useAppStore((state) => state.showAdvancedPanel);
  const toggleAdvancedPanel = useAppStore((state) => state.toggleAdvancedPanel);
  
  // Initialize theme on mount
  useEffect(() => {
    initializeTheme();
  }, []);

  return (
    <div className="min-h-screen flex flex-col bg-background text-foreground">
      {/* Modern header with glass effect */}
      <header className="sticky top-0 z-10 backdrop-blur-md bg-background/85">
        <div className="border-b border-border">
          <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
            {/* Logo and title */}
            <div className="flex items-center">
              <div className="flex items-center justify-center w-10 h-10 rounded-xl mr-3 bg-gradient-to-tr from-primary to-accent shadow-md">
                <FaMicrophone className="text-xl text-primary-foreground" />
              </div>
              <div>
                <h1 className="text-lg font-bold tracking-tight">MLX Audio UI</h1>
                <p className="text-xs text-muted-foreground">Text-to-speech powered by Apple Silicon</p>
              </div>
            </div>
            
            {/* Controls */}
            <div className="flex items-center gap-2">
              {/* Settings toggle */}
              <button 
                onClick={toggleAdvancedPanel}
                className={`w-9 h-9 flex items-center justify-center rounded-full transition-colors duration-200 ${
                  showAdvancedPanel ? 'bg-primary/20 text-primary' : 'bg-muted/80 text-muted-foreground hover:bg-muted/90'
                }`}
                aria-label="Toggle advanced settings"
              >
                <FaCog className="text-sm" />
              </button>
              
              {/* About/Info tooltip */}
              <button 
                onClick={() => alert('MLX Audio UI is a real-time text-to-speech web interface powered by Apple\'s MLX framework, designed for Apple Silicon.')}
                className="w-9 h-9 flex items-center justify-center rounded-full bg-muted/80 text-muted-foreground hover:bg-muted/90 transition-colors duration-200"
                aria-label="About this application"
              >
                <FaInfoCircle className="text-sm" />
              </button>

              {/* Dark mode toggle */}
              <ThemeToggle className="w-9 h-9 rounded-full bg-muted/80 hover:bg-muted/90" />
              
              {/* GitHub link */}
              <a 
                href="https://github.com/ml-explore/mlx-audio-ui" 
                target="_blank" 
                rel="noopener noreferrer"
                className="w-9 h-9 flex items-center justify-center rounded-full bg-muted/80 text-muted-foreground hover:bg-muted/90 transition-colors duration-200"
                aria-label="GitHub repository"
              >
                <FaGithub className="text-sm" />
              </a>
            </div>
          </div>
        </div>
        
        {/* Status bar */}
        <StatusBar />
      </header>
      
      {/* Main content */}
      <main className="flex-grow py-8">
        <div className="max-w-4xl mx-auto px-4">
          {/* Hero section with animated gradient */}
          <div className="mb-8 text-center relative">
            <div className="absolute -inset-4 bg-gradient-to-r from-primary/5 via-accent/5 to-primary/5 rounded-3xl -z-10 opacity-60 blur-xl"></div>
            <h2 className="text-2xl font-bold mb-3 inline-block bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
              Real-time Text-to-Speech
            </h2>
            <p className="text-lg max-w-2xl mx-auto text-muted-foreground">
              Transform text into natural-sounding speech powered by the MLX framework.
            </p>
          </div>
          
          {/* Main UI layout with optional panel */}
          <div className="grid gap-6 grid-cols-1 lg:grid-cols-4 mb-6">
            {/* Main content area */}
            <div className={`flex flex-col gap-6 ${showAdvancedPanel ? 'lg:col-span-3' : 'lg:col-span-4'}`}>
              {/* Input area */}
              <div className="bg-card rounded-xl overflow-hidden shadow-lg border border-border">
                <TextInput />
              </div>
              
              {/* Player controls */}
              <PlayerControls controls={audioControls} />
            </div>
            
            {/* Advanced panel (conditionally shown) */}
            {showAdvancedPanel && (
              <div className="lg:col-span-1 bg-card rounded-xl border border-border p-5 shadow-lg">
                <h3 className="text-sm font-semibold mb-4 flex items-center gap-2">
                  <FaCog className="text-primary" size={14} />
                  Advanced Options
                </h3>
                
                {/* Panel sections */}
                <div className="space-y-5">
                  {/* History section */}
                  <div className="pb-4 border-b border-border/60">
                    <h4 className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-1.5">
                      <FaHistory size={10} />
                      Recent Texts
                    </h4>
                    <div className="h-[320px]">
                      <HistoryPanel />
                    </div>
                  </div>
                  
                  {/* Visualization options */}
                  <div className="pb-4 border-b border-border/60">
                    <h4 className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-1.5">
                      <FaWaveSquare size={10} />
                      Audio Visualization
                    </h4>
                    <div className="text-xs bg-muted/10 rounded-lg p-4">
                      <p className="text-muted-foreground mb-3">
                        Customize how audio is visualized during playback.
                      </p>
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span>Visualization Type:</span>
                          <span className="font-medium text-foreground">Enhanced</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Sensitivity:</span>
                          <span className="font-medium text-foreground">Automatic</span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span>Quality:</span>
                          <span className="font-medium text-foreground">High</span>
                        </div>
                      </div>
                      
                      <div className="mt-3 pt-3 border-t border-border/30">
                        <span className="block mb-2">Color Theme:</span>
                        <div className="flex gap-2">
                          <button className="w-6 h-6 rounded-full bg-gradient-to-r from-primary to-accent border border-white/20 flex items-center justify-center">
                            <FaCheck size={10} className="text-white" />
                          </button>
                          <button className="w-6 h-6 rounded-full bg-gradient-to-r from-green-500 to-blue-500 opacity-60 hover:opacity-100 transition"></button>
                          <button className="w-6 h-6 rounded-full bg-gradient-to-r from-amber-500 to-red-500 opacity-60 hover:opacity-100 transition"></button>
                          <button className="w-6 h-6 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 opacity-60 hover:opacity-100 transition"></button>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* API details */}
                  <div>
                    <h4 className="text-xs font-medium text-muted-foreground mb-3 flex items-center gap-1.5">
                      <FaCode size={10} />
                      MLX Framework
                    </h4>
                    <div className="bg-muted/30 p-3 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs">Framework</span>
                        <span className="text-xs font-medium flex items-center gap-1">
                          <FaApple size={10} className="text-muted-foreground" />
                          MLX
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-xs">Status</span>
                        <span className="text-xs font-medium text-green-500 flex items-center gap-1">
                          <span className="w-1.5 h-1.5 rounded-full bg-green-500"></span>
                          Active
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer className="py-6 mt-auto text-center border-t border-border bg-muted/20 text-muted-foreground text-sm">
        <div className="max-w-7xl mx-auto px-4">
          <p>
            Powered by{' '}
            <a 
              href="https://github.com/ml-explore/mlx" 
              target="_blank" 
              rel="noopener noreferrer"
              className="font-medium underline underline-offset-2 decoration-1 text-primary hover:text-primary-hover transition-colors"
            >
              MLX
            </a>{' '}
            on Apple Silicon
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
