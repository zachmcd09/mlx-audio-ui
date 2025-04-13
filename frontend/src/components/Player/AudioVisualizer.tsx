import React, { useRef, useEffect, useState } from 'react';
import { useAppStore } from '../../store';
import { cx } from '../../utils/theme';

// Define visualization modes
export type VisualizationMode = 'frequency' | 'waveform' | 'spectrum';

interface AudioVisualizerProps {
  audioContext?: AudioContext | null;
  sourceNode?: AudioBufferSourceNode | null;
  mode?: VisualizationMode;
  showControls?: boolean;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ 
  audioContext, 
  sourceNode, 
  mode: initialMode = 'frequency',
  showControls = false 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const frequencyDataRef = useRef<Uint8Array | null>(null);
  const timeDataRef = useRef<Uint8Array | null>(null);
  
  const playbackState = useAppStore((state) => state.playbackState);
  
  // State for visualization mode and activity
  const [mode, setMode] = useState<VisualizationMode>(initialMode);
  const [isActive, setIsActive] = useState(false);
  const [sensitivity, setSensitivity] = useState<number>(0.8); // 0.0 to 1.0
  
  // Create and setup analyser node if we have audio context and source node
  useEffect(() => {
    if (!audioContext || !sourceNode) {
      setIsActive(false);
      return;
    }
    
    // Create or reuse analyser node
    if (!analyserRef.current) {
      analyserRef.current = audioContext.createAnalyser();
      analyserRef.current.fftSize = 2048; // Larger size for better resolution especially for waveform
      analyserRef.current.smoothingTimeConstant = 0.8; // Adjust smoothing for visualizations
      
      const frequencyBufferLength = analyserRef.current.frequencyBinCount;
      frequencyDataRef.current = new Uint8Array(frequencyBufferLength);
      timeDataRef.current = new Uint8Array(frequencyBufferLength);
    }
    
    // Connect source node to analyser node
    try {
      sourceNode.connect(analyserRef.current);
      // Note: Don't connect analyser to destination, that's handled by the hook
      setIsActive(true);
    } catch (error) {
      console.error('Failed to connect audio nodes:', error);
      setIsActive(false);
    }
    
    return () => {
      // Clean up connections when props change
      try {
        if (sourceNode && analyserRef.current) {
          sourceNode.disconnect(analyserRef.current);
        }
      } catch (e) {
        // Ignore disconnection errors (nodes may already be gone)
      }
    };
  }, [audioContext, sourceNode]);
  
  // Animation loop for visualizer with support for different modes
  useEffect(() => {
    // Cleanup function for animation frame
    const cleanupAnimation = () => {
      if (animationRef.current !== null) {
        cancelAnimationFrame(animationRef.current);
        animationRef.current = null;
      }
    };
    
    // Visualization is active when playing and we have an analyser
    const shouldVisualize = 
      playbackState === 'Playing' && 
      analyserRef.current && 
      frequencyDataRef.current && 
      timeDataRef.current;
    
    // Main draw function - calls appropriate visualization based on mode
    const draw = () => {
      const canvas = canvasRef.current;
      const analyser = analyserRef.current;
      
      if (!canvas || !analyser) return;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      
      // Get canvas dimensions, accounting for device pixel ratio
      const width = canvas.width;
      const height = canvas.height;
      
      // Clear canvas
      ctx.clearRect(0, 0, width, height);
      
      if (shouldVisualize) {
        switch (mode) {
          case 'waveform':
            drawWaveform(ctx, analyser, width, height);
            break;
          case 'spectrum':
            drawSpectrum(ctx, analyser, width, height);
            break;
          case 'frequency':
          default:
            drawFrequencyBars(ctx, analyser, width, height);
            break;
        }
      } else {
        // Draw idle/static visualization when not playing
        drawIdleVisualization(ctx, width, height);
      }
      
      // Continue the animation
      animationRef.current = requestAnimationFrame(draw);
    };

    // Frequency bars visualization
    const drawFrequencyBars = (
      ctx: CanvasRenderingContext2D, 
      analyser: AnalyserNode, 
      width: number, 
      height: number
    ) => {
      const canvas = canvasRef.current; // Get canvas ref
      const dataArray = frequencyDataRef.current;
      if (!canvas || !dataArray) return; // Add canvas check

      // Get computed styles ONCE
      const computedStyle = getComputedStyle(canvas);
      const primaryHsl = computedStyle.getPropertyValue('--primary').trim(); // e.g., "210 40% 96%"
      const accentHsl = computedStyle.getPropertyValue('--accent').trim();   // e.g., "48 96% 53%"

      // Construct valid CSS color strings
      const primaryColor = `hsl(${primaryHsl})`; // "hsl(210 40% 96%)"
      const accentColor = `hsl(${accentHsl})`;   // "hsl(48 96% 53%)"
      
      // Get frequency data
      analyser.getByteFrequencyData(dataArray);
      
      // Draw bars
      const barCount = Math.min(dataArray.length, 128); // Limit for performance
      const barWidth = width / barCount * 0.8; // Slightly thinner for spacing
      
      for (let i = 0; i < barCount; i++) {
        // Apply sensitivity multiplier
        const amplitudeValue = dataArray[i] * sensitivity;
        
        // Calculate bar height based on frequency data (0-255)
        const barHeight = (amplitudeValue / 255) * height * 0.85;
        
        // Calculate position
        const x = i * (width / barCount);
        const y = height - barHeight;
        
        // Create gradient for each bar
        const gradient = ctx.createLinearGradient(0, y, 0, height);
        
        // Use the computed colors - CSS variables already handle dark/light mode via index.css
        gradient.addColorStop(0, primaryColor); 
        gradient.addColorStop(1, accentColor);
        
        ctx.fillStyle = gradient;
        ctx.fillRect(x, y, barWidth, barHeight);
      }
    };
    
    // Waveform visualization (time domain)
    const drawWaveform = (
      ctx: CanvasRenderingContext2D, 
      analyser: AnalyserNode, 
      width: number, 
      height: number
    ) => {
      const canvas = canvasRef.current; // Get canvas ref
      const dataArray = timeDataRef.current;
      if (!canvas || !dataArray) return; // Add canvas check

      // Get computed styles ONCE
      const computedStyle = getComputedStyle(canvas);
      const primaryHsl = computedStyle.getPropertyValue('--primary').trim();
      const accentHsl = computedStyle.getPropertyValue('--accent').trim();

      // Construct valid CSS color strings
      const primaryColor = `hsl(${primaryHsl})`;
      const primaryColorAlpha = `hsla(${primaryHsl}, 0.1)`; // For fill
      const accentColorAlpha = `hsla(${accentHsl}, 0.1)`;  // For fill
      
      // Get time domain data
      analyser.getByteTimeDomainData(dataArray);
      
      // Draw waveform
      ctx.lineWidth = 2;
      ctx.strokeStyle = primaryColor; // Use computed color
      
      const sliceWidth = width / dataArray.length;
      
      ctx.beginPath();
      
      // Start at left side
      let x = 0;
      
      for (let i = 0; i < dataArray.length; i++) {
        // normalize the amplitude value from 0-255 to 0-1, then scale to canvas height
        // Apply sensitivity to make wave more pronounced
        const normalizedValue = (dataArray[i] / 128.0 - 1.0);
        const scaledValue = normalizedValue * sensitivity;
        const y = height / 2 + scaledValue * (height / 2 * 0.9);
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
        
        x += sliceWidth;
      }
      
      // Add gradient for the waveform
      ctx.lineTo(width, height / 2);
      ctx.stroke();
      
      // For aesthetics, add a subtle fill beneath the line
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      // Use computed colors with alpha
      gradient.addColorStop(0.4, primaryColorAlpha); 
      gradient.addColorStop(0.6, accentColorAlpha);
      
      // Create a closed path for filling
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.fillStyle = gradient;
      ctx.fill();
    };
    
    // Spectrum visualization (stylized frequency visualization)
    const drawSpectrum = (
      ctx: CanvasRenderingContext2D, 
      analyser: AnalyserNode, 
      width: number, 
      height: number
    ) => {
      const canvas = canvasRef.current; // Get canvas ref
      const dataArray = frequencyDataRef.current;
      if (!canvas || !dataArray) return; // Add canvas check

      // Get computed styles ONCE
      const computedStyle = getComputedStyle(canvas);
      const primaryHsl = computedStyle.getPropertyValue('--primary').trim();
      const accentHsl = computedStyle.getPropertyValue('--accent').trim();

      // Construct valid CSS color strings with alpha
      const primaryColorAlpha08 = `hsla(${primaryHsl}, 0.8)`;
      const accentColorAlpha02 = `hsla(${accentHsl}, 0.2)`;
      const primaryColorAlpha09 = `hsla(${primaryHsl}, 0.9)`;
      
      // Get frequency data
      analyser.getByteFrequencyData(dataArray);
      
      // Create gradient
      const gradient = ctx.createLinearGradient(0, 0, 0, height);
      gradient.addColorStop(0, primaryColorAlpha08);
      gradient.addColorStop(1, accentColorAlpha02);
      
      ctx.fillStyle = gradient;
      ctx.strokeStyle = primaryColorAlpha09; // Use computed color
      ctx.lineWidth = 2;
      
      // Draw a spectrum curve connecting frequency points
      ctx.beginPath();
      
      // Start at bottom left
      ctx.moveTo(0, height);
      
      // Use a subset of the data for better visuals
      const dataPoints = Math.min(dataArray.length, 256);
      const pointSpacing = width / dataPoints;
      
      // Draw curve through frequency points
      for (let i = 0; i < dataPoints; i++) {
        // Apply sensitivity to the amplitude
        const amplitudeValue = dataArray[i] * sensitivity;
        
        // Calculate the Y position based on the frequency amplitude
        const y = height - (amplitudeValue / 255) * height * 0.9;
        const x = i * pointSpacing;
        
        // For smoother spectrum, use quadratic curves
        if (i === 0) {
          ctx.lineTo(x, y);
        } else if (i % 2 === 0) { // Using every other point for smoother curve
          const prevX = (i - 1) * pointSpacing;
          const prevY = height - (dataArray[i-1] / 255) * height * 0.9;
          ctx.quadraticCurveTo(prevX, prevY, x, y);
        }
      }
      
      // Complete the path to fill
      ctx.lineTo(width, height);
      ctx.closePath();
      
      // Fill the spectrum area
      ctx.fill();
      
      // Add a subtle stroke along the top of the spectrum
      ctx.beginPath();
      for (let i = 0; i < dataPoints; i++) {
        const amplitudeValue = dataArray[i] * sensitivity;
        const y = height - (amplitudeValue / 255) * height * 0.9;
        const x = i * pointSpacing;
        
        if (i === 0) {
          ctx.moveTo(x, y);
        } else if (i % 2 === 0) {
          const prevX = (i - 1) * pointSpacing;
          const prevY = height - (dataArray[i-1] / 255) * height * 0.9;
          ctx.quadraticCurveTo(prevX, prevY, x, y);
        }
      }
      ctx.stroke();
    };
    
      // Function to draw an idle state visualization
    const drawIdleVisualization = (ctx: CanvasRenderingContext2D, width: number, height: number) => {
      const canvas = canvasRef.current; // Get canvas ref
      if (!canvas) return; // Add canvas check

      // Get computed styles ONCE
      const computedStyle = getComputedStyle(canvas);
      const primaryHsl = computedStyle.getPropertyValue('--primary').trim();
      const accentHsl = computedStyle.getPropertyValue('--accent').trim();
      const destructiveHsl = computedStyle.getPropertyValue('--destructive').trim();
      // Note: Paused state uses a hardcoded amber color, which is fine.

      // Construct color strings
      let lineColor: string, fillColor: string;
      
      if (playbackState === 'Buffering') {
        lineColor = `hsla(${accentHsl}, 0.8)`;
        fillColor = `hsla(${accentHsl}, 0.1)`; // Adjusted alpha for consistency
      } else if (playbackState === 'Paused') {
        // Keep hardcoded amber for paused state
        lineColor = 'hsla(38, 92%, 50%, 0.8)';
        fillColor = 'hsla(38, 92%, 50%, 0.1)'; // Adjusted alpha
      } else if (playbackState === 'Error') {
        lineColor = `hsla(${destructiveHsl}, 0.8)`;
        fillColor = `hsla(${destructiveHsl}, 0.1)`; // Adjusted alpha
      } else {
        // Idle or other states
        lineColor = `hsla(${primaryHsl}, 0.6)`;
        fillColor = `hsla(${primaryHsl}, 0.05)`;
      }
      
      // Draw a stylized waveform placeholder
      // Number of points in the sine wave
      const points = 100;
      
      // Create path
      ctx.beginPath();
      ctx.moveTo(0, height / 2);
      
      // Generate the animation offset based on time for subtle movement
      const now = Date.now() / 1000;
      const animationOffset = Math.sin(now) * 5;
      
      // Draw the sine wave with varying amplitude
      for (let i = 0; i <= points; i++) {
        const x = (width / points) * i;
        
        // Base amplitude varies based on playback state
        let amplitude;
        if (playbackState === 'Buffering') {
          // Pulsing effect for buffering
          amplitude = Math.sin(now * 3) * 10 + 25;
        } else if (playbackState === 'Paused') {
          // Static, medium amplitude
          amplitude = 20;
        } else {
          // Low, gentle waves for idle
          amplitude = 15;
        }
        
        // Create a more interesting waveform with multiple frequencies
        const y = height / 2 + 
          amplitude * Math.sin((i / points) * Math.PI * 6 + animationOffset) + 
          (amplitude / 3) * Math.sin((i / points) * Math.PI * 12 + animationOffset * 2);
        
        ctx.lineTo(x, y);
      }
      
      // Complete the path to the bottom-right corner, bottom-left corner, and back to start
      ctx.lineTo(width, height / 2);
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      
      // Fill and stroke the path
      ctx.fillStyle = fillColor;
      ctx.fill();
      
      ctx.strokeStyle = lineColor;
      ctx.lineWidth = 2;
      ctx.stroke();
    };
    
    // Start animation
    animationRef.current = requestAnimationFrame(draw);
    
    // Clean up on unmount or when dependencies change
    return cleanupAnimation;
  }, [playbackState, isActive, mode, sensitivity]);
  
  // Helper function to update sensitivity
  const handleSensitivityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSensitivity(parseFloat(e.target.value));
  };
  
  // Helper function to change visualization mode
  const handleModeChange = (newMode: VisualizationMode) => {
    setMode(newMode);
  };
  
  return (
    <div className="rounded-lg overflow-hidden bg-secondary/5 border border-border/40 shadow-inner">
      <canvas 
        ref={canvasRef}
        className="w-full h-24 block" 
        width={600} // Base resolution, will be scaled by CSS
        height={150} 
      />
      
      {/* Visualization Controls (conditionally rendered) */}
      {showControls && (
        <div className="p-3 border-t border-border/40 bg-background/50">
          <div className="flex flex-wrap items-center gap-3 justify-between">
            {/* Mode Selection */}
            <div className="flex gap-1">
              <button 
                onClick={() => handleModeChange('frequency')}
                className={cx(
                  "px-2 py-1 text-xs rounded-md transition",
                  mode === 'frequency' 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-secondary/80 text-secondary-foreground hover:bg-secondary"
                )}
              >
                Bars
              </button>
              <button 
                onClick={() => handleModeChange('waveform')}
                className={cx(
                  "px-2 py-1 text-xs rounded-md transition",
                  mode === 'waveform' 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-secondary/80 text-secondary-foreground hover:bg-secondary"
                )}
              >
                Waveform
              </button>
              <button 
                onClick={() => handleModeChange('spectrum')}
                className={cx(
                  "px-2 py-1 text-xs rounded-md transition",
                  mode === 'spectrum' 
                    ? "bg-primary text-primary-foreground" 
                    : "bg-secondary/80 text-secondary-foreground hover:bg-secondary"
                )}
              >
                Spectrum
              </button>
            </div>
            
            {/* Sensitivity Slider */}
            <div className="flex items-center gap-2">
              <label className="text-xs text-muted-foreground">Gain:</label>
              <input 
                type="range" 
                min="0.2" 
                max="1.5" 
                step="0.1"
                value={sensitivity}
                onChange={handleSensitivityChange}
                className="w-20 h-2 bg-secondary/70 rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3 [&::-webkit-slider-thumb]:w-3 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioVisualizer;
