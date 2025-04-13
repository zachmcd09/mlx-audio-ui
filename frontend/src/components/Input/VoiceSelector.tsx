import React, { useState, useEffect } from 'react';
import { useAppStore } from '../../store';
import { 
  FaVolumeUp, FaRobot, FaUserAlt, FaMicrophone,
  FaWaveSquare, FaSpinner, FaExclamationCircle, FaRedo
} from 'react-icons/fa';

// API base URL
const API_BASE_URL = 'http://127.0.0.1:5000';

export interface VoiceModel {
  id: string;
  name: string;
  description: string;
  icon?: JSX.Element;
}

interface VoiceSelectorProps {
  disabled?: boolean;
  className?: string;
}

const VoiceSelector: React.FC<VoiceSelectorProps> = ({ 
  disabled = false,
  className = ''
}) => {
  const voice = useAppStore((state) => state.voice);
  const setVoice = useAppStore((state) => state.setVoice);
  
  const [availableVoices, setAvailableVoices] = useState<VoiceModel[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Function to fetch voices from the API
  const fetchVoices = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
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
      
      // Add icons based on ID or name
      const voicesWithIcons = data.map((v: any) => ({
        ...v,
        icon: v.id === 'kokoro' ? <FaWaveSquare /> : 
              v.id === 'bark' ? <FaRobot /> : 
              <FaMicrophone />
      }));
      
      setAvailableVoices(voicesWithIcons);
      
      // Set default voice if not already set
      if (!voice && voicesWithIcons.length > 0) {
        setVoice(voicesWithIcons[0].id);
      }
    } catch (error) {
      console.error("Failed to fetch voices:", error);
      setError(error instanceof Error ? error.message : "Unknown error");
      
      // Set fallback error state
      setAvailableVoices([{ 
        id: 'default_error', 
        name: 'Error Loading Voices', 
        description: 'The server failed to initialize TTS models. Check server logs for details.' 
      }]);
    } finally {
      setIsLoading(false);
    }
  };
  
  // Fetch voices on component mount
  useEffect(() => {
    fetchVoices();
  }, []);
  
  // Get the selected voice
  const selectedVoice = availableVoices.find(v => v.id === voice) || 
                       availableVoices[0] || 
                       { id: '', name: 'Loading...', description: '' };
  
  return (
    <div className={`${className} space-y-2`}>
      {/* Voice dropdown */}
      <div className="relative">
        <select
          value={voice || ''}
          onChange={(e) => setVoice(e.target.value)}
          disabled={disabled || isLoading || availableVoices.length === 0}
          className="block w-full p-2.5 pl-4 rounded-lg border border-input bg-background text-foreground text-sm appearance-none pr-10 shadow-sm transition-all duration-200 focus:border-primary focus:ring-2 focus:ring-primary/15 disabled:opacity-50 disabled:cursor-not-allowed"
          style={{
            backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20' stroke='%236b7280'%3E%3Cpath stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'%3E%3C/path%3E%3C/svg%3E")`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: 'right 0.75rem center',
            backgroundSize: '1rem'
          }}
        >
          {isLoading ? (
            <option value="" disabled>Loading voices...</option>
          ) : availableVoices.length === 0 ? (
            <option value="" disabled>No voices available</option>
          ) : (
            availableVoices.map(v => (
              <option key={v.id} value={v.id}>{v.name}</option>
            ))
          )}
        </select>
        
        {/* Left accent bar */}
        <div className="absolute left-0 top-0 h-full w-0.5 bg-primary rounded-l-lg opacity-70"></div>
        
        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute right-9 top-1/2 transform -translate-y-1/2 text-muted-foreground animate-spin">
            <FaSpinner size={14} />
          </div>
        )}
      </div>
      
      {/* Voice description */}
      <p className="text-xs text-muted-foreground leading-relaxed">
        {selectedVoice.description || 'Select a voice for audio synthesis.'}
      </p>
      
      {/* Error message */}
      {error && (
        <div className="bg-destructive/10 border border-destructive/30 text-destructive rounded-md p-2 text-xs">
          <div className="flex items-center">
            <FaExclamationCircle className="mr-1.5" />
            <span className="font-medium">Failed to load voices</span>
          </div>
          <p className="mt-1">{error}</p>
          <button 
            onClick={fetchVoices}
            className="mt-1.5 text-xs flex items-center hover:underline"
            disabled={isLoading}
          >
            <FaRedo size={10} className="mr-1" />
            {isLoading ? 'Retrying...' : 'Retry loading voices'}
          </button>
        </div>
      )}
    </div>
  );
};

export default VoiceSelector;
