import React from 'react';
import { toggleDarkMode, isDarkMode } from '../../utils/theme';

interface ThemeToggleProps {
  className?: string;
}

const ThemeToggle: React.FC<ThemeToggleProps> = ({ className = '' }) => {
  const [darkMode, setDarkMode] = React.useState(false);
  
  // Sync state with document on mount and when toggled externally
  React.useEffect(() => {
    const updateState = () => {
      setDarkMode(isDarkMode());
    };
    
    // Set initial state
    updateState();
    
    // Listen for storage events (in case theme is changed in another tab)
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'theme') {
        updateState();
      }
    };
    
    window.addEventListener('storage', handleStorageChange);
    
    // Listen for system preference changes
    const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
    const handleMediaChange = () => {
      // Only update if no theme is explicitly set
      if (!localStorage.getItem('theme')) {
        updateState();
      }
    };
    
    if (mediaQuery.addEventListener) {
      mediaQuery.addEventListener('change', handleMediaChange);
    } else {
      // Fallback for older browsers
      mediaQuery.addListener(handleMediaChange);
    }
    
    return () => {
      window.removeEventListener('storage', handleStorageChange);
      if (mediaQuery.removeEventListener) {
        mediaQuery.removeEventListener('change', handleMediaChange);
      } else {
        // Fallback for older browsers
        mediaQuery.removeListener(handleMediaChange);
      }
    };
  }, []);
  
  const handleToggle = () => {
    toggleDarkMode();
    setDarkMode(isDarkMode());
  };
  
  return (
    <button
      aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
      onClick={handleToggle}
      className={`flex items-center justify-center w-8 h-8 rounded-md transition-colors ${
        darkMode 
          ? 'bg-primary/10 text-primary-foreground hover:bg-primary/20' 
          : 'bg-secondary text-foreground hover:bg-secondary/80'
      } ${className}`}
    >
      {darkMode ? (
        <SunIcon className="w-5 h-5" />
      ) : (
        <MoonIcon className="w-5 h-5" />
      )}
    </button>
  );
};

export default ThemeToggle;

// Sun icon for light mode
const SunIcon: React.FC<{ className?: string }> = ({ className = '' }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
  >
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
);

// Moon icon for dark mode
const MoonIcon: React.FC<{ className?: string }> = ({ className = '' }) => (
  <svg 
    xmlns="http://www.w3.org/2000/svg" 
    viewBox="0 0 24 24" 
    fill="none" 
    stroke="currentColor" 
    strokeWidth="2" 
    strokeLinecap="round" 
    strokeLinejoin="round" 
    className={className}
  >
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);
