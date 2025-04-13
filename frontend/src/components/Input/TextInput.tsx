import React, { useState, useEffect, useRef } from 'react';
import { useAppStore } from '../../store';
import HighlightableText from './HighlightableText';
import { 
  FaEdit, FaVolumeUp, FaMagic, FaTrash, FaCopy, 
  FaInfoCircle, FaCheck, FaFont, FaClipboard,
  FaRandom, FaPen, FaLightbulb, FaRegLightbulb,
  FaBold, FaItalic, FaUnderline, FaHeading, FaListUl,
  FaQuoteRight, FaKeyboard
} from 'react-icons/fa';

// Character limit for optimal performance
const CHAR_LIMIT = 2000;
const WARN_THRESHOLD = 0.8; // Show warning at 80% of limit

function TextInput() {
  const inputText = useAppStore((state) => state.inputText);
  const setInputText = useAppStore((state) => state.setInputText);
  const playbackState = useAppStore((state) => state.playbackState);
  const [charCount, setCharCount] = useState(0);
  const [isFocused, setIsFocused] = useState(false);
  const [showCopiedToast, setShowCopiedToast] = useState(false);
  const [showTip, setShowTip] = useState(false);
  const [showFormatting, setShowFormatting] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  
  // Determine if we should show read-only mode with highlighting
  const isReadOnly = playbackState === 'Playing' || playbackState === 'Buffering';
  
  // Update char count when input text changes
  useEffect(() => {
    setCharCount(inputText.length);
    
    // Auto-resize the textarea based on content
    if (textareaRef.current) {
      // Reset height to auto to correctly calculate the new height
      textareaRef.current.style.height = 'auto';
      // Set the height to scrollHeight to fit all content
      textareaRef.current.style.height = `${Math.max(180, textareaRef.current.scrollHeight)}px`;
    }
  }, [inputText]);

  // Handle copy toast auto-dismissal
  useEffect(() => {
    if (showCopiedToast) {
      const timer = setTimeout(() => setShowCopiedToast(false), 2000);
      return () => clearTimeout(timer);
    }
  }, [showCopiedToast]);

  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    const value = event.target.value;
    if (value.length <= CHAR_LIMIT) {
      setInputText(value);
    }
  };

  const handleFocus = () => {
    setIsFocused(true);
  };

  const handleBlur = () => {
    setIsFocused(false);
  };

  const clearText = () => {
    setInputText('');
    // Focus the textarea after clearing
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const copyText = () => {
    if (inputText) {
      navigator.clipboard.writeText(inputText)
        .then(() => {
          setShowCopiedToast(true);
        })
        .catch(err => {
          console.error('Failed to copy text: ', err);
        });
    }
  };

  // Calculate char limit percentage for the progress bar
  const charPercentage = Math.min((charCount / CHAR_LIMIT) * 100, 100);
  const isApproachingLimit = charCount > CHAR_LIMIT * WARN_THRESHOLD;
  const isAtLimit = charCount >= CHAR_LIMIT;

  // Get color based on character count percentage
  const getProgressBarClass = () => {
    if (isAtLimit) return 'bg-destructive';
    if (isApproachingLimit) return 'bg-gradient-to-r from-amber-400 to-destructive';
    return 'bg-gradient-to-r from-primary to-accent';
  };
  
  const getCountClass = () => {
    if (isAtLimit) return 'text-destructive';
    if (isApproachingLimit) return 'text-amber-500';
    if (charCount > 0) return 'text-primary';
    return 'text-muted-foreground';
  };

  // Sample texts with varying lengths and content
  const sampleTexts = [
    "Hello! This is a sample text to demonstrate the text-to-speech capabilities of MLX Audio UI. This interface is powered by Apple's MLX framework and provides real-time streaming audio playback.",
    "Imagine being able to convert any text into natural-sounding speech in real-time. That's exactly what MLX Audio UI does, leveraging the power of Apple Silicon to deliver high-quality audio synthesis directly in your browser.",
    "The quick brown fox jumps over the lazy dog. This pangram contains all the letters of the English alphabet.",
    "In a world where digital communication dominates, the ability to convert text to lifelike speech opens up new possibilities for accessibility and user engagement."
  ];

  // Tips to show randomly
  const tips = [
    "Use proper punctuation for natural-sounding pauses.",
    "Adding commas can improve the rhythm of synthesized speech.",
    "Question marks will add appropriate intonation to questions.",
    "Try different voices to find the one that best suits your content.",
    "Speed controls let you adjust playback to your preference.",
    "Longer texts are automatically split into manageable chunks."
  ];

  // Select a random tip
  const getRandomTip = () => {
    return tips[Math.floor(Math.random() * tips.length)];
  };

  // Show a random tip
  const toggleTip = () => {
    setShowTip(!showTip);
  };

  // Add text formatting functions
  const formatText = (formatType: string) => {
    if (!textareaRef.current) return;
    
    const textarea = textareaRef.current;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const selectedText = inputText.substring(start, end);
    
    let formattedText = '';
    let cursorOffset = 0;
    
    switch (formatType) {
      case 'bold':
        formattedText = `**${selectedText}**`;
        cursorOffset = 2;
        break;
      case 'italic':
        formattedText = `_${selectedText}_`;
        cursorOffset = 1;
        break;
      case 'heading':
        formattedText = `# ${selectedText}`;
        cursorOffset = 2;
        break;
      case 'quote':
        formattedText = `> ${selectedText}`;
        cursorOffset = 2;
        break;
      case 'list':
        formattedText = `- ${selectedText}`;
        cursorOffset = 2;
        break;
      default:
        return;
    }
    
    // Insert the formatted text
    const newText = inputText.substring(0, start) + formattedText + inputText.substring(end);
    
    // Only update if within character limit
    if (newText.length <= CHAR_LIMIT) {
      setInputText(newText);
      
      // Set cursor position after the formatting
      setTimeout(() => {
        textarea.focus();
        const newPosition = start + formattedText.length;
        textarea.setSelectionRange(newPosition, newPosition);
      }, 0);
    }
  };
  
  const toggleFormatting = () => {
    setShowFormatting(!showFormatting);
  };

  const addSampleText = () => {
    // Select a random sample text
    const randomIndex = Math.floor(Math.random() * sampleTexts.length);
    const sampleText = sampleTexts[randomIndex];
    
    // Only add if it doesn't exceed character limit
    if (sampleText.length <= CHAR_LIMIT) {
      setInputText(sampleText);
      // Focus and move cursor to the end
      if (textareaRef.current) {
        textareaRef.current.focus();
        textareaRef.current.setSelectionRange(sampleText.length, sampleText.length);
      }
    }
  };

  return (
    <div className="rounded-xl bg-card text-card-foreground shadow-lg border border-border/60 p-6 relative overflow-hidden transition-all duration-200 hover:-translate-y-0.5 hover:shadow-xl">
      {/* Header with label and character count */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <label 
          htmlFor="text-input" 
          className="inline-flex items-center text-sm font-semibold text-foreground tracking-tight"
        >
          <FaEdit className="mr-2 text-primary" size={14} />
          <span>Enter Text to Synthesize</span>
        </label>
        
        <div className="flex items-center gap-2">
          {/* Tip toggle button */}
          <button
            onClick={toggleTip}
            className="flex items-center justify-center w-7 h-7 rounded-full bg-muted/20 hover:bg-muted/30 text-muted-foreground transition-colors duration-150"
            aria-label="Show hint"
            title="Show tip"
          >
            {showTip ? 
              <FaLightbulb size={14} className="text-amber-400" /> : 
              <FaRegLightbulb size={14} />
            }
          </button>
          
          {/* Character count with animated transition */}
          <div 
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium transition-all duration-300 ${
              charCount > 0 ? 'bg-primary/8' : 'bg-transparent'
            } ${getCountClass()}`}
          >
            <span>{charCount}</span>
            <span className="opacity-70">/</span>
            <span>{CHAR_LIMIT}</span>
            
            {isAtLimit && (
              <FaInfoCircle 
                className="ml-1 text-destructive animate-pulse"
                size={11}
                title="Character limit reached"
              />
            )}
          </div>
        </div>
      </div>
      
      {/* Tip display */}
      {showTip && (
        <div className="mb-3 p-3 rounded-lg bg-accent/5 border border-accent/20 text-sm flex items-start gap-2 animate-in fade-in duration-200">
          <FaLightbulb className="text-amber-400 mt-0.5 flex-shrink-0" size={14} />
          <div>
            <span className="font-medium">Tip:</span> {getRandomTip()}
          </div>
        </div>
      )}
      
      {/* Character limit progress bar */}
      <div className="h-1 w-full mb-3 rounded-full overflow-hidden bg-secondary/40">
        <div 
          className={`h-full transition-all duration-300 ease-out rounded-full ${getProgressBarClass()}`}
          style={{ width: `${charPercentage}%` }}
        />
      </div>
      
      {/* Textarea or HighlightableText container with enhanced styling */}
      <div 
        className={`relative rounded-lg transition-all duration-200 ease-out ${
          isFocused && !isReadOnly
            ? 'shadow-lg -translate-y-0.5 ring-2 ring-primary/15' 
            : isReadOnly
              ? 'shadow-lg ring-2 ring-accent/15'
              : 'shadow-md'
        }`}
      >
        {isReadOnly ? (
          // Read-only mode with highlighting
          <div className={`block w-full min-h-[180px] max-h-[500px] rounded-lg border p-4 text-base transition-all duration-200 border-accent bg-background/80`}>
            <HighlightableText 
              text={inputText} 
              className="h-full max-h-[500px]"
            />
          </div>
        ) : (
          // Editable textarea
          <textarea
            ref={textareaRef}
            id="text-input"
            className={`block w-full min-h-[180px] rounded-lg border p-4 text-base transition-all duration-200 placeholder:text-muted-foreground/70 focus:outline-none resize-none ${
              isFocused 
                ? 'border-primary bg-background/80' 
                : 'border-input bg-background/80'
            }`}
            rows={8}
            onFocus={handleFocus}
            onBlur={handleBlur}
            placeholder="Enter text to generate speech. Try using proper punctuation and formatting for the best results."
            value={inputText}
            onChange={handleChange}
            aria-label="Text to synthesize"
            aria-describedby="char-limit-info"
            spellCheck="true"
            maxLength={CHAR_LIMIT}
          />
        )}
        
        {/* Animated background gradient when focused */}
        {isFocused && (
          <div 
            className="absolute -inset-0.5 rounded-xl -z-10 opacity-30 pointer-events-none bg-gradient-to-tr from-primary/30 via-accent/20 to-primary/30 bg-[length:200%_200%] animate-gradient"
          />
        )}
        
        {/* Animation keyframes */}
        <style>
          {`
            @keyframes gradientFlow {
              0% { background-position: 0% 50%; }
              50% { background-position: 100% 50%; }
              100% { background-position: 0% 50%; }
            }
            
            @keyframes pulse {
              0%, 100% { transform: scale(1); opacity: 0.8; }
              50% { transform: scale(1.05); opacity: 1; }
            }
            
            @keyframes fadeIn {
              from { opacity: 0; transform: translateY(5px); }
              to { opacity: 1; transform: translateY(0); }
            }
            
            .animate-pulse {
              animation: pulse 2s ease-in-out infinite;
            }
            
            .animate-in.fade-in {
              animation: fadeIn 0.5s ease-out forwards;
            }
            
            .animate-gradient {
              animation: gradientFlow 6s ease infinite;
            }
          `}
        </style>
        
        {/* Icon overlay when empty with pulsing animation - only show when not in read-only mode */}
        {inputText.length === 0 && !isReadOnly && (
          <div 
            className={`absolute inset-0 flex flex-col items-center justify-center pointer-events-none transition-opacity duration-300 ${
              isFocused ? 'opacity-5' : 'opacity-10'
            }`}
          >
            <FaVolumeUp className="text-muted-foreground animate-pulse text-6xl" />
            <div className="mt-4 text-sm text-muted-foreground font-medium opacity-60">
              Start typing or paste text
            </div>
          </div>
        )}
        
        {/* Text formatting toolbar - only show when not in read-only mode */}
        {isFocused && !isReadOnly && (
          <div 
            className={`absolute top-2 left-2 flex gap-1 p-1 rounded-md bg-background/90 backdrop-blur-sm shadow-sm transition-all duration-300 border border-border/40 ${
              showFormatting 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 -translate-y-2 pointer-events-none'
            }`}
          >
            <button 
              onClick={() => formatText('bold')}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all duration-150"
              aria-label="Bold text"
              title="Bold text"
            >
              <FaBold size={14} />
            </button>
            <button 
              onClick={() => formatText('italic')}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all duration-150"
              aria-label="Italic text"
              title="Italic text"
            >
              <FaItalic size={14} />
            </button>
            <button 
              onClick={() => formatText('heading')}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all duration-150"
              aria-label="Add heading"
              title="Add heading"
            >
              <FaHeading size={14} />
            </button>
            <button 
              onClick={() => formatText('quote')}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all duration-150"
              aria-label="Add quote"
              title="Add quote"
            >
              <FaQuoteRight size={14} />
            </button>
            <button 
              onClick={() => formatText('list')}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-primary hover:bg-primary/10 transition-all duration-150"
              aria-label="Add list item"
              title="Add list item"
            >
              <FaListUl size={14} />
            </button>
          </div>
        )}
        
        {/* Format toggle button - only show when not in read-only mode */}
        {isFocused && !isReadOnly && (
          <button 
            onClick={toggleFormatting}
            className={`absolute top-2 left-2 w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground transition-all duration-200 ${
              showFormatting 
                ? 'bg-primary/10 text-primary transform rotate-180' 
                : 'bg-background/80 hover:bg-secondary/30'
            }`}
            aria-label={showFormatting ? "Hide formatting options" : "Show formatting options"}
            title={showFormatting ? "Hide formatting options" : "Show formatting options"}
          >
            <FaKeyboard size={14} />
          </button>
        )}
        
        {/* Action buttons that appear when there's text - only show when not in read-only mode */}
        {inputText.length > 0 && !isReadOnly && (
          <div 
            className={`absolute bottom-2 right-2 flex gap-1 p-1 rounded-md bg-background/80 backdrop-blur-sm shadow-sm transition-all duration-200 border border-border/40 ${
              isFocused 
                ? 'opacity-100 translate-y-0' 
                : 'opacity-0 translate-y-2'
            }`}
          >
            <button 
              onClick={copyText}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-foreground hover:bg-secondary/70 transition-all duration-150"
              aria-label="Copy text"
              title="Copy text"
            >
              <FaClipboard size={14} />
            </button>
            <button 
              onClick={clearText}
              className="w-8 h-8 flex items-center justify-center rounded-md text-muted-foreground hover:text-destructive hover:bg-destructive/10 transition-all duration-150"
              aria-label="Clear text"
              title="Clear text"
            >
              <FaTrash size={14} />
            </button>
          </div>
        )}

        {/* Copied to clipboard toast notification */}
        {showCopiedToast && (
          <div className="absolute top-2 right-2 flex items-center py-1.5 px-3 rounded-full bg-primary text-primary-foreground text-xs font-medium shadow-md animate-in fade-in duration-200">
            <FaCheck size={10} className="mr-1" />
            Copied to clipboard
          </div>
        )}
      </div>
      
      {/* Helper text and suggestion button with improved styling */}
      <div className="flex flex-wrap justify-between items-center mt-4 gap-2">
        <div className="flex items-start text-xs text-muted-foreground max-w-[70%]">
          <FaInfoCircle className="mr-1.5 mt-0.5 flex-shrink-0" size={11} />
          <p id="char-limit-info" className="m-0 leading-relaxed">
            For best results, use proper punctuation and formatting.
            {isApproachingLimit && (
              <span className="text-amber-500 font-medium ml-1">
                Approaching character limit.
              </span>
            )}
          </p>
        </div>
        
        <div className="flex gap-2">
          <button 
            className="inline-flex items-center text-xs font-medium text-primary bg-primary/5 border border-primary/20 py-1.5 px-3 rounded-lg transition-all duration-150 hover:bg-primary/10 hover:border-primary/30 active:scale-95"
            onClick={addSampleText}
            aria-label="Add sample text"
          >
            <FaRandom className="mr-1.5" size={11} />
            Add sample text
          </button>
        </div>
      </div>
    </div>
  );
}

export default TextInput;
