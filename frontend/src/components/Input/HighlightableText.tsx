import React, { useEffect, useRef, useState } from 'react';
import { useAppStore } from '../../store';
import { cx } from '../../utils/theme';

interface HighlightableTextProps {
  text: string;
  className?: string;
}

// Helper function to determine if a range is visible within a container
const isRangeVisible = (
  rangeRect: DOMRect, 
  containerRect: DOMRect, 
  threshold = 0.5 // What portion of the element needs to be visible
): boolean => {
  const rangeHeight = rangeRect.height;
  const visibleTop = Math.max(rangeRect.top, containerRect.top);
  const visibleBottom = Math.min(rangeRect.bottom, containerRect.bottom);
  const visibleHeight = Math.max(0, visibleBottom - visibleTop);
  
  return visibleHeight / rangeHeight >= threshold;
};

const HighlightableText: React.FC<HighlightableTextProps> = ({ text, className = '' }) => {
  const textContainerRef = useRef<HTMLDivElement>(null);
  const highlightedTextRange = useAppStore((state) => state.highlightedTextRange);
  const playbackState = useAppStore((state) => state.playbackState);
  const [lastScrollPosition, setLastScrollPosition] = useState<number>(0);
  const lastHighlightRef = useRef<{start: number, end: number} | null>(null);
  
  // Improved auto-scroll when highlighted text changes
  useEffect(() => {
    if (!highlightedTextRange || !textContainerRef.current) return;
    
    // If no change in the highlight, don't scroll
    if (
      lastHighlightRef.current && 
      lastHighlightRef.current.start === highlightedTextRange.start && 
      lastHighlightRef.current.end === highlightedTextRange.end
    ) {
      return;
    }
    
    // Save the current highlight for later comparison
    lastHighlightRef.current = highlightedTextRange;
    
    const container = textContainerRef.current;
    
    // Create a range to measure where the highlighted text is
    const range = document.createRange();
    const textNode = container.firstChild;
    
    if (!textNode) return;
    
    // Set the range to the highlighted portion
    const start = Math.max(0, Math.min(highlightedTextRange.start, text.length));
    const end = Math.max(0, Math.min(highlightedTextRange.end, text.length));
    
    try {
      range.setStart(textNode, start);
      range.setEnd(textNode, end);
      
      // Get the bounding rectangle of the highlighted text
      const rect = range.getBoundingClientRect();
      
      // Get the container's bounding rect
      const containerRect = container.getBoundingClientRect();
      
      // Check if the highlighted text is already visible
      if (isRangeVisible(rect, containerRect)) {
        // Already visible, no need to scroll
        return;
      }
      
      // Calculate how much to scroll
      const containerScrollTop = container.scrollTop;
      const rangeTop = rect.top - containerRect.top + containerScrollTop;
      const rangeHeight = rect.height;
      const visibleHeight = containerRect.height;
      
      // Target position - aim to position the highlighted text slightly above center
      const targetPosition = rangeTop - (visibleHeight * 0.4 - rangeHeight / 2);
      
      // Only scroll if we're moving by a significant amount
      if (Math.abs(targetPosition - containerScrollTop) > 20) {
        // Save the scroll position we're navigating to
        setLastScrollPosition(targetPosition);
        
        container.scrollTo({
          top: targetPosition,
          behavior: playbackState === 'Playing' ? 'smooth' : 'auto'
        });
      }
    } catch (e) {
      console.error('Error setting range:', e);
      // Enhanced fallback: scroll to a calculated position based on character count
      const charsPerLine = Math.max(50, Math.min(120, Math.floor(container.clientWidth / 8))); // Estimate chars per line based on container width
      const averageLineHeight = 24; // pixels, estimate
      const lineNumber = Math.floor(start / charsPerLine);
      const scrollPosition = lineNumber * averageLineHeight;
      
      container.scrollTop = scrollPosition;
      setLastScrollPosition(scrollPosition);
    }
  }, [highlightedTextRange, text, playbackState]);
  
  // Render the text with enhanced highlighting
  const renderText = () => {
    if (!highlightedTextRange || playbackState !== 'Playing') {
      // If not playing or no highlight, just show the text
      return text;
    }
    
    const start = Math.max(0, Math.min(highlightedTextRange.start, text.length));
    const end = Math.max(0, Math.min(highlightedTextRange.end, text.length));
    
    // Three parts: before highlight, highlighted, after highlight
    const beforeText = text.substring(0, start);
    const highlightedText = text.substring(start, end);
    const afterText = text.substring(end);
    
    return (
      <>
        {beforeText}
        <span className="bg-primary/20 text-foreground dark:bg-primary/30 font-medium rounded-sm px-1 py-0.5 transition-colors duration-300 relative">
          {/* Subtle animation effect for the highlight */}
          <span className="absolute inset-0 rounded-sm animate-pulse bg-primary/10 dark:bg-primary/10"></span>
          {highlightedText}
        </span>
        {afterText}
      </>
    );
  };
  
  // Handle the scenario when playback stops - keep the last highlight visible
  useEffect(() => {
    if (playbackState !== 'Playing' && playbackState !== 'Paused' && textContainerRef.current) {
      // If we're stopping, keep the scroll position at the last highlighted text
      if (lastScrollPosition > 0) {
        textContainerRef.current.scrollTop = lastScrollPosition;
      }
    }
  }, [playbackState, lastScrollPosition]);
  
  return (
    <div 
      ref={textContainerRef}
      className={cx(
        "font-sans whitespace-pre-wrap break-words overflow-y-auto",
        "text-base leading-relaxed tracking-wide",
        "transition-colors duration-150",
        className
      )}
      style={{ scrollBehavior: 'smooth' }}
    >
      {renderText()}
    </div>
  );
};

export default HighlightableText;
