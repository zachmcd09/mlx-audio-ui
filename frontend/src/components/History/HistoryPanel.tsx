import React, { useState, useEffect } from 'react';
import { 
  FaHistory, FaTrash, FaStar, FaRegStar, 
  FaDownload, FaUpload, FaExclamationCircle,
  FaPlay, FaTimes, FaSearch, FaEllipsisH
} from 'react-icons/fa';
import { useAppStore } from '../../store';
import { 
  HistoryEntry, 
  getHistoryEntries, 
  deleteHistoryEntry, 
  togglePinHistoryEntry,
  clearAllHistory,
  exportHistory,
  importHistory
} from '../../utils/historyStorage';
import { cx } from '../../utils/theme';

interface HistoryPanelProps {
  className?: string;
}

const HistoryPanel: React.FC<HistoryPanelProps> = ({ className = '' }) => {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [showConfirmClear, setShowConfirmClear] = useState(false);
  const [showActions, setShowActions] = useState<string | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [importError, setImportError] = useState<string | null>(null);
  
  const setInputText = useAppStore((state) => state.setInputText);
  
  // Load entries on mount
  useEffect(() => {
    refreshEntries();
  }, []);
  
  // Refresh entries from storage
  const refreshEntries = () => {
    const allEntries = getHistoryEntries();
    setEntries(allEntries);
  };
  
  // Toggle pin status
  const handleTogglePin = (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click
    togglePinHistoryEntry(id);
    refreshEntries();
  };
  
  // Delete an entry
  const handleDelete = (id: string, e: React.MouseEvent) => {
    e.stopPropagation(); // Prevent row click
    deleteHistoryEntry(id);
    refreshEntries();
  };
  
  // Clear all history (with confirmation)
  const handleClearAll = () => {
    clearAllHistory();
    refreshEntries();
    setShowConfirmClear(false);
  };
  
  // Load text into input
  const handleLoadText = (entry: HistoryEntry) => {
    setInputText(entry.text);
    // Close any open action menus
    setShowActions(null);
  };
  
  // Export history
  const handleExport = () => {
    const historyData = exportHistory();
    const blob = new Blob([historyData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    
    // Create and click a download link
    const a = document.createElement('a');
    a.href = url;
    a.download = `mlx-audio-history-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    
    // Clean up
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
  
  // Import history
  const handleImport = (e: React.ChangeEvent<HTMLInputElement>) => {
    setImportError(null);
    
    const file = e.target.files?.[0];
    if (!file) return;
    
    const reader = new FileReader();
    
    reader.onload = (event) => {
      try {
        setIsImporting(true);
        const content = event.target?.result as string;
        const success = importHistory(content);
        
        if (success) {
          refreshEntries();
        } else {
          setImportError('Failed to import history. Invalid format.');
        }
      } catch (error) {
        setImportError('Error reading file: ' + (error instanceof Error ? error.message : 'Unknown error'));
      } finally {
        setIsImporting(false);
      }
    };
    
    reader.onerror = () => {
      setImportError('Error reading file');
      setIsImporting(false);
    };
    
    reader.readAsText(file);
    
    // Reset the input so the same file can be selected again
    e.target.value = '';
  };
  
  // Filter entries based on search
  const filteredEntries = searchQuery.trim()
    ? entries.filter(entry => 
        entry.title.toLowerCase().includes(searchQuery.toLowerCase()) || 
        entry.text.toLowerCase().includes(searchQuery.toLowerCase()))
    : entries;
  
  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    
    // Check if it's today
    const today = new Date();
    const isToday = date.getDate() === today.getDate() &&
                   date.getMonth() === today.getMonth() &&
                   date.getFullYear() === today.getFullYear();
    
    if (isToday) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }
    
    // Check if it's yesterday
    const yesterday = new Date();
    yesterday.setDate(yesterday.getDate() - 1);
    const isYesterday = date.getDate() === yesterday.getDate() &&
                       date.getMonth() === yesterday.getMonth() &&
                       date.getFullYear() === yesterday.getFullYear();
    
    if (isYesterday) {
      return 'Yesterday';
    }
    
    // Otherwise, show the date
    return date.toLocaleDateString([], { month: 'short', day: 'numeric' });
  };
  
  return (
    <div className={cx(
      "flex flex-col",
      "h-full overflow-hidden",
      className
    )}>
      {/* Header with search and actions */}
      <div className="flex flex-col gap-2 mb-3">
        {/* Search bar */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search history..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            className="w-full px-8 py-1.5 text-xs rounded-md 
                      bg-background border border-input
                      focus:outline-none focus:ring-1 focus:ring-primary
                      placeholder:text-muted-foreground"
          />
          <FaSearch className="absolute left-2.5 top-1/2 transform -translate-y-1/2 text-muted-foreground" size={12} />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 transform -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <FaTimes size={12} />
            </button>
          )}
        </div>
        
        {/* Action buttons */}
        <div className="flex justify-between items-center">
          <div className="text-xs text-muted-foreground">
            {filteredEntries.length} {filteredEntries.length === 1 ? 'entry' : 'entries'}
          </div>
          
          <div className="flex gap-1">
            {/* Import button */}
            <label className="relative cursor-pointer">
              <input 
                type="file" 
                accept=".json" 
                onChange={handleImport} 
                className="absolute inset-0 opacity-0 w-full cursor-pointer"
                disabled={isImporting}
              />
              <div className="flex items-center justify-center rounded-md 
                            bg-secondary/60 hover:bg-secondary
                            text-secondary-foreground text-xs
                            px-2 py-1 transition-colors">
                <FaUpload size={10} className="mr-1.5" />
                <span>Import</span>
              </div>
            </label>
            
            {/* Export button */}
            <button
              onClick={handleExport}
              disabled={entries.length === 0}
              className="flex items-center justify-center rounded-md 
                        bg-secondary/60 hover:bg-secondary
                        text-secondary-foreground text-xs
                        px-2 py-1 transition-colors
                        disabled:opacity-50 disabled:pointer-events-none"
            >
              <FaDownload size={10} className="mr-1.5" />
              <span>Export</span>
            </button>
            
            {/* Clear all button */}
            <button
              onClick={() => setShowConfirmClear(true)}
              disabled={entries.length === 0}
              className="flex items-center justify-center rounded-md 
                        bg-destructive/10 hover:bg-destructive/20
                        text-destructive text-xs
                        px-2 py-1 transition-colors
                        disabled:opacity-50 disabled:pointer-events-none"
            >
              <FaTrash size={10} className="mr-1.5" />
              <span>Clear</span>
            </button>
          </div>
        </div>
      </div>
      
      {/* Import error */}
      {importError && (
        <div className="bg-destructive/10 border border-destructive/30 text-destructive rounded-md p-2 mb-3 text-xs">
          <div className="flex items-center">
            <FaExclamationCircle className="mr-1.5" />
            <span className="font-medium">Error importing</span>
          </div>
          <p className="mt-1">{importError}</p>
          <button 
            onClick={() => setImportError(null)}
            className="mt-1 text-xs underline text-destructive/90 hover:text-destructive"
          >
            Dismiss
          </button>
        </div>
      )}
      
      {/* Confirm clear dialog */}
      {showConfirmClear && (
        <div className="bg-card border border-border rounded-md p-3 mb-3 text-sm">
          <h4 className="font-medium mb-2">Clear all history?</h4>
          <p className="text-xs text-muted-foreground mb-3">
            This will permanently delete all saved entries. This action cannot be undone.
          </p>
          <div className="flex justify-end gap-2">
            <button
              onClick={() => setShowConfirmClear(false)}
              className="px-3 py-1 text-xs rounded-md bg-secondary/60 hover:bg-secondary text-secondary-foreground"
            >
              Cancel
            </button>
            <button
              onClick={handleClearAll}
              className="px-3 py-1 text-xs rounded-md bg-destructive hover:bg-destructive/90 text-destructive-foreground"
            >
              Delete All
            </button>
          </div>
        </div>
      )}
      
      {/* Entries list */}
      <div className="flex-1 overflow-y-auto -mx-1 px-1">
        {filteredEntries.length === 0 ? (
          <div className="flex flex-col items-center justify-center text-center py-6 h-full text-muted-foreground">
            <FaHistory className="text-muted-foreground/50 mb-2" size={24} />
            {entries.length === 0 ? (
              <>
                <p className="text-sm mb-1">No history entries yet</p>
                <p className="text-xs">Generated texts will appear here</p>
              </>
            ) : (
              <p className="text-sm">No matching entries</p>
            )}
          </div>
        ) : (
          <ul className="space-y-2">
            {filteredEntries.map(entry => (
              <li 
                key={entry.id}
                onClick={() => handleLoadText(entry)}
                className="relative group bg-muted/20 hover:bg-muted/40 rounded-md p-2 cursor-pointer transition-colors border border-border/40"
              >
                {/* Title and pin button row */}
                <div className="flex justify-between items-start mb-1">
                  <h3 className="text-xs font-medium truncate pr-8">{entry.title}</h3>
                  <button
                    onClick={(e) => handleTogglePin(entry.id, e)}
                    className="text-xs text-muted-foreground hover:text-amber-500 focus:outline-none transition-colors"
                    aria-label={entry.isPinned ? "Unpin" : "Pin"}
                  >
                    {entry.isPinned ? (
                      <FaStar className="text-amber-500" size={14} />
                    ) : (
                      <FaRegStar className="opacity-0 group-hover:opacity-100" size={14} />
                    )}
                  </button>
                </div>
                
                {/* Text preview */}
                <div className="text-xs text-muted-foreground line-clamp-2 leading-relaxed mb-2">
                  {entry.text}
                </div>
                
                {/* Meta info & actions row */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                    <span>{formatDate(entry.date)}</span>
                    {entry.voice && (
                      <span className="bg-secondary/30 rounded-full px-1.5 py-0.5">
                        {entry.voice}
                      </span>
                    )}
                  </div>
                  
                  <div className="flex items-center">
                    <button
                      onClick={(e) => { e.stopPropagation(); setShowActions(showActions === entry.id ? null : entry.id); }}
                      className="rounded-md p-1 bg-secondary/0 hover:bg-secondary/50 transition-colors"
                      aria-label="More actions"
                    >
                      <FaEllipsisH size={10} className="text-muted-foreground" />
                    </button>
                    
                    {/* Actions menu */}
                    {showActions === entry.id && (
                      <div className="absolute right-0 bottom-full mb-1 bg-popover text-popover-foreground rounded-md shadow-lg border border-border p-1 text-xs min-w-24 z-10">
                        <button
                          onClick={(e) => handleLoadText(entry)}
                          className="flex items-center w-full px-2 py-1 hover:bg-muted rounded text-left"
                        >
                          <FaPlay size={10} className="mr-1.5" />
                          <span>Load text</span>
                        </button>
                        <button
                          onClick={(e) => handleTogglePin(entry.id, e)}
                          className="flex items-center w-full px-2 py-1 hover:bg-muted rounded text-left"
                        >
                          {entry.isPinned ? <FaRegStar size={10} className="mr-1.5" /> : <FaStar size={10} className="mr-1.5" />}
                          <span>{entry.isPinned ? 'Unpin' : 'Pin'}</span>
                        </button>
                        <button
                          onClick={(e) => handleDelete(entry.id, e)}
                          className="flex items-center w-full px-2 py-1 hover:bg-destructive/10 text-destructive rounded text-left"
                        >
                          <FaTrash size={10} className="mr-1.5" />
                          <span>Delete</span>
                        </button>
                      </div>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default HistoryPanel;
