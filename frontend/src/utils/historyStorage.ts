/**
 * Utilities for managing the history of generated text in local storage
 */

export interface HistoryEntry {
  id: string;
  text: string;
  voice: string | null;
  date: string; // ISO string
  title: string; // Optional title or generated from text
  isPinned?: boolean;
}

// Key used for localStorage
const HISTORY_STORAGE_KEY = 'mlx-audio-history';

// Maximum number of history entries to store
const MAX_HISTORY_ENTRIES = 50;

/**
 * Generate a title from text (first ~30 chars)
 */
export const generateTitleFromText = (text: string): string => {
  // Get the first sentence, or first 30 chars
  const firstSentence = text.split(/[.!?]/)[0].trim();
  
  if (firstSentence.length <= 30) {
    return firstSentence;
  }
  
  // Truncate at word boundary
  const truncated = firstSentence.substring(0, 30).split(' ');
  truncated.pop(); // Remove the last (potentially partial) word
  return truncated.join(' ') + '...';
};

/**
 * Save a new history entry
 */
export const saveToHistory = (entry: Omit<HistoryEntry, 'id' | 'date'>): HistoryEntry => {
  // Generate ID and add timestamp
  const newEntry: HistoryEntry = {
    ...entry,
    id: `history_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    date: new Date().toISOString(),
    title: entry.title || generateTitleFromText(entry.text),
  };
  
  // Get existing entries
  const existingEntries = getHistoryEntries();
  
  // Add new entry at the beginning
  const updatedEntries = [newEntry, ...existingEntries];
  
  // Limit the number of entries
  const trimmedEntries = updatedEntries.slice(0, MAX_HISTORY_ENTRIES);
  
  // Save back to storage
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(trimmedEntries));
  
  return newEntry;
};

/**
 * Get all history entries
 */
export const getHistoryEntries = (): HistoryEntry[] => {
  try {
    const entriesJson = localStorage.getItem(HISTORY_STORAGE_KEY);
    if (!entriesJson) return [];
    
    const entries = JSON.parse(entriesJson);
    if (!Array.isArray(entries)) return [];
    
    return entries;
  } catch (error) {
    console.error('Failed to parse history entries:', error);
    return [];
  }
};

/**
 * Update an existing history entry
 */
export const updateHistoryEntry = (id: string, updates: Partial<HistoryEntry>): boolean => {
  const entries = getHistoryEntries();
  
  const index = entries.findIndex(entry => entry.id === id);
  if (index === -1) return false;
  
  // Update the entry
  entries[index] = { ...entries[index], ...updates };
  
  // Save back to storage
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(entries));
  
  return true;
};

/**
 * Delete a history entry by ID
 */
export const deleteHistoryEntry = (id: string): boolean => {
  const entries = getHistoryEntries();
  
  const filteredEntries = entries.filter(entry => entry.id !== id);
  
  // If no entries were removed, return false
  if (filteredEntries.length === entries.length) return false;
  
  // Save back to storage
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(filteredEntries));
  
  return true;
};

/**
 * Clear all history entries
 */
export const clearAllHistory = (): void => {
  localStorage.removeItem(HISTORY_STORAGE_KEY);
};

/**
 * Toggle pinned status of an entry
 */
export const togglePinHistoryEntry = (id: string): boolean => {
  const entries = getHistoryEntries();
  
  const index = entries.findIndex(entry => entry.id === id);
  if (index === -1) return false;
  
  // Toggle isPinned
  entries[index].isPinned = !entries[index].isPinned;
  
  // Resort entries to keep pinned at top
  const pinnedEntries = entries.filter(entry => entry.isPinned);
  const unpinnedEntries = entries.filter(entry => !entry.isPinned);
  
  const sortedEntries = [
    ...pinnedEntries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()),
    ...unpinnedEntries.sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime())
  ];
  
  // Save back to storage
  localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(sortedEntries));
  
  return true;
};

/**
 * Export history entries as JSON
 */
export const exportHistory = (): string => {
  const entries = getHistoryEntries();
  return JSON.stringify(entries, null, 2);
};

/**
 * Import history entries from JSON
 */
export const importHistory = (json: string): boolean => {
  try {
    const newEntries = JSON.parse(json);
    
    if (!Array.isArray(newEntries)) {
      throw new Error('Invalid format: data is not an array');
    }
    
    // Validate each entry
    newEntries.forEach(entry => {
      if (!entry.id || !entry.text || !entry.date) {
        throw new Error('Invalid entry format: missing required fields');
      }
    });
    
    // Replace existing entries
    localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(newEntries));
    
    return true;
  } catch (error) {
    console.error('Failed to import history:', error);
    return false;
  }
};
