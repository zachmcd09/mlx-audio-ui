/**
 * Theme utility functions for consistent color and style usage
 * This centralizes access to theme variables and provides helper functions
 */

// CSS variable names for theme properties
export const themeVars = {
  // Colors
  background: 'var(--background)',
  foreground: 'var(--foreground)',
  card: 'var(--card)',
  cardForeground: 'var(--card-foreground)',
  popover: 'var(--popover)',
  popoverForeground: 'var(--popover-foreground)',
  primary: 'var(--primary)',
  primaryForeground: 'var(--primary-foreground)',
  secondary: 'var(--secondary)',
  secondaryForeground: 'var(--secondary-foreground)',
  muted: 'var(--muted)',
  mutedForeground: 'var(--muted-foreground)',
  accent: 'var(--accent)',
  accentForeground: 'var(--accent-foreground)',
  destructive: 'var(--destructive)',
  destructiveForeground: 'var(--destructive-foreground)',
  border: 'var(--border)',
  input: 'var(--input)',
  ring: 'var(--ring)',
  
  // Other properties
  radius: 'var(--radius)',
};

/**
 * Convert a CSS variable reference to the corresponding Tailwind class
 * @param varName - CSS variable name without the 'var(--' prefix
 * @returns The corresponding Tailwind class name
 */
export const getTailwindClass = (varName: string): string => {
  const classMap: Record<string, string> = {
    'background': 'bg-background',
    'foreground': 'text-foreground',
    'primary': 'bg-primary',
    'primary-foreground': 'text-primary-foreground',
    'secondary': 'bg-secondary',
    'secondary-foreground': 'text-secondary-foreground',
    'accent': 'bg-accent',
    'accent-foreground': 'text-accent-foreground',
    'muted': 'bg-muted',
    'muted-foreground': 'text-muted-foreground',
    'border': 'border-border',
    // Add other mappings as needed
  };
  
  return classMap[varName] || '';
};

/**
 * Helper for creating conditional class names
 * @param conditions - Object with class names as keys and boolean conditions as values
 * @returns String of class names for which the condition is true
 */
export const cx = (...classes: (string | boolean | undefined | null)[]): string => {
  return classes.filter(Boolean).join(' ');
};

/**
 * Determines if the current theme is dark
 * @returns Boolean indicating if dark mode is active
 */
export const isDarkMode = (): boolean => {
  if (typeof document === 'undefined') return false;
  return document.documentElement.classList.contains('dark');
};

/**
 * Toggle between dark and light mode
 */
export const toggleDarkMode = (): void => {
  if (typeof document === 'undefined') return;
  const isDark = isDarkMode();
  
  if (isDark) {
    document.documentElement.classList.remove('dark');
    localStorage.setItem('theme', 'light');
  } else {
    document.documentElement.classList.add('dark');
    localStorage.setItem('theme', 'dark');
  }
};

/**
 * Initialize theme based on user preference or system preference
 */
export const initializeTheme = (): void => {
  if (typeof document === 'undefined') return;
  
  // Check for saved preference
  const savedTheme = localStorage.getItem('theme');
  
  if (savedTheme === 'dark') {
    document.documentElement.classList.add('dark');
  } else if (savedTheme === 'light') {
    document.documentElement.classList.remove('dark');
  } else {
    // Check system preference
    const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
    if (prefersDark) {
      document.documentElement.classList.add('dark');
    }
  }
};
