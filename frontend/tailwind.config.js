/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  // Add safelist for dynamically generated classes
  safelist: [
    // Background colors with opacity variations
    'bg-background',
    'bg-foreground',
    'bg-primary',
    'bg-primary-foreground',
    'bg-secondary',
    'bg-secondary-foreground',
    'bg-muted',
    'bg-muted-foreground',
    'bg-accent',
    'bg-accent-foreground',
    'bg-destructive',
    'bg-destructive-foreground',
    'bg-card',
    'bg-card-foreground',
    'bg-popover',
    'bg-popover-foreground',
    'bg-border',
    'bg-input',
    'bg-ring',
    // Text colors with opacity variations
    'text-background',
    'text-foreground',
    'text-primary',
    'text-primary-foreground',
    'text-secondary',
    'text-secondary-foreground',
    'text-muted',
    'text-muted-foreground',
    'text-accent',
    'text-accent-foreground',
    'text-destructive',
    'text-destructive-foreground',
    'text-card',
    'text-card-foreground',
    'text-popover',
    'text-popover-foreground',
    // Border colors
    'border-border',
    'border-input',
    'border-ring',
    'border-primary',
    'border-secondary',
    'border-accent',
    'border-destructive',
    // Background opacity variants
    {
      pattern: /bg-(background|foreground|primary|secondary|muted|accent|destructive|card|popover|border|input|ring)\/[0-9]+/,
    },
    // Text opacity variants
    {
      pattern: /text-(background|foreground|primary|secondary|muted|accent|destructive|card|popover|border|input|ring)\/[0-9]+/,
    },
  ],
  theme: {
    extend: {
      colors: {
        'border': {
          DEFAULT: 'hsl(var(--border) / <alpha-value>)',
        },
        'input': {
          DEFAULT: 'hsl(var(--input) / <alpha-value>)',
        },
        'ring': {
          DEFAULT: 'hsl(var(--ring) / <alpha-value>)',
        },
        'background': {
          DEFAULT: 'hsl(var(--background) / <alpha-value>)',
        },
        'foreground': {
          DEFAULT: 'hsl(var(--foreground) / <alpha-value>)',
        },
        primary: {
          DEFAULT: 'hsl(var(--primary) / <alpha-value>)',
          foreground: 'hsl(var(--primary-foreground) / <alpha-value>)',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary) / <alpha-value>)',
          foreground: 'hsl(var(--secondary-foreground) / <alpha-value>)',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive) / <alpha-value>)',
          foreground: 'hsl(var(--destructive-foreground) / <alpha-value>)',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted) / <alpha-value>)',
          foreground: 'hsl(var(--muted-foreground) / <alpha-value>)',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent) / <alpha-value>)',
          foreground: 'hsl(var(--accent-foreground) / <alpha-value>)',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover) / <alpha-value>)',
          foreground: 'hsl(var(--popover-foreground) / <alpha-value>)',
        },
        card: {
          DEFAULT: 'hsl(var(--card) / <alpha-value>)',
          foreground: 'hsl(var(--card-foreground) / <alpha-value>)',
        },
        // Add the neutral color palette
        neutral: {
          50: '#fafafa',
          100: '#f5f5f5',
          200: '#e5e5e5',
          300: '#d4d4d4',
          400: '#a3a3a3',
          500: '#737373',
          600: '#525252',
          700: '#404040',
          800: '#262626',
          900: '#171717',
          950: '#0a0a0a',
        },
      },
      borderRadius: {
        lg: `var(--radius)`,
        md: `calc(var(--radius) - 2px)`,
        sm: "calc(var(--radius) - 4px)",
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', '"Helvetica Neue"', 'Arial', '"Noto Sans"', 'sans-serif', '"Apple Color Emoji"', '"Segoe UI Emoji"', '"Segoe UI Symbol"', '"Noto Color Emoji"'],
      },
      keyframes: {
        "accordion-down": {
          from: { height: "0" },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: "0" },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [require("tailwindcss-animate")], // Add animate plugin
}
