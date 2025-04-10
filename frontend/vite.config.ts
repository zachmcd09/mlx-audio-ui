/// <reference types="vitest" />
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true, // Use Vitest global APIs (describe, it, expect, etc.)
    environment: 'jsdom', // Simulate browser environment for testing React components
    setupFiles: './src/setupTests.ts', // Run global setup before tests
    // Optional: Configure code coverage (using istanbul provider as recommended)
    coverage: {
      provider: 'istanbul', // or 'v8'
      reporter: ['text', 'json', 'html', 'lcov'],
      include: ['src/**/*.{ts,tsx}'], // Adjust include/exclude patterns as needed
      exclude: [
        'src/main.tsx',
        'src/vite-env.d.ts',
        'src/mocks/**',
        'src/setupTests.ts',
        '**/*.test.{ts,tsx}', // Exclude test files themselves
        '**/info.md', // Exclude info files
      ],
      // Optional: Thresholds for coverage enforcement
      // thresholds: {
      //   lines: 80,
      //   functions: 80,
      //   branches: 80,
      //   statements: 80,
      // },
    },
  },
})
