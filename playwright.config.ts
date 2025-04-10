import { defineConfig, devices } from '@playwright/test';
import path from 'path';

// Read environment variables or use defaults
const PORT = process.env.PORT || '5173'; // Default Vite dev server port
const BASE_URL = process.env.BASE_URL || `http://localhost:${PORT}`;

// Define where the E2E tests are located
const testDir = path.resolve(__dirname, 'e2e');

export default defineConfig({
  // Directory containing the test files
  testDir: testDir,

  // Maximum time one test can run for (milliseconds). Increased due to potential TTS delays.
  timeout: 60 * 1000, // 60 seconds

  // Maximum time expect() should wait for conditions (milliseconds).
  expect: {
    timeout: 10 * 1000, // 10 seconds
  },

  // Run tests in files in parallel
  fullyParallel: true,

  // Fail the build on CI if you accidentally left test.only in the source code.
  forbidOnly: !!process.env.CI,

  // Retry on CI only
  retries: process.env.CI ? 2 : 0,

  // Opt out of parallel tests on CI if needed (can sometimes help with resource contention)
  // workers: process.env.CI ? 1 : undefined,

  // Reporter to use. See https://playwright.dev/docs/test-reporters
  reporter: 'html', // Generates a nice HTML report

  // Shared settings for all the projects below. See https://playwright.dev/docs/api/class-testoptions.
  use: {
    // Base URL to use in actions like `await page.goto('/')`
    baseURL: BASE_URL,

    // Collect trace when retrying the failed test. See https://playwright.dev/docs/trace-viewer
    trace: 'on-first-retry',

    // Viewport size
    // viewport: { width: 1280, height: 720 },
  },

  // Configure projects for major browsers
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },

    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },

    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] }, // Crucial for macOS testing
    },

    /* Test against mobile viewports. */
    // {
    //   name: 'Mobile Chrome',
    //   use: { ...devices['Pixel 5'] },
    // },
    // {
    //   name: 'Mobile Safari',
    //   use: { ...devices['iPhone 12'] },
    // },

    /* Test against branded browsers. */
    // {
    //   name: 'Microsoft Edge',
    //   use: { ...devices['Desktop Edge'], channel: 'msedge' },
    // },
    // {
    //   name: 'Google Chrome',
    //   use: { ...devices['Desktop Chrome'], channel: 'chrome' },
    // },
  ],

  // Run your local dev server before starting the tests
  // Assumes the frontend dev server runs on port 5173
  webServer: {
    command: 'npm run dev --prefix frontend', // Command to start frontend dev server
    url: BASE_URL, // URL to poll to ensure server is ready
    reuseExistingServer: !process.env.CI, // Reuse server locally, start fresh on CI
    timeout: 120 * 1000, // Increase timeout for server start (120s)
    stdout: 'pipe', // Pipe stdout for logging
    stderr: 'pipe', // Pipe stderr for logging
  },
});
