// frontend/src/setupTests.ts
// This file runs before all tests, perfect for global setup.
import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './mocks/server';
// Optional: Import testing-library matchers like @testing-library/jest-dom/vitest
import '@testing-library/jest-dom/vitest'; // Uncommented this line

// Establish API mocking before all tests.
// Use { onUnhandledRequest: 'error' } to catch requests not handled by MSW.
// Change to 'warn' or 'bypass' if needed during development.
beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));

// Reset any request handlers that we may add during the tests,
// so they don't affect other tests.
afterEach(() => server.resetHandlers());

// Clean up after the tests are finished.
afterAll(() => server.close());

// Add any other global setup here, e.g., mocking global objects if necessary
// Example: Mocking matchMedia for components that use it
// Object.defineProperty(window, 'matchMedia', {
//   writable: true,
//   value: vi.fn().mockImplementation(query => ({
//     matches: false,
//     media: query,
//     onchange: null,
//     addListener: vi.fn(), // deprecated
//     removeListener: vi.fn(), // deprecated
//     addEventListener: vi.fn(),
//     removeEventListener: vi.fn(),
//     dispatchEvent: vi.fn(),
//   })),
// });
