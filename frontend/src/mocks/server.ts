// frontend/src/mocks/server.ts
// Used for setting up MSW in Node.js environments (like Vitest)
import { setupServer } from 'msw/node';
import { handlers } from './handlers';

// This configures a request mocking server with the given request handlers.
export const server = setupServer(...handlers);
