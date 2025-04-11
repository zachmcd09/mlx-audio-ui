import { test, expect, Page } from '@playwright/test';

// Reusable function to check audio element state
async function checkAudioPlayingState(page: Page, audioLocator = page.locator('audio')) {
  // Wait briefly for state propagation if needed, prefer event-based waits when possible
  await page.waitForTimeout(500); // Use sparingly

  const audioElementState = await audioLocator.evaluate(element => {
    const audioEl = element as HTMLAudioElement;
    return {
      paused: audioEl.paused,
      currentTime: audioEl.currentTime,
      readyState: audioEl.readyState,
      ended: audioEl.ended,
    };
  });

  expect(audioElementState.paused, 'Audio should not be paused').toBe(false);
  expect(audioElementState.currentTime, 'Current time should be greater than 0').toBeGreaterThan(0);
  expect(audioElementState.readyState, 'Ready state should indicate data available').toBeGreaterThanOrEqual(HTMLMediaElement.HAVE_CURRENT_DATA);
  expect(audioElementState.ended, 'Audio should not have ended immediately').toBe(false);
}


test.describe('TTS Happy Path', () => {
  test('should generate and start playing audio for valid text input', async ({ page }) => {
    // 1. Navigate to the app
    await page.goto('/');

    // 2. Locate key elements (use data-testid attributes if possible for robustness)
    const textInput = page.getByRole('textbox'); // Adjust selector if needed
    const playButton = page.getByRole('button', { name: /play/i }); // Adjust selector if needed
    const statusBar = page.getByTestId('status-bar'); // Assuming a data-testid for status
    const audioLocator = page.locator('audio'); // Assuming a single audio element

    // 3. Enter text
    const inputText = 'This is a test of the text-to-speech system.';
    await textInput.fill(inputText);
    expect(textInput).toHaveValue(inputText);

    // 4. Click Play
    // Start waiting for the network response *before* clicking
    const ttsResponsePromise = page.waitForResponse(
        response => response.url().includes('/tts') && response.status() === 200, // Corrected endpoint URL
        { timeout: 30000 } // Increase timeout for potentially slow TTS
    );
    await playButton.click();

    // 5. Wait for UI state changes & Network response
    // Expect status to show "Buffering" (or similar)
    await expect(statusBar).toContainText(/buffering/i, { timeout: 5000 });

    // Wait for the TTS network request to complete successfully
    const ttsResponse = await ttsResponsePromise;
    expect(ttsResponse.ok()).toBeTruthy();
    // The backend currently returns JSON with a filename, not raw audio/pcm
    // expect(ttsResponse.headers()['content-type']).toContain('audio/pcm');
    // We need to check the JSON response body instead
    const responseBody = await ttsResponse.json();
    expect(responseBody).toHaveProperty('filename');
    expect(responseBody.filename).toMatch(/tts_.*\.wav/);


    // Expect status to show "Playing" (or similar) after buffering/network response
    await expect(statusBar).toContainText(/playing/i, { timeout: 15000 }); // Allow time for buffering

    // 6. Assert Audio Element State
    await checkAudioPlayingState(page, audioLocator);

    // Optional: Check if play button changed to pause button
    // const pauseButton = page.getByRole('button', { name: /pause/i });
    // await expect(pauseButton).toBeVisible();
  });

  // Add more happy path variations if needed (e.g., longer text)
});
