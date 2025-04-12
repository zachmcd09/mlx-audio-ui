import { useRef, useCallback, useState, useEffect } from 'react'; // Import useEffect
import { useAppStore, PlaybackState } from '../store';
import { fetchTTSAudioStream } from '../services/ttsApi';

// Define the structure for queued audio buffers
interface DecodedAudioChunk {
  index: number;
  buffer: AudioBuffer;
}

// Define the parameters expected by the hook's control functions
interface PlaybackParams {
  text: string;
  voice: string | null;
  speed: number;
}

// Define the interface for the hook's return value (controls)
export interface AudioStreamerControls {
  play: (params: PlaybackParams) => Promise<void>;
  pause: () => void;
  resume: () => Promise<void>; // Basic resume for now
  stop: () => void;
  adjustSpeed: (newSpeed: number) => void;
}

export function useAudioStreamer(): AudioStreamerControls {
  // Zustand state access and actions
  const {
    _setPlaybackState,
    _setErrorMessage,
    _setTotalChunks,
    _setCurrentChunkIndex,
  } = useAppStore.getState();
  const currentSpeed = useAppStore((state) => state.speed); // Read speed directly

  // Refs for managing audio context, nodes, queue, etc.
  const audioContextRef = useRef<AudioContext | null>(null);
  const currentSourceNodeRef = useRef<AudioBufferSourceNode | null>(null);
  const decodedQueueRef = useRef<DecodedAudioChunk[]>([]);
  const readerRef = useRef<ReadableStreamDefaultReader<Uint8Array> | null>(null);
  const nextTokenIndexRef = useRef<number>(0); // Tracks the index of the *next* chunk to play
  const isFetchingRef = useRef<boolean>(false);
  const currentPlaybackParamsRef = useRef<PlaybackParams | null>(null); // Store params for resume
  const leftoverBytesRef = useRef<Uint8Array | null>(null); // Buffer for leftover bytes

  // Internal state for the hook if needed (e.g., sample rate)
  const [sampleRate, setSampleRate] = useState<number>(24000); // Default, read from header

  // --- Helper Functions ---

  const getAudioContext = useCallback((): AudioContext => {
    if (!audioContextRef.current || audioContextRef.current.state === 'closed') {
      audioContextRef.current = new AudioContext();
      // Reset internal state when context is new/resumed
      nextTokenIndexRef.current = 0;
      decodedQueueRef.current = [];
      currentSourceNodeRef.current = null;
      console.log('AudioContext created or reopened.');
    }
    if (audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
      console.log('AudioContext resumed.');
    }
    return audioContextRef.current;
  }, []);

  const cleanup = useCallback(() => {
    console.log('Cleaning up audio resources...');
    readerRef.current?.cancel().catch(() => {}); // Cancel stream reading
    readerRef.current = null;
    leftoverBytesRef.current = null; // Clear leftover bytes on cleanup
    if (currentSourceNodeRef.current) {
      try {
        currentSourceNodeRef.current.onended = null; // Remove handler
        currentSourceNodeRef.current.stop();
      } catch (e) {
         console.warn("Error stopping source node:", e)
      }
      currentSourceNodeRef.current = null;
    }
    decodedQueueRef.current = [];
    nextTokenIndexRef.current = 0;
    isFetchingRef.current = false;
    currentPlaybackParamsRef.current = null;
    // Optionally close context if not needed immediately
    // audioContextRef.current?.close();
    // audioContextRef.current = null;
  }, []);

  const _decodeSentenceData = useCallback(
    async (bytes: Uint8Array): Promise<AudioBuffer | null> => {
      const ctx = getAudioContext();
      // TODO: Implement actual PCM decoding based on headers (Sample Rate, Bit Depth, Channels)
      // This is a placeholder assuming Float32 PCM for now
      // Example for 16-bit signed PCM:
      // const pcm16 = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);
      // const float32Data = new Float32Array(pcm16.length);
      // for (let i = 0; i < pcm16.length; i++) {
      //   float32Data[i] = pcm16[i] / 32768.0;
      // --- Correct 16-bit Signed PCM to Float32 Conversion ---
      // The calling loop (_fetchAndDecodeLoop) now ensures `bytes` has an even length.

      // Create an Int16Array view on the received bytes
      // Assumes little-endian byte order, which is common for PCM WAV. Adjust if needed.
      const pcm16Data = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.length / 2);

      // Create a Float32Array of the same length
      const float32Data = new Float32Array(pcm16Data.length);

      // Convert each 16-bit sample to a float between -1.0 and 1.0
      for (let i = 0; i < pcm16Data.length; i++) {
        float32Data[i] = pcm16Data[i] / 32768.0; // Divide by 2^15
      }
      // --- End Conversion ---

      const numSamples = float32Data.length;
      const numberOfChannels = 1; // TODO: Read from header if backend supports stereo

      if (numSamples === 0) {
          console.warn("Decoded chunk resulted in zero samples.");
          return null;
      }

      try {
        const audioBuffer = ctx.createBuffer(
          numberOfChannels,
          numSamples,
          sampleRate // Use state variable read from header
        );
        audioBuffer.copyToChannel(float32Data, 0); // Assuming mono
        return audioBuffer;
      } catch (error) {
        console.error('Error decoding audio data:', error);
        _setErrorMessage(`Audio decoding error: ${error}`);
        _setPlaybackState('Error');
        cleanup();
        return null;
      }
    },
    [getAudioContext, sampleRate, _setErrorMessage, _setPlaybackState, cleanup]
  );

  // Refs to hold the latest versions of callback functions, initialized to null
  const playBufferRef = useRef<((buffer: AudioBuffer, index: number) => void) | null>(null);
  const tryPlayNextRef = useRef<(() => void) | null>(null);


  // Define _handlePlaybackEnd first
  const _handlePlaybackEnd = useCallback(() => {
    console.log(`Chunk ${nextTokenIndexRef.current - 1} ended.`);
    currentSourceNodeRef.current = null; // Clear the ref for the ended node
    // Immediately try to play the next chunk from the queue via ref
    tryPlayNextRef.current?.();
  }, []); // No dependencies needed as it uses the ref

  // Define _playBuffer second (using useCallback)
  const _playBuffer = useCallback(
    (buffer: AudioBuffer, index: number) => {
      const ctx = getAudioContext();
      try {
        const sourceNode = ctx.createBufferSource();
        sourceNode.buffer = buffer;
        sourceNode.playbackRate.value = currentSpeed; // Apply current speed
        sourceNode.connect(ctx.destination);
        sourceNode.onended = _handlePlaybackEnd; // Set the end handler
        sourceNode.start(0); // Play immediately

        currentSourceNodeRef.current = sourceNode; // Store the currently playing node
        _setPlaybackState('Playing');
        _setCurrentChunkIndex(index); // Update store with the index of the chunk NOW playing
        console.log(`Playing chunk ${index} (Speed: ${currentSpeed}x)`);

      } catch (error) {
         console.error("Error playing buffer:", error);
         _setErrorMessage(`Playback error: ${error}`);
         _setPlaybackState('Error');
         cleanup();
      }
    },
    // Add _handlePlaybackEnd to dependencies
    [getAudioContext, currentSpeed, _handlePlaybackEnd, _setPlaybackState, _setCurrentChunkIndex, _setErrorMessage, cleanup]
  );


  // Define _tryPlayNextFromQueue third (using useCallback)
  const _tryPlayNextFromQueue = useCallback(() => {
    // Check if not already playing and queue has the *next* expected chunk
    if (
      !currentSourceNodeRef.current &&
      decodedQueueRef.current.length > 0 &&
      decodedQueueRef.current[0].index === nextTokenIndexRef.current
    ) {
      const chunkToPlay = decodedQueueRef.current.shift(); // Dequeue
      if (chunkToPlay) {
        // Use the ref to call the latest version of _playBuffer
        playBufferRef.current?.(chunkToPlay.buffer, chunkToPlay.index);
        nextTokenIndexRef.current++; // Increment expected index *after* starting playback
      }
    } else if (!currentSourceNodeRef.current && decodedQueueRef.current.length === 0 && !isFetchingRef.current) {
       // If nothing is playing, queue is empty, and fetching is done, we are finished
       console.log("Playback finished.");
       _setPlaybackState('Idle'); // Or 'Stopped'
       cleanup();
    }
    // Use playBufferRef.current in the function body if needed
  }, [_setPlaybackState, cleanup]); // Removed _playBuffer from deps

  // Update refs with the latest function definitions using useEffect
  useEffect(() => {
    // Explicitly cast types during assignment (though shouldn't be needed)
    playBufferRef.current = _playBuffer as (buffer: AudioBuffer, index: number) => void;
    tryPlayNextRef.current = _tryPlayNextFromQueue as () => void;
  }, [_playBuffer, _tryPlayNextFromQueue]); // Update refs when the memoized functions change


  const _fetchAndDecodeLoop = useCallback(
    async (reader: ReadableStreamDefaultReader<Uint8Array>) => {
      isFetchingRef.current = true;
      let fetchedIndex = 0; // Counter for decoded chunks
      leftoverBytesRef.current = null; // Ensure leftover buffer is clear at start

      try {
        while (true) {
          console.log('Waiting for next chunk...');
          const { done, value } = await reader.read();

          if (done) {
            console.log('Stream finished.');
            // Check for any remaining leftover byte after stream ends
            if (leftoverBytesRef.current && leftoverBytesRef.current.length > 0) {
              console.error(`Stream ended with leftover byte: ${leftoverBytesRef.current[0]}. Total stream length might be odd.`);
              // Optionally set an error state here if this is critical
              _setErrorMessage("Incomplete final audio sample received.");
              _setPlaybackState('Error');
            }
            leftoverBytesRef.current = null; // Clear it regardless
            isFetchingRef.current = false;
            // If the queue becomes empty after fetching is done, trigger final state check
            _tryPlayNextFromQueue();
            break;
          }

          if (value) {
            // Combine leftover bytes from previous chunk (if any) with the new chunk
            const combinedBytes: Uint8Array = new Uint8Array(
              (leftoverBytesRef.current ? leftoverBytesRef.current.length : 0) + value.length
            );
            if (leftoverBytesRef.current) {
              combinedBytes.set(leftoverBytesRef.current, 0);
            }
            combinedBytes.set(value, leftoverBytesRef.current ? leftoverBytesRef.current.length : 0);

            // Determine how many full 16-bit samples we have (even number of bytes)
            const bytesToProcessLength = Math.floor(combinedBytes.length / 2) * 2;
            let bytesToProcess: Uint8Array | null = null;

            if (bytesToProcessLength > 0) {
              bytesToProcess = combinedBytes.slice(0, bytesToProcessLength);
            }

            // Store the remaining odd byte (if any) for the next iteration
            if (combinedBytes.length % 2 !== 0) {
              leftoverBytesRef.current = combinedBytes.slice(bytesToProcessLength);
              // console.log(`Storing leftover byte: ${leftoverBytesRef.current[0]}`);
            } else {
              leftoverBytesRef.current = null; // No leftover byte
            }

            // Decode the processable bytes
            if (bytesToProcess) {
              // console.log(`Decoding ${bytesToProcess.length} bytes for chunk ${fetchedIndex}...`);
              const decodedBuffer = await _decodeSentenceData(bytesToProcess);
              if (decodedBuffer) {
                decodedQueueRef.current.push({ index: fetchedIndex, buffer: decodedBuffer });
                console.log(`Chunk ${fetchedIndex} decoded and queued.`);
                fetchedIndex++;
                // Try to play immediately if nothing is playing
                _tryPlayNextFromQueue();
              } else {
                 // Decoding error handled within _decodeSentenceData
                 // Clear leftover bytes as the stream is likely corrupted
                 leftoverBytesRef.current = null;
                 break; // Stop fetching on decode error
              }
            } else {
              // console.log("Received chunk resulted in 0 processable bytes after combining leftovers, waiting for more data.");
            }
          }
        }
      } catch (error) {
        console.error('Error reading stream:', error);
        _setErrorMessage(`Stream reading error: ${error}`);
        _setPlaybackState('Error');
        cleanup();
      } finally {
        isFetchingRef.current = false;
        reader.releaseLock(); // Release lock when done or on error
        console.log("Stream reader released.")
        // Ensure leftover bytes are cleared if loop exits unexpectedly
        if (useAppStore.getState().playbackState === 'Error') { // Check Zustand state directly
            leftoverBytesRef.current = null;
        }
      }
    },
    // Add _tryPlayNextFromQueue (the memoized version) to dependencies
    [_decodeSentenceData, _tryPlayNextFromQueue, _setErrorMessage, _setPlaybackState, cleanup]
  );

  // --- Control Functions Exposed by Hook ---

  const play = useCallback(
    async (params: PlaybackParams) => {
      console.log('Play requested:', params);
      // Ensure previous playback is stopped and resources are clean
      cleanup();
      _setPlaybackState('Buffering'); // Update Zustand state via action
      currentPlaybackParamsRef.current = params; // Store for potential resume

      try {
        const response = await fetchTTSAudioStream(params.text, params.voice, params.speed);

        // Read headers
        const totalChunksHeader = response.headers.get('X-Audio-Total-Chunks');
        const sampleRateHeader = response.headers.get('X-Audio-Sample-Rate');
        // TODO: Read other headers (channels, bit depth)

        const backendTotalChunks = totalChunksHeader ? parseInt(totalChunksHeader, 10) : 0;
        const backendSampleRate = sampleRateHeader ? parseInt(sampleRateHeader, 10) : 24000; // Default

        _setTotalChunks(backendTotalChunks);
        setSampleRate(backendSampleRate); // Set internal state for decoder
        console.log(`Received headers: TotalChunks=${backendTotalChunks}, SampleRate=${backendSampleRate}`);


        if (!response.body) {
          throw new Error('Response body is null.');
        }

        const reader = response.body.getReader();
        readerRef.current = reader; // Store reader for cancellation

        // Start the async loop to fetch, decode, and queue chunks
        _fetchAndDecodeLoop(reader);

      } catch (error) {
        console.error('Failed to start playback:', error);
        _setErrorMessage(`Failed to start playback: ${error}`);
        _setPlaybackState('Error');
        cleanup();
      }
    },
    [cleanup, _setPlaybackState, _setTotalChunks, _setErrorMessage, fetchTTSAudioStream, _fetchAndDecodeLoop]
  );

  const pause = useCallback(() => {
    console.log('Pause requested.');
    const currentState = useAppStore.getState().playbackState;
    if (currentState !== 'Playing') return;

    if (currentSourceNodeRef.current) {
      try {
         currentSourceNodeRef.current.stop(); // Stop current sound
         // Note: onended will fire, which calls _tryPlayNextFromQueue, but it won't play
         // because currentSourceNodeRef is immediately set to null below.
      } catch(e) {
         console.warn("Error stopping source node on pause:", e);
      }
      currentSourceNodeRef.current = null;
    }
    // Stop fetching more data
    readerRef.current?.cancel().catch(() => {});
    readerRef.current = null;
    isFetchingRef.current = false;

    // Clear the queue of upcoming buffers
    decodedQueueRef.current = [];

    _setPlaybackState('Paused');
    // Note: currentChunkIndex remains where it was
    console.log('Playback paused.');
  }, [_setPlaybackState]);

  const resume = useCallback(async () => {
    console.log('Resume requested.');
    const currentState = useAppStore.getState().playbackState;
    if (currentState !== 'Paused' || !currentPlaybackParamsRef.current) {
       console.warn("Cannot resume from state:", currentState);
       return;
    }

    // Basic Resume: Restart playback from the beginning using stored params
    // TODO: Implement smarter resume (e.g., starting from currentChunkIndex + 1)
    // This requires backend support to start streaming from a specific chunk index.
    console.log('Performing basic resume (restarting)...');
    await play(currentPlaybackParamsRef.current);

  }, [play]); // Depends on the play function

  const stop = useCallback(() => {
    console.log('Stop requested.');
    cleanup(); // Perform full cleanup
    _setPlaybackState('Stopped'); // Set final state
    _setCurrentChunkIndex(-1); // Reset index
    _setTotalChunks(0); // Reset total
    console.log('Playback stopped and resources cleaned.');
  }, [cleanup, _setPlaybackState, _setCurrentChunkIndex, _setTotalChunks]);

  const adjustSpeed = useCallback((newSpeed: number) => {
     // Update the speed for the currently playing node, if any
     if (currentSourceNodeRef.current) {
       currentSourceNodeRef.current.playbackRate.value = newSpeed;
       console.log(`Adjusted speed for current node to ${newSpeed}x`);
     }
     // The `currentSpeed` variable read from Zustand will ensure
     // subsequent nodes are created with the new speed in _playBuffer.
   }, []);


  // Return the control functions
  return {
    play,
    pause,
    resume,
    stop,
    adjustSpeed,
  };
}
