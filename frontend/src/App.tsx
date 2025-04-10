import React from 'react'; // Import React if using context later
import './App.css';
import TextInput from './components/Input/TextInput';
import PlayerControls from './components/Player/PlayerControls';
import StatusBar from './components/StatusBar/StatusBar';
import { useAudioStreamer } from './hooks/useAudioStreamer'; // Import the hook

function App() {
  // Instantiate the audio streamer hook
  const audioControls = useAudioStreamer();

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 flex flex-col">
      <header className="bg-blue-600 text-white p-4 shadow-md flex-shrink-0">
        <h1 className="text-2xl font-bold text-center">MLX Audio UI</h1>
      </header>
      <StatusBar /> {/* Add StatusBar below header */}
      <main className="flex-grow p-4 container mx-auto max-w-3xl">
        {/* Input Area */}
        <TextInput />

        {/* Player Controls Area - includes Progress Indicator internally */}
        {/* Pass controls down as props */}
        <PlayerControls controls={audioControls} />

      </main>
      <footer className="text-center p-4 text-gray-600 text-sm flex-shrink-0 mt-auto"> {/* Added mt-auto to push footer down */}
        Open Source TTS Frontend - Powered by MLX & React
      </footer>
    </div>
  );
}

export default App;
