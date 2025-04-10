import React from 'react';
import { useAppStore } from '../../store';

function TextInput() {
  // Select the inputText state and the action to update it
  const inputText = useAppStore((state) => state.inputText);
  const setInputText = useAppStore((state) => state.setInputText);

  const handleChange = (event: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputText(event.target.value);
  };

  return (
    <div className="mb-4">
      <label htmlFor="text-input" className="block text-sm font-medium text-gray-700 mb-1">
        Paste your text here:
      </label>
      <textarea
        id="text-input"
        rows={10}
        className="w-full p-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
        placeholder="Enter text to be read aloud..."
        value={inputText}
        onChange={handleChange}
      />
    </div>
  );
}

export default TextInput;
