# MLX Audio UI

*A web interface for real-time text-to-speech using MLX.*

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
<!-- Add other badges here (Build Status, Python Version, etc.) once CI/CD is set up -->

## Description

**MLX Audio UI** provides an easy-to-use web interface for experimenting with text-to-speech (TTS) models accelerated by Apple's MLX framework. It features a Python/MLX backend for audio generation and a React/TypeScript frontend for user interaction, delivering real-time streaming audio playback directly in your browser.

This project consists of:
*   A core Python package (`mlx_audio`) containing MLX implementations of TTS models (like Bark, Kokoro) and potentially audio codecs.
*   A Flask server (`app.py`) that exposes the TTS functionality via an API.
*   A modern React/TypeScript frontend (`frontend`) providing the user interface and handling audio streaming playback.

## Visuals

*(Placeholder for a screenshot or GIF of the UI)*
![MLX Audio UI Screenshot](placeholder.png)

## Table of Contents

*   [Features](#features)
*   [Project Structure](#project-structure)
*   [Prerequisites](#prerequisites)
*   [Installation](#installation)
*   [Usage](#usage)
*   [Configuration](#configuration)
*   [Examples](#examples)
*   [Technologies Used](#technologies-used)
*   [Contributing](#contributing)
*   [Code of Conduct](#code-of-conduct)
*   [Support](#support)
*   [License](#license)

## Features

*   **Real-time Text-to-Speech:** Generate speech from text using MLX-accelerated models.
*   **Web-Based Interface:** Easy-to-use UI accessible via a web browser.
*   **Streaming Audio Playback:** Hear the generated audio almost instantly thanks to streaming via the Web Audio API.
*   **Model Support:** Leverages models implemented in the `mlx_audio` package (e.g., Bark, Kokoro).
*   **Adjustable Playback Speed:** Control the speed of the audio playback.
*   **Clear Status Indicators:** See the current state (Idle, Buffering, Playing, Paused, Error).
*   **(Future):** Potential for exploring audio codecs, speech-to-speech, and other audio tasks.

## Project Structure

```
.
├── app.py              # Flask backend server
├── frontend/           # React/TypeScript frontend SPA
├── mlx_audio/          # Core MLX audio processing library (Python)
├── examples/           # Example scripts and projects
├── requirements.txt    # Backend Python dependencies
├── setup.py            # Python package setup for mlx_audio
├── LICENSE             # Apache 2.0 License file
└── README.md           # This file
```

## Prerequisites

*   **Python:** 3.10.14 or higher recommended.
*   **Pip:** Python package installer (usually comes with Python).
*   **Node.js:** LTS version recommended (includes npm). Alternatively, use `yarn`.
*   **Operating System:** macOS with Apple Silicon (M1/M2/M3) is required for MLX acceleration.
*   **Git:** For cloning the repository.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/mlx-audio-ui.git
    cd mlx-audio-ui
    ```

2.  **Backend Setup:**
    *   Create and activate a Python virtual environment:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        # On Windows use `venv\Scripts\activate`
        ```
    *   Install Python dependencies:
        ```bash
        pip install -r requirements.txt
        # Or, for development including mlx_audio package:
        # pip install -e .
        ```
    *   *(Note: Model weights might be downloaded automatically on first run. Check specific model documentation within `mlx_audio` if needed.)*

3.  **Frontend Setup:**
    *   Navigate to the frontend directory:
        ```bash
        cd frontend
        ```
    *   Install Node.js dependencies:
        ```bash
        npm install
        ```
        or
        ```bash
        yarn install
        ```

## Usage

1.  **Start the Backend Server:**
    *   Ensure your virtual environment is activated (`source venv/bin/activate`).
    *   From the **root** project directory (`mlx-audio-ui`), run the Flask app:
        ```bash
        flask run
        ```
        or
        ```bash
        python app.py
        ```
    *   The backend will typically start on `http://127.0.0.1:5000`.

2.  **Start the Frontend Development Server:**
    *   In a **new terminal**, navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   Run the Vite development server:
        ```bash
        npm run dev
        ```
        or
        ```bash
        yarn dev
        ```
    *   The frontend will typically start on `http://localhost:5173` (check terminal output).

3.  **Access the UI:**
    *   Open your web browser and navigate to the frontend URL (e.g., `http://localhost:5173`).
    *   Paste your desired text into the input area.
    *   Click the "Play" button to start audio synthesis and playback.

## Configuration

*(Placeholder: Mention any key configuration options here, e.g., environment variables for model selection, if applicable.)*

Currently, configuration options (like model selection) might need to be adjusted directly in the source code (`app.py` or `mlx_audio` defaults).

## Examples

The `examples/` directory contains scripts and projects demonstrating how to use the `mlx_audio` library or the application components.

*   **`bible-audiobook`:** An example project for generating an audiobook version of the Bible. See `examples/bible-audiobook/README.md` for details.

## Technologies Used

*   **Backend:** Python, MLX, Flask
*   **Frontend:** React, TypeScript, Vite, Tailwind CSS, Zustand, Web Audio API
*   **Build/Environment:** Node.js, Pip

## Contributing

Contributions are welcome! Please see `CONTRIBUTING.md` for details on how to set up your development environment, coding standards, testing procedures, and the pull request process.

*(Note: `CONTRIBUTING.md` needs to be created)*

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.

## Code of Conduct

This project adheres to a Code of Conduct. Please see `CODE_OF_CONDUCT.md` for details.

*(Note: `CODE_OF_CONDUCT.md` needs to be created, likely adopting the Contributor Covenant)*

## Support

Please use the [GitHub Issue Tracker](https://github.com/your-username/mlx-audio-ui/issues) for bug reports, feature requests, and questions.

## License

Distributed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for more information.
