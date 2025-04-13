#!/usr/bin/env python3
"""
Setup script for isolated TTS model testing environment.

This script creates a dedicated virtual environment with pinned dependencies
and sets up isolation between different model tests to avoid cross-contamination.
"""
import os
import sys
import subprocess
import argparse
import platform
import json
from pathlib import Path


# Configuration
DEFAULT_VENV_DIR = "tts-debug-venv"
DEFAULT_CACHE_DIR = "model_cache"
REQUIREMENTS = [
    "mlx",
    "mlx-audio",
    "flask",
    "flask-cors",
    "huggingface_hub",
    "safetensors",
    "numpy",
]


def run_command(cmd, check=True):
    """Run a command with proper error handling."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        if check:
            sys.exit(1)
        return None


def create_virtual_environment(venv_dir):
    """Create a new virtual environment."""
    venv_path = Path(venv_dir)
    if venv_path.exists():
        print(f"Virtual environment already exists at {venv_path}")
        return str(venv_path)
    
    print(f"Creating virtual environment at {venv_path}")
    run_command([sys.executable, "-m", "venv", str(venv_path)])
    
    # Get the path to the Python executable in the virtual environment
    if platform.system() == "Windows":
        python_path = str(venv_path / "Scripts" / "python.exe")
    else:
        python_path = str(venv_path / "bin" / "python")
    
    # Upgrade pip
    run_command([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    
    return str(venv_path)


def install_dependencies(venv_dir):
    """Install dependencies in the virtual environment."""
    if platform.system() == "Windows":
        python_path = str(Path(venv_dir) / "Scripts" / "python.exe")
        pip_path = str(Path(venv_dir) / "Scripts" / "pip.exe")
    else:
        python_path = str(Path(venv_dir) / "bin" / "python")
        pip_path = str(Path(venv_dir) / "bin" / "pip")
    
    # Install each requirement
    for req in REQUIREMENTS:
        print(f"Installing {req}...")
        run_command([pip_path, "install", req])
    
    # Get the versions of installed packages
    pip_freeze = run_command([pip_path, "freeze"])
    
    # Save the dependencies to a requirements file
    with open("debug_requirements.txt", "w") as f:
        f.write(pip_freeze)
    
    print(f"Saved installed package versions to debug_requirements.txt")
    
    # Log versions of key packages
    print("\nInstalled dependency versions:")
    for req in REQUIREMENTS:
        for line in pip_freeze.split("\n"):
            if line.lower().startswith(req.lower()):
                print(f"  {line}")
                break


def create_cache_directory(cache_dir):
    """Create a directory for caching model files."""
    cache_path = Path(cache_dir)
    if not cache_path.exists():
        print(f"Creating model cache directory at {cache_path}")
        cache_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each model
    (cache_path / "kokoro").mkdir(exist_ok=True)
    (cache_path / "bark").mkdir(exist_ok=True)
    (cache_path / "encodec").mkdir(exist_ok=True)
    
    return str(cache_path)


def setup_environment_variables(venv_dir, cache_dir):
    """Setup environment variables for the testing environment."""
    env_vars = {
        "TRANSFORMERS_CACHE": str(Path(cache_dir) / "transformers"),
        "HF_HOME": str(Path(cache_dir) / "huggingface"),
        "MLX_AUDIO_CACHE": str(Path(cache_dir) / "mlx_audio"),
        "PYTHONPATH": os.getcwd(),  # Add current directory to Python path
    }
    
    # Create activation script for different shells
    if platform.system() == "Windows":
        # Windows batch file
        with open(Path(venv_dir) / "Scripts" / "activate_debug.bat", "w") as f:
            for var, value in env_vars.items():
                f.write(f"SET {var}={value}\n")
        
        print(f"Created environment setup script at {venv_dir}/Scripts/activate_debug.bat")
    else:
        # Bash script
        with open(Path(venv_dir) / "bin" / "activate_debug", "w") as f:
            f.write("#!/bin/bash\n")
            for var, value in env_vars.items():
                f.write(f"export {var}={value}\n")
        
        # Make it executable
        os.chmod(Path(venv_dir) / "bin" / "activate_debug", 0o755)
        print(f"Created environment setup script at {venv_dir}/bin/activate_debug")
    
    # Create a JSON file with environment info
    env_info = {
        "venv_dir": str(Path(venv_dir).absolute()),
        "cache_dir": str(Path(cache_dir).absolute()),
        "environment_variables": env_vars,
        "creation_time": str(subprocess.check_output(["date"]).decode().strip())
    }
    
    with open("debug_env_info.json", "w") as f:
        json.dump(env_info, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Setup isolated TTS model testing environment")
    parser.add_argument("--venv-dir", default=DEFAULT_VENV_DIR, 
                        help=f"Virtual environment directory (default: {DEFAULT_VENV_DIR})")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR,
                        help=f"Model cache directory (default: {DEFAULT_CACHE_DIR})")
    args = parser.parse_args()
    
    print("Setting up isolated TTS model testing environment")
    print("================================================")
    
    venv_dir = create_virtual_environment(args.venv_dir)
    install_dependencies(venv_dir)
    cache_dir = create_cache_directory(args.cache_dir)
    setup_environment_variables(venv_dir, cache_dir)
    
    print("\nEnvironment setup complete!")
    if platform.system() == "Windows":
        print(f"To activate, run: {venv_dir}\\Scripts\\activate.bat")
        print(f"Then run: {venv_dir}\\Scripts\\activate_debug.bat")
    else:
        print(f"To activate, run: source {venv_dir}/bin/activate")
        print(f"Then run: source {venv_dir}/bin/activate_debug")
    
    print("\nNext steps:")
    print("1. Activate the virtual environment")
    print("2. Run the model analysis scripts")
    print("3. Run incremental model testing")


if __name__ == "__main__":
    main()
