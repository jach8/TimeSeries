"""Fresh installation script for TimeSeriesTools."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str) -> None:
    """Run a shell command and handle errors."""
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def clean_install():
    """Remove previous installation artifacts and install fresh."""
    print("Starting fresh installation of TimeSeriesTools...")

    # Get current directory
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    
    # Remove previous build artifacts
    dirs_to_remove = ['build', 'dist', '*.egg-info']
    for pattern in dirs_to_remove:
        for item in current_dir.glob(pattern):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"Removed {item}")
    
    # Remove __pycache__ directories
    for cache_dir in current_dir.rglob('__pycache__'):
        shutil.rmtree(cache_dir)
        print(f"Removed {cache_dir}")
    
    # Create test_data directory if it doesn't exist
    test_data_dir = current_dir / 'timeseriestools' / 'test_data'
    test_data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created/verified test_data directory: {test_data_dir}")
    
    # Generate test data
    print("\nGenerating test data...")
    run_command('python generate_data.py')
    
    # Install package in editable mode
    print("\nInstalling package...")
    run_command('pip install -e .')
    
    # Run installation test
    print("\nTesting installation...")
    run_command('python install_test.py')


if __name__ == "__main__":
    clean_install()