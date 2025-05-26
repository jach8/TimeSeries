"""Quick setup script for TimeSeriesTools."""

import os
import sys
import subprocess
import platform


def run_command(cmd: str) -> None:
    """Run a shell command and print output."""
    print(f"\nExecuting: {cmd}")
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)


def main():
    """Run setup process."""
    print("Setting up TimeSeriesTools package...")
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Create virtual environment
    if not os.path.exists('venv'):
        print("\nCreating virtual environment...")
        run_command('python -m venv venv')
    
    # Activate virtual environment and install dependencies
    if platform.system() == 'Windows':
        activate_cmd = 'call venv\\Scripts\\activate.bat'
    else:
        activate_cmd = 'source venv/bin/activate'
    
    # Create test data directory
    os.makedirs('timeseriestools/test_data', exist_ok=True)
    
    # Install package and generate data
    commands = [
        activate_cmd,
        'pip install -r requirements.txt',
        'python generate_data.py',
        'pip install -e .',
        'python install_test.py'
    ]
    
    # Run commands
    for cmd in commands:
        run_command(cmd)
    
    print("""
Setup complete! ✨

You can now:
1. Run the package tests:
   pytest tests/

2. Try the examples:
   cd examples
   python package_test.py

3. Check the documentation in README.md
""")


if __name__ == "__main__":
    main()