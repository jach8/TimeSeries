"""Package structure verification script."""

import os
import sys
from pathlib import Path
from typing import List, Dict


class PackageChecker:
    """Check package structure and files."""
    
    def __init__(self):
        """Initialize checker with package root directory."""
        self.root = Path(__file__).parent
        self.issues: List[str] = []
        
    def check_core_files(self) -> bool:
        """Check if all required core files exist."""
        required_files = [
            'setup.py',
            'pyproject.toml',
            'requirements.txt',
            'requirements-dev.txt',
            'README.md',
            'README_DEV.md',
            'LICENSE',
            'MANIFEST.in',
            'pytest.ini',
        ]
        
        missing = []
        for file in required_files:
            if not (self.root / file).exists():
                missing.append(file)
        
        if missing:
            self.issues.append(f"Missing core files: {', '.join(missing)}")
            return False
        return True

    def check_package_structure(self) -> bool:
        """Check package directory structure."""
        required_dirs = [
            'timeseriestools',
            'timeseriestools/utils',
            'timeseriestools/test_data',
            'tests',
            'examples'
        ]
        
        missing = []
        for dir_path in required_dirs:
            if not (self.root / dir_path).is_dir():
                missing.append(dir_path)
        
        if missing:
            self.issues.append(f"Missing directories: {', '.join(missing)}")
            return False
        return True

    def check_module_files(self) -> bool:
        """Check if all Python module files exist."""
        required_modules = {
            'timeseriestools': [
                '__init__.py',
                'analyze.py',
                'causality.py',
                'correlation.py',
                'data.py',
                'stationarity.py'
            ],
            'timeseriestools/utils': [
                '__init__.py',
                'granger.py'
            ],
            'tests': [
                'conftest.py',
                'test_analyze.py',
                'test_causality.py',
                'test_correlation.py',
                'test_data.py',
                'test_stationarity.py'
            ]
        }
        
        missing: Dict[str, List[str]] = {}
        for dir_path, files in required_modules.items():
            dir_missing = []
            for file in files:
                if not (self.root / dir_path / file).is_file():
                    dir_missing.append(file)
            if dir_missing:
                missing[dir_path] = dir_missing
        
        if missing:
            for dir_path, files in missing.items():
                self.issues.append(f"Missing in {dir_path}: {', '.join(files)}")
            return False
        return True

    def check_installation_files(self) -> bool:
        """Check if installation and setup files exist."""
        required_files = [
            'install.sh',
            'install.bat',
            'setup_fresh.py',
            'verify_install.py',
            'minimal_test.py'
        ]
        
        missing = []
        for file in required_files:
            if not (self.root / file).exists():
                missing.append(file)
        
        if missing:
            self.issues.append(f"Missing installation files: {', '.join(missing)}")
            return False
        return True

    def check_documentation(self) -> bool:
        """Check if documentation files exist."""
        required_docs = [
            'README.md',
            'README_DEV.md',
            'LICENSE',
            'CHANGELOG.md',
            'examples/README.md'
        ]
        
        missing = []
        for doc in required_docs:
            if not (self.root / doc).exists():
                missing.append(doc)
        
        if missing:
            self.issues.append(f"Missing documentation: {', '.join(missing)}")
            return False
        return True

    def run_all_checks(self) -> bool:
        """Run all package structure checks."""
        print("Checking TimeSeriesTools package structure...\n")
        
        checks = [
            (self.check_core_files, "Core files"),
            (self.check_package_structure, "Package structure"),
            (self.check_module_files, "Module files"),
            (self.check_installation_files, "Installation files"),
            (self.check_documentation, "Documentation")
        ]
        
        all_passed = True
        for check_func, name in checks:
            if check_func():
                print(f"✓ {name} OK")
            else:
                print(f"✗ {name} FAILED")
                all_passed = False
        
        if not all_passed:
            print("\nIssues found:")
            for issue in self.issues:
                print(f"- {issue}")
            print("\nPlease fix these issues before proceeding.")
        else:
            print("\n✨ Package structure verified successfully!")
            print("\nYou can now:")
            print("1. Install the package:")
            print("   ./install.sh  # or install.bat on Windows")
            print("2. Run the tests:")
            print("   pytest tests/")
            print("3. Try the examples in examples/")
        
        return all_passed


if __name__ == "__main__":
    checker = PackageChecker()
    success = checker.run_all_checks()
    sys.exit(0 if success else 1)