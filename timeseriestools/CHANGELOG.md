# Changelog

All notable changes to TimeSeriesTools will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-05-25

### Added
- Initial release of TimeSeriesTools
- Core functionality for time series analysis:
  - Stationarity testing with multiple methods
  - Granger causality analysis
  - Correlation analysis with automatic stationarity checks
  - Feature decomposition using PCA
- Built-in test data and examples
- Comprehensive test suite
- Documentation and example notebooks
- Development setup scripts for both Windows and Unix systems

### Core Features
- `Analyze` class for high-level interface
- `CausalityAnalyzer` for causality testing
- `AnalyzeCorrelation` for correlation analysis
- `StationaryTests` for stationarity checking
- Data loading utilities and test datasets

### Tests
- Unit tests for all major components
- Integration tests for end-to-end workflows
- Test data generation utilities

### Documentation
- API documentation with Google-style docstrings
- Example notebooks demonstrating usage
- Development guide and contribution guidelines
- Type hints throughout the codebase

## [Unreleased]

### Planned Features
- Additional causality tests
- More visualization tools
- Enhanced documentation
- Performance optimizations
- Additional example notebooks
- Support for more data formats
- Extended test coverage

### Upcoming Changes
- Improved error handling
- Better type hint coverage
- Additional visualization options
- More example datasets
- Performance enhancements for large datasets

## Release Process

### Version Numbers
- MAJOR version for incompatible API changes
- MINOR version for added functionality in backward-compatible manner
- PATCH version for backward-compatible bug fixes

### Changelog Entry Format
```markdown
## [x.y.z] - YYYY-MM-DD

### Added
- New features

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security vulnerability fixes
```

### Release Steps
1. Update version number
2. Update changelog
3. Run test suite
4. Create release branch
5. Tag release
6. Build and publish
7. Merge to main branch