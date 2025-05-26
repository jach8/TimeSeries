"""Setup configuration for TimeSeriesTools package."""

from setuptools import setup, find_packages

setup(
    name="timeseriestools",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "statsmodels>=0.13.0",
        "scikit-learn>=0.24.0",
        "arch>=5.0.0",
        "pmdarima>=2.0.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "isort>=5.0",
            "mypy>=0.9",
            "flake8>=4.0",
        ],
    },
    author="TimeSeriesTools Contributors",
    author_email="contributors@timeseriestools.org",
    description="A comprehensive Python package for time series analysis",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/timeseriestools",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering",
    ],
    package_data={
        "timeseriestools": [
            "test_data/*.pkl",
            "test_data/*.csv",
        ],
    },
)