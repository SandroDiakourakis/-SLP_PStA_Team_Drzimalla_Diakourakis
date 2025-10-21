"""
Setup script for SAND Task 1 Pipeline
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="sand-task1",
    version="1.0.0",
    author="SAND Challenge Team",
    author_email="your.email@example.com",
    description="Local, cost-free pipeline for SAND Challenge Task 1 - Multi-Class Classification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/sand-task1",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "xgboost>=1.7.0",
        "lightgbm>=4.0.0",
        "scikit-learn>=1.3.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "opensmile>=2.4.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "joblib>=1.3.0",
        "openpyxl>=3.1.0",  # For Excel reading in prep.py
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
            "ipykernel>=6.23.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sandcli=sandcli.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["conf/*.yaml"],
    },
    zip_safe=False,
)