# Installation Guide

## Prerequisites
- Python 3.12+
- UV package manager (recommended) or pip

## Installation

### Using UV (Recommended)
```bash
# Clone repository
git clone <repository-url>
cd BICNet

# Install dependencies
uv sync

# Activate virtual environment  
uv run python -m bicnet
```

### Using pip
```bash
# Clone repository
git clone <repository-url>
cd BICNet

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .
```

## Dependencies

### Core Dependencies
- `numpy`: Numerical computations
- `matplotlib`: Plotting and visualization
- `scipy`: Scientific computing
- `torch`: Neural network operations
- `jax`: Accelerated computing

### Specialized Dependencies  
- `biopython`: DNA sequence analysis
- `gymnasium`: Reinforcement learning environments
- `robosuite`: Robotic simulation
- `pyqt5`, `pyqtgraph`: GUI components
- `opencv-python`: Computer vision
- `pandas`, `seaborn`: Data analysis

### Optional Dependencies
- `imageio`: Video generation
- `pygame`: Interactive simulations

## Verification
```bash
# Test installation
uv run python -c "import bicnet; print(bicnet.__version__)"

# Run example
uv run python bicnet/examples/main.py
```