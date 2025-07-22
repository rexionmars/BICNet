# Contributing to BICNet

## Development Setup

1. Fork and clone the repository
2. Install development dependencies:
   ```bash
   uv sync --dev
   ```

## Code Structure

```
bicnet/
├── core/                 # Core neural systems
│   ├── brain/           # Brain structures and neurons
│   ├── memory/          # Memory systems  
│   ├── genes/           # Gene regulatory networks
│   └── consciousness/   # Consciousness integration
├── modules/             # Specialized modules
│   ├── visualization/   # Plotting and visualization
│   ├── analysis/        # Data analysis tools
│   └── simulation/      # Simulation frameworks
├── data/                # Data storage
│   ├── dna/            # DNA sequences and analysis
│   ├── logs/           # Execution logs
│   └── results/        # Simulation results
├── examples/            # Usage examples
├── tests/              # Test suites
└── docs/               # Documentation

## Coding Standards

- Follow PEP 8 style guidelines
- Use type hints where possible
- Document functions with docstrings
- Write unit tests for new features
- Keep functions focused and modular

## Adding New Features

1. Create feature branch from master
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## Testing

```bash
# Run all tests
uv run pytest tests/

# Run specific test
uv run pytest tests/test_brain.py
```

## Documentation

Update relevant documentation in:
- Docstrings for functions/classes
- API.md for new modules
- README.md for significant changes