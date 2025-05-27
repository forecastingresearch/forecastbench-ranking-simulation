# Ranking Simulation

A simulation environment for evaluating different ranking methods for `ForecastBench`.

## Overview

The simulation uses 2024 July forecasting round which had 111 forecasters (humans & LLMs) answering 471 questions. Each forecaster provided a forecast for each question, providing a very clean dataset that can be used to construct a simulation environment.

This project currently compares three ranking methods:
- **Brier Score**: Standard accuracy metric for probabilistic predictions
- **Brier Skill Score (BSS)**: Performance relative to a reference model
- **Peer Score**: Performance relative to the average of all models

The simulation tests how robust these rankings are across simulations.

## Project Structure
```
ranking-simulation/
├── src/                    # Main source code
│   └── ranking_sim.py      # Core simulation functions
├── tests/                  # Unit tests
│   └── test_ranking_sim.py # Test suite
├── notebooks/              # Jupyter notebooks for development
│   └── dev.ipynb           # Development playground
├── data/                   # Data directory (contents not tracked)
│   ├── raw/                # Input data
│   ├── processed/          # Processed data
│   └── results/            # Simulation outputs
└── run_simulation.py       # Main script to run simulations
```

## Usage

Run the simulation:
```bash
python run_simulation.py
```

Run tests:
```bash
cd ./tests/
pytest
```

## Requirements

- Python 3.7+
- pandas
- numpy
- pytest (for running tests)
