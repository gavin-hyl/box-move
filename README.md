# Box Move DQN

A Deep Q-Network (DQN) implementation for solving the Box Move environment, where an agent needs to move boxes from one zone to another.

## Project Structure

```none
box-move/
├── src/                    # Source code
│   ├── BoxMoveEnvGym.py   # Main environment implementation
│   ├── QNetTrain.py       # DQN agent training implementation
│   ├── QNet.py            # Neural network architecture
│   ├── Box.py             # Box class implementation
│   ├── BoxAction.py       # Action class implementation
│   ├── Constants.py       # Environment constants
│   ├── Benchmark.py       # Benchmarking utilities
│   └── __init__.py        # Package initialization
├── tests/                 # Test suite
│   ├── BoxMoveTest.py     # Environment tests
│   └── __init__.py        # Package initialization
├── train.py              # Main training script
└── requirements.txt      # Python dependencies
```

## Installation

- Clone the repository:

```bash
git clone https://github.com/yourusername/box-move.git
cd box-move
```

- Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To train a new DQN agent:

```bash
python train.py --mode train --timesteps 10000 --n_boxes 4 --horizon 30
```

Key arguments:

- `--mode`: Choose between 'train', 'evaluate', or 'benchmark'
- `--timesteps`: Total timesteps to train
- `--n_boxes`: Number of boxes in the environment
- `--horizon`: Episode horizon
- `--model_path`: Path to save/load the model (default: models/dqn_model.pt)

### Evaluation

To evaluate a trained model:

```bash
python train.py --mode evaluate --model_path models/dqn_model.pt
```

### Benchmarking

To benchmark against baseline policies:

```bash
python train.py --mode benchmark --n_boxes 6 --horizon 40
```

This will compare the DQN against random and greedy policies.

## Environment

The Box Move environment consists of two zones:

- Zone 0: Initial zone where boxes start
- Zone 1: Target zone where boxes need to be moved

The agent needs to learn to efficiently move boxes from Zone 0 to Zone 1 while adhering to physical constraints.

## Results

Training results, including learning curves and evaluation metrics, will be saved in a `results/` directory when generated.

## Testing

To run the test suite:

```bash
python -m pytest tests/
```