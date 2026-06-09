# PyTorch-SNN-to-Hardware

A modular PyTorch farmework for training Convolution Spiking Neural Networks (CSNNs) on event-based neuromorphic datasets (N-MNIST, CIFAR10-DVS, DVS Gesture).

The primary goal of this project is to train SNNs and export **Fake-Quantized INT8 weights**, ensuring the models are strictly formatted and ready for physical deployment on hardware accelerators like **FPGA** and **SpiNNaker**.

---

## Installation

This project uses `uv` for dependency management. Ensure you have Python 3.10+ and a CUDA environment set up could be required.

1. Clone the repository:
```bash
git clone <the-repo-url>
cd snn_learn
```

2. Install dependencies and setup the virtual environment:
```bash
uv sync
```

---

## Usage

The main entry point for training and evaluation is main.py. The script automatically handles caching, training, checkpointing and final weight quantization.

### Basic training run

```bash
uv run main.py --dataset nmnist --epochs 20 --batch_size 64 --Time 16
```

### Available Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dataset` | `str` | `nmnist` | Dataset to use. Choices: `nmnist`, `cifar10`, `dvs_gesture`. |
| `--epochs` | `int` | `20` | Total number of training epochs. |
| `--batch_size` | `int` | `64` | Number of samples per batch. |
| `--lr` | `float`| `1e-3` | Learning rate for the AdamW optimizer. |
| `--Time` | `int` | `16` | Number of time steps (bins) for the Spiking Neural Network. |
| `--save_dir` | `str` | `./saved_models` | Directory to save checkpoints and final quantized weights. |
| `--use_wandb` | `flag` | `False` | Add this flag to log metrics to Weights & Biases. |
| `--wandb_project`| `str` | `snn_learn` | Target WandB project name (if `--use_wandb` is set). |

### Output Files

After a successful run, your `--save_dir` will contain:
- `checkpoint_latest.pth`: State dict to resume training.
- `model_best.pth`: Weights of the model with the highest test accuracy.
- `<dataset>_base.pth`: Standard float32 PyTorch weights.
- `<dataset>_quantized.pth`: The final **INT8 weights** ready for hardware deployment.

---

The repository includes a comprehensive test suite to ensure the SNN's dynamics, hardware quantization constraints, and deterministic checkpoints remain intact.

Run the test using `pytest`:
```bash
uv run pytest
```

---

## Project Structure

- `main.py`: Orchestrator script for the training loop.
- `src/models/`: Network architectures for each dataset.
- `src/data_loaders/`: Tonic-based data pipelines with disk caching.
- `src/engine/`: Core logic (training steps, checkpointing, INT8 quantization).
- `src/tests/`: Pytest suite for model validation and hardware constraints.