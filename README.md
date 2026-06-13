# CIFAR-10 Image Classifier

A custom convolutional neural network built from scratch in PyTorch that achieves **91%+ accuracy** on the CIFAR-10 benchmark — competitive with published results for hand-designed architectures without residual connections.

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **91.01%** |
| Dataset | CIFAR-10 (50k train / 10k test) |
| Training Epochs | 300 |
| Batch Size | 256 |

## Architecture

The model (`CNN.py`) is a fully configurable deep CNN with the following default configuration:

- **5 convolutional blocks**, each with Conv2d → BatchNorm → ReLU → MaxPool
- Channels double at each block: 64 → 128 → 256 → 512 → 1024
- **4 fully connected layers** with dropout (p=0.5) between each
- Dynamically computed flattened size

```python
net = ImageNeuralNetwork(
    channels=64,
    layers=4,
    conv_blocks=5,
    num_classes=10,
    dropout_rate=0.5
)
```

## Training Details

Several techniques were combined to reach 91%:

**Data Augmentation**
- Random crop (32×32 with padding=4)
- Random horizontal flip
- Random rotation (±15°)
- Random affine translation
- Color jitter (brightness & contrast)
- Random erasing (cutout regularization)

**Optimization**
- SGD with Nesterov momentum (lr=0.1, momentum=0.9)
- Weight decay: 1e-4
- Label smoothing: 0.1
- Cosine annealing LR schedule over 300 epochs (η_min=1e-7)

**Hardware**
- Trained using `torch-directml` (AMD/DirectX GPU backend)

## Project Structure

```
├── CNN.py            # Model architecture
├── main.py           # Training loop, evaluation, and inference
├── simplerunner.py   # Lightweight inference script
├── main.ipynb        # Notebook version
└── trained_net_91.01.pth  # Pre-trained weights
```

## Getting Started

**Install dependencies**
```bash
pip install torch torchvision
```

> If you're on an AMD GPU, also install `torch-directml`. Otherwise, replace `torch_directml.device()` with `torch.device('cuda')` or `torch.device('cpu')` in `main.py`.

**Run inference on your own image**
```bash
python simplerunner.py
```

Or edit the `image_paths` list in `main.py` to point to your image. The model resizes any input to 32×32 and outputs one of:

`plane · car · bird · cat · deer · dog · frog · horse · ship · truck`

**Retrain from scratch**
```bash
python main.py
```

Training takes ~300 epochs. The model is saved as `trained_net_{accuracy}.pth` when complete.

Reaching 91% on CIFAR-10 without residual connections or attention mechanisms required careful stacking of regularization techniques. The key contributors:

- **Cosine annealing** outperformed step-decay LR by ~1.5% in experiments
- **Label smoothing** reduced overconfidence and improved generalization
- **Random erasing** (cutout) was the single biggest augmentation boost
- **Nesterov momentum** converged faster and more stably than vanilla SGD
