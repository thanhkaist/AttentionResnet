# AttentionResnet

Image classification using ResNet backbones augmented with attention mechanisms, evaluated on CIFAR-100.

## Overview

This project implements and compares three attention mechanisms applied to ResNet-34 and ResNet-50:

| Mechanism | Attention Types | Reference |
|-----------|----------------|-----------|
| **SE** (Squeeze-and-Excitation) | Channel | [Hu et al., 2018](https://arxiv.org/abs/1709.01507) |
| **BAM** (Bottleneck Attention Module) | Channel, Spatial, Joint | [Park et al., 2018](https://arxiv.org/abs/1807.06514) |
| **CBAM** (Convolutional Block Attention Module) | Channel, Spatial, Joint | [Woo et al., 2018](https://arxiv.org/abs/1807.06521) |

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy
- tqdm
- tensorboardX
- OpenCV (`cv2`) — for visualization only
- torchsummary — for visualization only

## Dataset

[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) is downloaded automatically to `./data/` on first run.

## Training

Run all experiments at once:

```bash
./run.sh
```

Or train a single model directly:

```bash
# Baseline ResNet-50
python main.py --model resnet50

# SE-Net (channel attention)
python main.py --model se_resnet50 --attention channel

# BAM (spatial / channel / joint attention)
python main.py --model bam_resnet50 --attention spatial
python main.py --model bam_resnet50 --attention channel
python main.py --model bam_resnet50 --attention joint

# CBAM (spatial / channel / joint attention)
python main.py --model cbam_resnet50 --attention spatial
python main.py --model cbam_resnet50 --attention channel
python main.py --model cbam_resnet50 --attention joint
```

Replace `resnet50` / `bam_resnet50` / `se_resnet50` / `cbam_resnet50` with the `resnet34` variants to use the shallower backbone.

## Testing

Run all test evaluations at once:

```bash
./run_test.sh
```

Or test a single checkpoint:

```bash
python main.py --model cbam_resnet50 --attention joint --test
```

## Configuration

All hyperparameters are set via command-line arguments (defined in `configs.py`):

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `resnet50` | Model architecture |
| `--attention` | `no` | Attention type (`no`, `channel`, `spatial`, `joint`) |
| `--norm` | `bn` | Normalization (`bn`, `gn`, `gbn`, `none`) |
| `--batch_size` | `128` | Batch size |
| `--num_epochs` | `100` | Number of training epochs |
| `--learning_rate` | `0.1` | Initial learning rate (SGD with Nesterov momentum) |
| `--weight_decay` | `5e-4` | L2 weight decay |
| `--schedule` | `50 70 80 90 95` | Epochs at which to decay LR by ×0.2 |
| `--checkpoint` | `checkpoint` | Directory for saving model checkpoints |
| `--test` | — | Run in test-only mode |

## Visualization (Grad-CAM)

Place sample images in `results/sample/` (PNG format), then run:

```bash
./visualRun.sh
```

This produces Grad-CAM overlays and vanilla/deconvolution backpropagation visualizations saved to `results/<model>_<attention>/`.

## Project Structure

```
AttentionResnet/
├── models/
│   ├── __init__.py         # Model registry (get_model)
│   └── resnet.py           # ResNet + SE / BAM / CBAM attention layers
├── configs.py              # Argument parsing and Configs class
├── main.py                 # Training and evaluation loop
├── grad_cam.py             # GradCAM, BackPropagation, Deconvnet utilities
├── visualize.py            # Visualization script (Grad-CAM, backprop)
├── run.sh                  # Script to train all model variants
├── run_test.sh             # Script to test all model variants
└── visualRun.sh            # Script to run visualization
```


