# Stencilizer

Creates multiple stencil layers from a monochrome image so that each layer has bridges to avoid isolated holes. Overlaying all layers reconstructs the original silhouette.

## Features

- Automatic island detection with configurable sensitivity for thin/diagonal walls
- Organic curved bridges using Catmull-Rom splines and Chaikin smoothing
- Multiple bridges per island for stronger stencil support
- GPU acceleration via CuPy (optional)
- Difference visualization to verify output accuracy

## Install

### Requirements
- Python 3.11.x
- (GPU mode) NVIDIA CUDA 12.x

### Create a virtual environment (Windows)

```
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
```

### Install dependencies

```
pip install -r requirements.txt
```

## Usage

```
python stencilizer.py input.png --layers 3 --outdir output
```

### GPU mode (optional)

Requires an NVIDIA GPU with CUDA and CuPy installed.

```
pip install cupy-cuda12x
python stencilizer.py input.png --layers 3 --outdir output --gpu
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--layers` | 3 | Number of stencil layers to generate |
| `--outdir` | output | Output directory |
| `--invert` | off | Invert input (use if foreground is white) |
| `--threshold` | 128 | Binarization threshold (0-255) |
| `--bridge-width` | 1 | Bridge width in pixels |
| `--smooth-iterations` | 2 | Chaikin smoothing passes for curved bridges |
| `--no-smooth` | off | Disable curved bridges (straight lines only) |
| `--min-island-size` | 1 | Minimum island size in pixels (smaller ignored) |
| `--erode-passes` | 1 | Erosion passes for island detection (higher = catches more diagonal/thin wall islands) |
| `--bridges-per-island` | 1 | Number of bridges per island (more = stronger support) |
| `--gpu` | off | Use GPU acceleration via CuPy |
| `--verbose` | off | Enable verbose logging |
| `--log` | off | Save console output to file (default: log.txt) |

## Output

The following files are written to the output directory:

| File | Description |
|------|-------------|
| `layer_01.png`, `layer_02.png`, ... | Stencil layers (black = holes, transparent = material) |
| `combined.png` | Combined result of all layers overlaid |
| `islands.png` | Detected islands (green) overlayed on original |
| `bridges.png` | All bridges (blue) overlayed on original |
| `diff.png` | Differences (red) overlayed on original |
| `log.txt` | Console output (only with `--log`) |

The program also outputs difference statistics showing the number and percentage of pixels that differ from the original.
