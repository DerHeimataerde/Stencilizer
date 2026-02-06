# Stencilizer

Creates multiple stencil layers from a monochrome image so that each layer has bridges to avoid isolated holes. Overlaying all layers reconstructs the original silhouette.

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
- `--invert`: Use if your foreground is white and background is black.
- `--threshold`: Binarization threshold (0-255).
- `--layers`: Number of stencils to generate.
- `--gpu`: Use GPU acceleration via CuPy.
- `--verbose`: Enable verbose logging.
- `--no-smooth`: Disable curved bridges.
- `--smooth-iterations`: Number of Chaikin smoothing passes (default 2).
- `--bridge-width`: Bridge width in pixels (default 1).

## Output

Layers are written as `layer_01.png`, `layer_02.png`, ... in the output directory.
