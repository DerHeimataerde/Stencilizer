# Stencilizer

Creates multiple stencil layers from a monochrome image so that each layer has bridges to avoid isolated holes. Overlaying all layers reconstructs the original silhouette.

## Features

- Automatic island detection with configurable sensitivity for thin/diagonal walls
- Organic curved bridges using Catmull-Rom splines and Chaikin smoothing
- Multiple bridges per island for stronger stencil support
- Structural bridges: automatically subdivides large open-cutout regions so no single piece of material floats unsupported
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
python stencilizer.py input.png --layers 2 --outdir output --gpu --min-island-size 2 --erode-passes 1 --bridges-per-island 2 --bridge-width 2 --smooth-iterations 3 --max-cutout-size 5000 --verbose --log log.txt
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| ``--layers`` | 2 | Number of bridge fill layers (not counting the base layer and the structural fill layer) |
| ``--outdir`` | output | Output directory |
| ``--invert`` | off | Invert input (use if foreground is white) |
| ``--threshold`` | 128 | Binarization threshold (0-255) |
| ``--bridge-width`` | 1 | Bridge width in pixels |
| ``--smooth-iterations`` | 0 | Chaikin smoothing passes for curved bridges |
| ``--no-smooth`` | off | Disable curved bridges (straight lines only) |
| ``--min-island-size`` | 1 | Minimum island size in pixels (smaller ignored) |
| ``--erode-passes`` | 0 | Erosion passes for island detection (higher = catches more diagonal/thin wall islands) |
| ``--bridges-per-island`` | 1 | Number of bridges per island (more = stronger support) |
| ``--structural-bridges`` / ``--no-structural-bridges`` | on | Enable/disable structural bridge subdivision for large cutout regions |
| ``--max-cutout-size`` | 0 (disabled) | Max foreground region size in pixels before a structural bridge is inserted to split it (0 = disabled) |
| ``--clip-structural`` | off | Clip dilated structural bridges back to their own region mask |
| ``--gpu`` | off | Use GPU acceleration via CuPy |
| ``--verbose`` | off | Enable verbose logging |
| ``--log`` | off | Save console output to file (default: log.txt) |

## Output

The following files are written to the output directory:

| File | Description |
|------|-------------|
| ``layers/layer_0_base.png`` | Base layer: full design with all bridges solid (use first when painting) |
| ``layers/layer_1_structural.png`` | Structural fill layer: holes at structural bridge locations (only when structural bridges are inserted) |
| ``layers/layer_2_bridges.png``, ``layers/layer_3_bridges.png``, ... | Bridge fill layers: holes at regular bridge locations (round-robin across ``--layers`` layers) |
| ``combined.png`` | Combined result of all layers overlaid |
| ``islands.png`` | Detected islands (green) overlayed on original |
| ``bridges.png`` | All bridges (blue) overlayed on original |
| ``cutouts.png`` | Foreground regions after regular bridges, each in a random colour |
| ``structural.png`` | Oversized regions (coloured) + structural bridges (pink) overlayed on original |
| ``struct_bridges.png`` | Structural bridges (pink) overlaid on original, saved after generation completes |
| ``diff.png`` | Differences (red) overlayed on original |
| ``log.txt`` | Console output (only with ``--log``) |

### Layer order when painting

1. **``layers/layer_0_base``** - paint first. All bridge and structural gaps are solid here.
2. **``layers/layer_1_structural``** *(if structural bridges were inserted)* - fills in structural bridge gaps.
3. **``layers/layer_2_bridges``**, **``layers/layer_3_bridges``**, ... - fill in regular bridge gaps, one layer per ``--layers`` value.

Overlaying all layers in this order reconstructs the complete original silhouette.

The program also outputs difference statistics showing the number and percentage of pixels that differ from the original.
