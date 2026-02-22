# Stencilizer

Creates multiple stencil layers from a monochrome image so that each layer has bridges to avoid isolated holes. Overlaying all layers reconstructs the original silhouette.

## How It Works

A physical stencil is a flat piece of material with holes cut through it. Paint passes through the holes onto the surface below. The problem is that any region of the design that is completely surrounded by background (a "hole within the stencil material") causes the encircled piece of material to fall out — it has nothing holding it in place.

The solution is **bridges**: thin strips of stencil material that cross the hole to connect the inner island to the outer body, keeping everything in one piece. The downside is that bridges block paint and leave gaps in the final result. Stencilizer solves this by distributing bridges across multiple layers. Each layer uses a different set of bridges, so the gaps in one layer are covered by solid material in another. Painting all layers in sequence reconstructs the complete original design.

---

### Stage 1 — Image Binarisation

The input is read as RGBA. Luminance is computed with the ITU-R BT.601 coefficients:

$$L = 0.299R + 0.587G + 0.114B$$

Fully transparent pixels (`alpha = 0`) are forced to white (255) before thresholding so transparency is treated as background. The result is a boolean array `foreground` where `True` denotes pixels where the stencil removes material (the design) and `False` denotes background (the carrier sheet).

---

### Stage 2 — Island Detection

An **island** is a connected region of background that is fully enclosed by foreground — it cannot reach the image border. If cut, the piece of foreground material surrounding it would have no connection to the rest of the sheet.

Two detection strategies run in parallel and their results are OR-combined:

#### 2a. 4-Connected Flood Fill (baseline)

Starting from every pixel on the image border, a flood fill propagates through background pixels using **4-connectivity** (up/down/left/right only). Any background pixel not reached is enclosed and belongs to an island.

4-connectivity is intentionally chosen for the "outside" fill. A diagonal background gap — two background pixels touching only at a corner — is not a physical opening a paint brush can pass through. 4-connectivity correctly leaves such gaps sealed.

Island pixels are subsequently grouped into connected components using **8-connectivity** (includes diagonals). 8-connectivity is correct here because foreground pixels touching only at a corner still form a continuous material wall; conversely, two background regions separated only by such a diagonal wall are genuinely distinct holes.

#### 2b. Dilated-Foreground Detection (`--erode-passes N`)

The baseline misses islands that are separated from the outside by passages thinner than one pixel (e.g. a diagonal line of foreground pixels with background on both sides). From a flood-fill perspective these passages are open; physically they are walls.

The fix: dilate the foreground mask with a full 3×3 structuring element for N passes (equivalent to a morphological closing of the background by a (2N+1)×(2N+1) diamond). This thickens every foreground wall by N pixels in all 8 directions, sealing thin/diagonal passages in the background. Islands are then re-detected via 4-connected flood fill in this shrunk background. Any pixel that is isolated in the shrunk space is guaranteed to be inside a physically enclosed region in the original image. These pixels are added directly to the island mask without expansion, preserving separation between distinct islands that the dilation may have merged.

The combined island mask is then component-labelled with 8-connectivity to produce the final list of islands. Islands smaller than `--min-island-size` pixels are discarded.

---

### Stage 3 — Bridge Routing

#### 3a. Anchor Targets

A bridge must connect an island to the **outer foreground boundary** — foreground pixels that are physically adjacent to the outside of the design, where the stencil is held against the surface. These are found by:

1. 4-connected flood fill from the border to label all outside background pixels.
2. Every foreground pixel that has at least one 8-connected outside background neighbour is added to the anchor target set.

Outside detection uses 4-connectivity for the same reason as island detection. Anchor detection uses 8-connectivity so bridges can reach diagonally adjacent anchors.

#### 3b. Path Finding (CPU mode)

For each island, the boundary of the island — foreground pixels 8-adjacent to island pixels — forms the set of bridge starting points. BFS then searches for a path to any anchor target, trying three progressively more permissive strategies:

1. **4-connected through foreground only** — the cleanest result. The path follows existing material without jumping corners or crossing background.
2. **8-connected through foreground only** — permits diagonal steps, allowing the path to thread through tight foreground passages unreachable by 4-connectivity.
3. **8-connected through any pixel** — last resort. Allows the path to cross background gaps. This handles degenerate inputs where foreground is fragmented.

BFS guarantees the **shortest** path for the given connectivity and allowed mask, minimising bridge length (and therefore the visible gap in the painted result).

#### 3c. Path Finding (GPU mode)

When `--gpu` is enabled, a **geodesic distance field** is computed on the GPU in place of per-island BFS. The distance field $D(p)$ stores the shortest 8-connected path length from $p$ to any anchor target, constrained to traverse only foreground pixels.

A custom CUDA kernel implements the Bellman-Ford relaxation. Each thread $t$ owns pixel $p_t$ and reads the current distances of all 8 neighbours:

$$D_{\text{new}}(p) = \min_{q \in N_8(p)} \bigl(D(q) + 1\bigr)$$

where $N_8(p)$ is the 8-neighbourhood of $p$. The kernel updates `dist[p]` and `parent[p]` only when the candidate is strictly smaller than the current value. Because each thread writes only to its own cell and reads neighbours read-only, there are no write conflicts and no atomics are needed on the main arrays. A single `int` convergence flag (set via `atomicOr`) indicates whether any pixel improved during the pass.

Convergence is checked with a single scalar GPU→CPU copy (`int(improved[0])`). This is the only CPU synchronisation point per pass. The kernel is compiled once via `cp.RawKernel` and cached.

**Why Bellman-Ford and not BFS?** True frontier BFS requires one GPU→CPU sync per distance level (to check whether the frontier is empty). For a 2000×2000 image with max geodesic distance ~1500, that is >1500 stalls. Bellman-Ford converges in ~5–20 passes regardless of image size, each pass touching every pixel — the total compute is proportional to $H \times W \times \text{passes}$, but with a small constant and no stalls between passes.

The parent map encodes the full shortest-path tree at no extra cost. Each pixel stores an integer direction code (1–8) recording which neighbour it descended from. Tracing a path from any pixel to the nearest target is then O(L) — one array lookup per step — compared to O(L × 8) for the naive gradient-descent approach of scanning all neighbours at every step.

#### 3d. Multiple Bridges Per Island

When `--bridges-per-island N > 1`, N starting points are chosen from the island boundary at maximally separated angles. For each boundary pixel $(r, c)$, its angle relative to the island centroid $(\bar r, \bar c)$ is:

$$\theta = \text{atan2}(r - \bar r,\ c - \bar c)$$

Boundary pixels are sorted by $\theta$ and N pixels are selected by matching to the ideal evenly-spaced angles $\{-\pi + \tfrac{2\pi k}{N} : k = 0, \dots, N-1\}$ with minimum circular-arc distance. A separate bridge path is computed from each selected starting pixel. Previously used anchor endpoints are excluded from subsequent searches to spread bridge landing points across the boundary rather than bunching them together.

#### 3e. Bridge Rasterisation and Smoothing

Raw BFS paths follow the pixel grid and look jagged. The smoothing pipeline transforms them into organic curves:

**Path compression**: direction-change keypoints (corners) are extracted — only pixels where the step direction changes are retained, reducing a 500-point path to perhaps 20 keypoints. For long straight segments (Manhattan distance > 10 between consecutive keypoints), a midpoint is inserted to give the spline enough control points to curve smoothly.

**Catmull-Rom spline**: given control points $P_0, P_1, \ldots, P_n$, each span $[P_i, P_{i+1}]$ is evaluated with the standard Catmull-Rom basis:

$$\mathbf{q}(t) = \tfrac{1}{2}\begin{bmatrix}1 & t & t^2 & t^3\end{bmatrix} \begin{bmatrix}0 & 2 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 2 & -5 & 4 & -1 \\ -1 & 3 & -3 & 1\end{bmatrix} \begin{bmatrix}P_{i-1} \\ P_i \\ P_{i+1} \\ P_{i+2}\end{bmatrix}$$

The schema passes through every control point ($\mathbf{q}(0) = P_i$, $\mathbf{q}(1) = P_{i+1}$) and is $C^1$ continuous at knots: the tangent at $P_i$ equals $\tfrac{1}{2}(P_{i+1} - P_{i-1})$. Ghost points at both ends are constructed by reflecting: $P_{-1} = 2P_0 - P_1$ and $P_{n+1} = 2P_n - P_{n-1}$.

**Chaikin corner cutting**: each pass replaces every edge $(P_i, P_{i+1})$ with two points at parametric positions $t = \tfrac{1}{4}$ and $t = \tfrac{3}{4}$:

$$Q_i = \tfrac{3}{4}P_i + \tfrac{1}{4}P_{i+1}, \qquad R_i = \tfrac{1}{4}P_i + \tfrac{3}{4}P_{i+1}$$

After $k$ passes the curve converges to a uniform quadratic B-spline. The limit curve is $C^1$ everywhere. Combined with the Catmull-Rom spline, this produces bridges that look hand-drawn rather than algorithmic.

**Rasterisation**: the final floating-point polyline is rasterised segment by segment using Bresenham's line algorithm, which produces pixel-perfect line drawing with no gaps by integrating an error accumulator along the major axis.

**Thick paths**: width > 1 is applied by stamping a circular disk at every path pixel. The disk is precomputed as a set of integer (Δr, Δc) offsets satisfying $\Delta r^2 + \Delta c^2 < r^2$ and cached by radius. Broadcasting over all path pixels is done entirely in numpy: `pts_r = path[:, 0:1] + offsets[:, 0]` produces an (N_path × N_offsets) array in one operation; the result is clipped to image bounds and written to the mask in a single index assignment.

---

### Stage 4 — Structural Bridges

#### 4a. Motivation

Regular bridges guarantee that every enclosed hole (island) is connected to the outer body. They do not guarantee that the outer body itself is structurally sound. A large solid foreground region — the letter O's outer ring, for example — could still be a single large piece that sags, warps, or tears when cut from thin material. Structural bridges subdivide such regions.

#### 4b. Component Sizing

After all regular bridges are inserted, 8-connected component labelling is applied to the updated foreground mask. Component sizes are computed with `np.bincount` over the flattened label array — O(H × W) rather than O(count × H × W). Components with more than `--max-cutout-size` pixels are flagged as oversized.

#### 4c. Chord Placement — Median Projection

For each oversized component, the algorithm sweeps 36 chord orientations at 5° intervals (0°, 5°, …, 175°). For each orientation angle $\alpha$:

Let the chord's **normal direction** be $\hat{n} = (-\sin\alpha, \cos\alpha)$ and its **chord direction** be $\hat{c} = (\cos\alpha, \sin\alpha)$.

Project every pixel coordinate $(r_i, c_i)$ of the region onto the normal:

$$s_i = r_i \cdot \hat{n}_r + c_i \cdot \hat{n}_c$$

Place the chord at $s = \text{median}(\{s_i\})$. This choice is optimal for balanced splitting: by definition of the median, roughly half the pixels have $s_i \leq \text{median}$ and half have $s_i \geq \text{median}$, regardless of whether the region is convex or concave, and regardless of whether the centroid lies inside the region (it may not for C- or U-shaped regions).

The chord spine consists of pixels within a tolerance of the median line. Tolerance is set to $\max(0.5,\ \text{d}_\text{min} + 0.5)$ where $\text{d}_\text{min}$ is the minimum distance any pixel has to the median — this ensures at least one pixel is selected even if the projection distribution is discrete and sparse. Chord endpoints are the extremes of the spine projected along $\hat{c}$.

The chord is then rasterised (Bresenham), dilated to `bridge_width`, and its effect tested: if removing it yields ≥ 2 connected components in the cropped bounding box, it is a **clean split**. All work is done inside the region's bounding box (plus `bridge_width + 1` padding for dilation), so cost is O(region area), not O(image area).

Score is chord length (Euclidean distance between endpoints). The shortest clean-split chord is chosen. If no angle produces a clean split, the shortest chord of any orientation is inserted anyway — the region will still shrink toward the threshold over subsequent iterations.

Larger components are processed first within each iteration so the most problematic regions are resolved earliest.

---

### Stage 5 — Layer Assembly

Bridges are assigned to fill-in layers in round-robin order: bridge 0 → layer 0, bridge 1 → layer 1, …, bridge N → layer (N mod `--layers`). This distributes the visual gaps as evenly as possible across layers.

Let $F$ = foreground mask, $B_i$ = bridge $i$ mask, $S_j$ = structural bridge $j$ mask. The layers are:

$$\text{layer}_0 = \lnot F \cup \bigcup_i B_i \cup \bigcup_j S_j \qquad \text{(base — all bridges solid)}$$

$$\text{layer}_{1+k} = \lnot \bigcup_{i \in \text{group}_k} B_i \qquad k = 0, \ldots, \text{layers}-1 \qquad \text{(bridge fill layers)}$$

$$\text{layer}_\text{struct} = \lnot \bigcup_j S_j \qquad \text{(structural fill layer)}$$

`True` = opaque stencil material. `False` = hole (paint passes through).

When all layers are painted with the same colour in sequence, each pixel receives paint from whichever layer has it as a hole, and the union of all holes exactly equals $F$ — the original design.

**Correctness**: every foreground pixel $p \in F$ belongs to at most one bridge group (or none). If $p$ is a bridge pixel in group $k$, then $\text{layer}_{1+k}$ has $p$ as a hole, so $p$ receives paint from that layer. $\text{layer}_0$ blocks $p$ (bridge is solid), but all other fill layers do not block $p$ (they have no bridge at $p$). If $p$ is not a bridge pixel, $\text{layer}_0$ has $p$ as a hole (since $\lnot F$ is False at $p$, but the bridge union doesn't include $p$, so $\text{layer}_0 = \lnot F \cup \ldots$ — wait, the base layer is `NOT foreground OR bridges`, meaning foreground pixels **without** bridges become holes in the base layer).

More precisely: for a non-bridge foreground pixel, `layer_0` = False (hole), so it receives paint on the first pass. Bridge pixels are solid in `layer_0` but become holes in exactly one fill layer.

---

### Connectivity Convention Summary

| Situation | Connectivity | Reason |
|-----------|-------------|--------|
| "Is this background pixel outside?" | 4-conn | Physical gap requires a direct opening, not just a diagonal touch |
| "Is this background pixel an island?" | 4-conn outside fill → island = anything not reached | Same as above |
| "Group island pixels into components" | 8-conn | Adjacent diagonal backgrounds form one logically contiguous hole |
| "Which foreground pixels are boundary anchors?" | Foreground 8-adj to outside | Bridges can approach diagonally |
| "Route a bridge path" | 4-conn preferred, 8-conn fallback | Prefer material-aligned paths; diagonals used when necessary |
| "Label foreground components for structural bridges" | 8-conn | Conservative: diagonal contact still = one connected piece of material |

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
