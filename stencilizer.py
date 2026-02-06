import argparse
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from PIL import Image

Coord = Tuple[int, int]


@dataclass
class BridgeResult:
    path: List[Coord]
    relaxed: bool


def load_binary_image(path: Path, invert: bool = False, threshold: int = 128) -> np.ndarray:
    image = Image.open(path).convert("RGBA")
    rgba = np.array(image)
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]

    # Convert to luminance.
    luminance = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    # Treat fully transparent pixels as white.
    luminance = np.where(alpha == 0, 255.0, luminance)

    foreground = luminance < threshold  # True for black pixels
    if invert:
        foreground = ~foreground
    return foreground


def save_binary_image(path: Path, foreground: np.ndarray) -> None:
    out = np.where(foreground, 0, 255).astype(np.uint8)
    img = Image.fromarray(out, mode="L")
    img.save(path)


def neighbors(coord: Coord, height: int, width: int) -> Iterable[Coord]:
    r, c = coord
    if r > 0:
        yield (r - 1, c)
    if r + 1 < height:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < width:
        yield (r, c + 1)


def flood_fill(mask: np.ndarray, starts: Iterable[Coord]) -> np.ndarray:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue = deque()
    for s in starts:
        if mask[s] and not visited[s]:
            visited[s] = True
            queue.append(s)
    while queue:
        cur = queue.popleft()
        for nxt in neighbors(cur, height, width):
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    return visited


def find_hole_components(background: np.ndarray) -> List[List[Coord]]:
    height, width = background.shape
    border = [(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
    border += [(r, 0) for r in range(height)] + [(r, width - 1) for r in range(height)]
    outside = flood_fill(background, border)
    holes_mask = background & ~outside

    visited = np.zeros_like(background, dtype=bool)
    components: List[List[Coord]] = []

    for r in range(height):
        for c in range(width):
            if holes_mask[r, c] and not visited[r, c]:
                comp: List[Coord] = []
                queue = deque([(r, c)])
                visited[r, c] = True
                while queue:
                    cur = queue.popleft()
                    comp.append(cur)
                    for nxt in neighbors(cur, height, width):
                        if holes_mask[nxt] and not visited[nxt]:
                            visited[nxt] = True
                            queue.append(nxt)
                components.append(comp)
    return components


def compute_boundary_black(hole: Sequence[Coord], foreground: np.ndarray) -> Set[Coord]:
    height, width = foreground.shape
    boundary: Set[Coord] = set()
    for r, c in hole:
        for nr, nc in neighbors((r, c), height, width):
            if foreground[nr, nc]:
                boundary.add((nr, nc))
    return boundary


def compute_outer_boundary_black(foreground: np.ndarray, background: np.ndarray) -> Set[Coord]:
    height, width = foreground.shape
    border = [(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
    border += [(r, 0) for r in range(height)] + [(r, width - 1) for r in range(height)]
    outside = flood_fill(background, border)
    outer_boundary: Set[Coord] = set()
    for r in range(height):
        for c in range(width):
            if foreground[r, c]:
                for nr, nc in neighbors((r, c), height, width):
                    if outside[nr, nc]:
                        outer_boundary.add((r, c))
                        break
    return outer_boundary


def bfs_path(
    sources: Sequence[Coord],
    targets: Set[Coord],
    allowed: np.ndarray,
) -> Optional[List[Coord]]:
    height, width = allowed.shape
    queue = deque()
    prev: dict[Coord, Optional[Coord]] = {}

    for s in sources:
        if allowed[s]:
            queue.append(s)
            prev[s] = None

    while queue:
        cur = queue.popleft()
        if cur in targets:
            path: List[Coord] = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nxt in neighbors(cur, height, width):
            if allowed[nxt] and nxt not in prev:
                prev[nxt] = cur
                queue.append(nxt)
    return None


def compress_path(path: Sequence[Coord]) -> List[Coord]:
    if len(path) < 3:
        return list(path)
    out = [path[0]]
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for idx in range(1, len(path) - 1):
        cur_dir = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if cur_dir != prev_dir:
            out.append(path[idx])
        prev_dir = cur_dir
    out.append(path[-1])
    return out


def chaikin_smooth(points: Sequence[Tuple[float, float]], iterations: int) -> List[Tuple[float, float]]:
    if len(points) < 3 or iterations <= 0:
        return list(points)
    result = list(points)
    for _ in range(iterations):
        new_points: List[Tuple[float, float]] = [result[0]]
        for p0, p1 in zip(result[:-1], result[1:]):
            q = (0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1])
            r = (0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1])
            new_points.extend([q, r])
        new_points.append(result[-1])
        result = new_points
    return result


def draw_line(mask: np.ndarray, start: Coord, end: Coord) -> None:
    r0, c0 = start
    r1, c1 = end
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    r, c = r0, c0
    height, width = mask.shape
    while True:
        if 0 <= r < height and 0 <= c < width:
            mask[r, c] = True
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def rasterize_polyline(points: Sequence[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if len(points) == 0:
        return mask
    rounded = [(int(round(p[0])), int(round(p[1]))) for p in points]
    for p0, p1 in zip(rounded[:-1], rounded[1:]):
        draw_line(mask, p0, p1)
    return mask


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 1:
        return mask
    height, width = mask.shape
    out = mask.copy()
    for dr in range(-(radius - 1), radius):
        for dc in range(-(radius - 1), radius):
            if dr == 0 and dc == 0:
                continue
            r0 = max(0, dr)
            r1 = min(height, height + dr)
            c0 = max(0, dc)
            c1 = min(width, width + dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def rasterize_path(
    path: Sequence[Coord],
    shape: Tuple[int, int],
    smooth: bool,
    smooth_iterations: int,
    bridge_width: int,
) -> np.ndarray:
    if not path:
        return np.zeros(shape, dtype=bool)
    if smooth:
        compressed = compress_path(path)
        points = [(float(r), float(c)) for r, c in compressed]
        smoothed = chaikin_smooth(points, smooth_iterations)
        mask = rasterize_polyline(smoothed, shape)
    else:
        mask = np.zeros(shape, dtype=bool)
        for r, c in path:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
    return dilate_mask(mask, bridge_width)


def compute_bridge_path(
    boundary_black: Sequence[Coord],
    outer_black: Set[Coord],
    foreground: np.ndarray,
    avoid_mask: np.ndarray,
) -> Optional[List[Coord]]:
    allowed = foreground & ~avoid_mask
    return bfs_path(boundary_black, outer_black, allowed)


def generate_layers(
    foreground: np.ndarray,
    layers: int,
    max_relax_per_hole: int = 1,
    smooth: bool = True,
    smooth_iterations: int = 2,
    bridge_width: int = 1,
) -> Tuple[List[np.ndarray], List[str]]:
    height, width = foreground.shape
    background = ~foreground
    logging.info("Scanning for holes...")
    holes = find_hole_components(background)
    logging.info("Found %d hole component(s)", len(holes))
    logging.info("Computing outer boundary...")
    outer_black = compute_outer_boundary_black(foreground, background)

    layer_images = [foreground.copy() for _ in range(layers)]
    usage_count = np.zeros_like(foreground, dtype=np.uint8)
    warnings: List[str] = []

    for hole_index, hole in enumerate(holes, start=1):
        if hole_index % 10 == 0 or hole_index == 1 or hole_index == len(holes):
            logging.info("Processing hole %d/%d", hole_index, len(holes))
        boundary_black = list(compute_boundary_black(hole, foreground))
        if not boundary_black:
            continue

        for layer_index in range(layers):
            logging.debug("Finding bridge for hole %d, layer %d", hole_index, layer_index + 1)
            avoid_mask = usage_count >= (layers - 1)
            path = compute_bridge_path(boundary_black, outer_black, foreground, avoid_mask)
            relaxed = False

            if path is None and max_relax_per_hole > 0:
                path = compute_bridge_path(boundary_black, outer_black, foreground, np.zeros_like(foreground, dtype=bool))
                relaxed = True

            if path is None:
                warnings.append(
                    f"Hole {hole_index}: unable to find bridge for layer {layer_index + 1}."
                )
                continue

            path_mask = rasterize_path(
                path,
                foreground.shape,
                smooth=smooth,
                smooth_iterations=smooth_iterations,
                bridge_width=bridge_width,
            )
            layer_images[layer_index][path_mask] = False
            usage_count = np.minimum(usage_count + path_mask.astype(np.uint8), layers)

            if relaxed:
                warnings.append(
                    f"Hole {hole_index}: bridge for layer {layer_index + 1} required relaxation; union may differ if bridges overlap across all layers."
                )

    return layer_images, warnings


def _gpu_bfs_path(
    sources_mask,
    targets_mask,
    allowed_mask,
):
    import cupy as cp

    if not cp.any(sources_mask & allowed_mask):
        return None
    if not cp.any(targets_mask):
        return None

    frontier = sources_mask & allowed_mask
    visited = frontier.copy()
    pred = cp.zeros(allowed_mask.shape, dtype=cp.int8)

    while True:
        if cp.any(frontier & targets_mask):
            break

        new_frontier = cp.zeros_like(frontier)

        # Up (from below)
        cand = cp.zeros_like(frontier)
        cand[:-1, :] = frontier[1:, :]
        new = cand & allowed_mask & ~visited
        if cp.any(new):
            pred = cp.where(new, 2, pred)
            visited |= new
            new_frontier |= new

        # Down (from above)
        cand = cp.zeros_like(frontier)
        cand[1:, :] = frontier[:-1, :]
        new = cand & allowed_mask & ~visited
        if cp.any(new):
            pred = cp.where(new, 1, pred)
            visited |= new
            new_frontier |= new

        # Left (from right)
        cand = cp.zeros_like(frontier)
        cand[:, :-1] = frontier[:, 1:]
        new = cand & allowed_mask & ~visited
        if cp.any(new):
            pred = cp.where(new, 4, pred)
            visited |= new
            new_frontier |= new

        # Right (from left)
        cand = cp.zeros_like(frontier)
        cand[:, 1:] = frontier[:, :-1]
        new = cand & allowed_mask & ~visited
        if cp.any(new):
            pred = cp.where(new, 3, pred)
            visited |= new
            new_frontier |= new

        frontier = new_frontier
        if not cp.any(frontier):
            return None

    targets = cp.argwhere(visited & targets_mask)
    if targets.size == 0:
        return None

    target = targets[0]
    pred_cpu = cp.asnumpy(pred)
    target_cpu = tuple(int(x) for x in cp.asnumpy(target))

    path: List[Coord] = []
    r, c = target_cpu
    path.append((r, c))
    while pred_cpu[r, c] != 0:
        direction = pred_cpu[r, c]
        if direction == 1:
            r -= 1
        elif direction == 2:
            r += 1
        elif direction == 3:
            c -= 1
        elif direction == 4:
            c += 1
        path.append((r, c))

    path.reverse()
    return path


def generate_layers_gpu(
    foreground,
    layers: int,
    max_relax_per_hole: int = 1,
    smooth: bool = True,
    smooth_iterations: int = 2,
    bridge_width: int = 1,
) -> Tuple[List, List[str]]:
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

    structure = cp.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=cp.uint8,
    )

    background = ~foreground
    result = cndi.label(background, structure=structure)
    labels, num = cast(Tuple[Any, int], result)

    if num == 0:
        return [foreground.copy() for _ in range(layers)], []

    border_labels = cp.unique(
        cp.concatenate(
            [labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
        )
    )
    border_labels = border_labels[border_labels != 0]

    all_labels = cp.arange(1, num + 1, dtype=cp.int32)
    hole_labels = all_labels[~cp.isin(all_labels, border_labels)]
    holes_count = int(hole_labels.size)

    logging.info("Found %d hole component(s) (GPU)", holes_count)

    outside_mask = (labels > 0) & cp.isin(labels, border_labels)
    outer_black = foreground & cndi.binary_dilation(outside_mask, structure=structure)

    layer_images = [foreground.copy() for _ in range(layers)]
    usage_count = cp.zeros_like(foreground, dtype=cp.uint8)
    warnings: List[str] = []

    for hole_index, label_id in enumerate(hole_labels, start=1):
        if hole_index % 10 == 0 or hole_index == 1 or hole_index == holes_count:
            logging.info("Processing hole %d/%d (GPU)", hole_index, holes_count)

        hole_mask = labels == label_id
        boundary_black = foreground & cndi.binary_dilation(hole_mask, structure=structure)
        if not cp.any(boundary_black):
            continue

        for layer_index in range(layers):
            avoid_mask = usage_count >= (layers - 1)
            path = _gpu_bfs_path(boundary_black, outer_black, foreground & ~avoid_mask)
            relaxed = False

            if path is None and max_relax_per_hole > 0:
                path = _gpu_bfs_path(boundary_black, outer_black, foreground)
                relaxed = True

            if path is None:
                warnings.append(
                    f"Hole {hole_index}: unable to find bridge for layer {layer_index + 1}."
                )
                continue

            path_mask_cpu = rasterize_path(
                path,
                foreground.shape,
                smooth=smooth,
                smooth_iterations=smooth_iterations,
                bridge_width=bridge_width,
            )
            path_mask = cp.asarray(path_mask_cpu)
            layer_images[layer_index] = layer_images[layer_index] & ~path_mask
            usage_count = cp.minimum(usage_count + path_mask.astype(cp.uint8), layers)

            if relaxed:
                warnings.append(
                    f"Hole {hole_index}: bridge for layer {layer_index + 1} required relaxation; union may differ if bridges overlap across all layers."
                )

    return layer_images, warnings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multiple stencil layers to avoid holes while preserving the original silhouette when overlayed."
    )
    parser.add_argument("input", type=Path, help="Input monochrome PNG or BMP")
    parser.add_argument("--layers", type=int, default=3, help="Number of stencil layers to generate")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--invert", action="store_true", help="Invert input if white is foreground")
    parser.add_argument("--threshold", type=int, default=128, help="Threshold for binarization (0-255)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration via CuPy")
    parser.add_argument("--no-smooth", action="store_true", help="Disable curved bridges")
    parser.add_argument("--smooth-iterations", type=int, default=2, help="Chaikin smoothing iterations")
    parser.add_argument("--bridge-width", type=int, default=1, help="Bridge width in pixels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )
    if args.layers < 1:
        raise SystemExit("--layers must be >= 1")

    logging.info("Loading input image: %s", args.input)
    foreground = load_binary_image(args.input, invert=args.invert, threshold=args.threshold)
    logging.info("Generating %d layer(s)...", args.layers)

    smooth = not args.no_smooth
    if args.gpu:
        try:
            import cupy as cp
        except Exception as exc:
            raise SystemExit(
                "GPU mode requires CuPy. Install with: pip install cupy-cuda12x (or cupy-cuda11x)."
            ) from exc

        foreground_gpu = cp.asarray(foreground)
        layers, warnings = generate_layers_gpu(
            foreground_gpu,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
        )
        layers = [cp.asnumpy(layer) for layer in layers]
    else:
        layers, warnings = generate_layers(
            foreground,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for idx, layer in enumerate(layers, start=1):
        out_path = args.outdir / f"layer_{idx:02d}.png"
        save_binary_image(out_path, layer)
        logging.info("Wrote %s", out_path)

    if warnings:
        logging.warning("Warnings:")
        for w in warnings:
            logging.warning("- %s", w)

    logging.info("Done. Wrote %d layer(s) to %s", len(layers), args.outdir)


if __name__ == "__main__":
    main()
