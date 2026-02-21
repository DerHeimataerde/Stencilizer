import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from PIL import Image
from tqdm import tqdm

Coord = Tuple[int, int]


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


def save_binary_image(path: Path, stencil: np.ndarray) -> None:
    """Save stencil as PNG: black for holes (False), transparent for stencil material (True)."""
    height, width = stencil.shape
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    # Black opaque pixels where stencil is False (holes where paint passes)
    holes = ~stencil
    rgba[holes, 3] = 255  # Alpha = opaque for holes
    # RGB stays 0 (black) for holes, transparent elsewhere
    img = Image.fromarray(rgba, mode="RGBA")
    img.save(path)


def neighbors4(coord: Coord, height: int, width: int) -> Iterable[Coord]:
    """4-connectivity: orthogonal neighbors only."""
    r, c = coord
    if r > 0:
        yield (r - 1, c)
    if r + 1 < height:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < width:
        yield (r, c + 1)


def neighbors8(coord: Coord, height: int, width: int) -> Iterable[Coord]:
    """8-connectivity: orthogonal + diagonal neighbors."""
    r, c = coord
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                yield (nr, nc)


def flood_fill(mask: np.ndarray, starts: Iterable[Coord], use_8conn: bool = True) -> np.ndarray:
    """Flood fill using 8-connectivity (default) or 4-connectivity."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    queue = deque()
    for s in starts:
        if mask[s] and not visited[s]:
            visited[s] = True
            queue.append(s)
    neighbor_fn = neighbors8 if use_8conn else neighbors4
    while queue:
        cur = queue.popleft()
        for nxt in neighbor_fn(cur, height, width):
            if mask[nxt] and not visited[nxt]:
                visited[nxt] = True
                queue.append(nxt)
    return visited


def find_hole_components(background: np.ndarray, erode_passes: int = 1) -> List[List[Coord]]:
    """Find enclosed white regions (islands) that need bridges.
    
    Combines multiple detection strategies:
    1. Standard 4-conn flood fill (finds fully enclosed regions)
    2. With dilated foreground (finds regions separated by thin/diagonal walls)
    
    erode_passes controls how aggressively thin passages are treated as barriers.
    """
    from scipy import ndimage
    
    height, width = background.shape
    border = [(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
    border += [(r, 0) for r in range(height)] + [(r, width - 1) for r in range(height)]
    
    # Strategy 1: Standard 4-conn (baseline - catches fully enclosed regions)
    outside_4conn = flood_fill(background, border, use_8conn=False)
    holes_mask = background & ~outside_4conn
    
    # Strategy 2: Dilate foreground to close thin passages, find additional islands
    if erode_passes > 0:
        foreground = ~background
        # Use 8-connectivity for dilation so diagonal walls connect
        full_kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        dilated_fg = ndimage.binary_dilation(foreground, structure=full_kernel, iterations=erode_passes).astype(bool)
        shrunk_bg = background & ~dilated_fg
        
        # Find what's outside in shrunk space
        outside_shrunk = flood_fill(shrunk_bg, border, use_8conn=False)
        
        # Find pixels in shrunk_bg that are isolated (can't reach border)
        isolated_in_shrunk = shrunk_bg & ~outside_shrunk
        
        # Add these isolated core pixels directly - they're guaranteed to be in enclosed regions
        # Don't flood fill expand since that could merge separate islands
        holes_mask = holes_mask | isolated_in_shrunk

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
                    for nxt in neighbors8(cur, height, width):  # 8-conn for island grouping
                        if holes_mask[nxt] and not visited[nxt]:
                            visited[nxt] = True
                            queue.append(nxt)
                components.append(comp)
    return components


def compute_boundary_black(hole: Sequence[Coord], foreground: np.ndarray) -> Set[Coord]:
    height, width = foreground.shape
    boundary: Set[Coord] = set()
    for r, c in hole:
        for nr, nc in neighbors8((r, c), height, width):  # 8-conn for boundary detection
            if foreground[nr, nc]:
                boundary.add((nr, nc))
    return boundary


def compute_outer_boundary_black(foreground: np.ndarray, background: np.ndarray) -> Set[Coord]:
    """Find foreground pixels adjacent to the outside (valid bridge targets).
    
    Uses 4-connectivity for outside detection - paint can't flow through diagonal gaps.
    """
    height, width = foreground.shape
    border = [(0, c) for c in range(width)] + [(height - 1, c) for c in range(width)]
    border += [(r, 0) for r in range(height)] + [(r, width - 1) for r in range(height)]
    # Use 4-connectivity for outside - paint can't flow through diagonal gaps
    outside = flood_fill(background, border, use_8conn=False)
    outer_boundary: Set[Coord] = set()
    for r in range(height):
        for c in range(width):
            if foreground[r, c]:
                # Check 8-neighbors for reaching outside (bridges can go diagonally)
                for nr, nc in neighbors8((r, c), height, width):
                    if outside[nr, nc]:
                        outer_boundary.add((r, c))
                        break
    return outer_boundary


def bfs_path(
    sources: Sequence[Coord],
    targets: Set[Coord],
    allowed: np.ndarray,
    use_8conn: bool = False,
) -> Optional[List[Coord]]:
    """BFS to find path from any source to any target.
    
    Sources are always added to the queue regardless of allowed mask.
    Expansion can go to allowed pixels OR target pixels.
    """
    height, width = allowed.shape
    queue = deque()
    prev: dict[Coord, Optional[Coord]] = {}

    # Add ALL sources to the queue - they are valid starting points
    for s in sources:
        if 0 <= s[0] < height and 0 <= s[1] < width:
            queue.append(s)
            prev[s] = None

    neighbor_fn = neighbors8 if use_8conn else neighbors4
    while queue:
        cur = queue.popleft()
        if cur in targets:
            path: List[Coord] = []
            while cur is not None:
                path.append(cur)
                cur = prev[cur]
            path.reverse()
            return path
        for nxt in neighbor_fn(cur, height, width):
            if nxt not in prev and (allowed[nxt] or nxt in targets):
                prev[nxt] = cur
                queue.append(nxt)
    return None


def compress_path(path: Sequence[Coord]) -> List[Coord]:
    """Compress path keeping key turning points, but also sample intermediates for smoother curves."""
    if len(path) < 3:
        return list(path)
    
    # First pass: find direction change points
    corners = [path[0]]
    prev_dir = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    for idx in range(1, len(path) - 1):
        cur_dir = (path[idx + 1][0] - path[idx][0], path[idx + 1][1] - path[idx][1])
        if cur_dir != prev_dir:
            corners.append(path[idx])
        prev_dir = cur_dir
    corners.append(path[-1])
    
    # Second pass: add intermediate points between corners for smoother curves
    if len(corners) < 3:
        return corners
    
    out = [corners[0]]
    for p0, p1 in zip(corners[:-1], corners[1:]):
        dist = abs(p1[0] - p0[0]) + abs(p1[1] - p0[1])
        if dist > 10:
            # Add midpoint for long segments
            mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
            out.append(mid)
        out.append(p1)
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


def catmull_rom_spline(points: Sequence[Tuple[float, float]], num_segments: int = 10) -> List[Tuple[float, float]]:
    """Generate smooth curve through points using Catmull-Rom spline interpolation."""
    if len(points) < 2:
        return list(points)
    if len(points) == 2:
        return list(points)
    
    result: List[Tuple[float, float]] = []
    
    # Extend endpoints for boundary conditions
    pts = list(points)
    pts.insert(0, (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1]))
    pts.append((2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1]))
    
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        
        for t_idx in range(num_segments):
            t = t_idx / num_segments
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom basis functions
            x = 0.5 * (
                (2 * p1[0]) +
                (-p0[0] + p2[0]) * t +
                (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2 * p1[1]) +
                (-p0[1] + p2[1]) * t +
                (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            result.append((x, y))
    
    result.append(points[-1])
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
    # Handle single point - just set that pixel
    if len(rounded) == 1:
        r, c = rounded[0]
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            mask[r, c] = True
        return mask
    for p0, p1 in zip(rounded[:-1], rounded[1:]):
        draw_line(mask, p0, p1)
    return mask


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate mask with circular kernel for smooth bridge edges."""
    if radius <= 1:
        return mask
    # If mask is a cupy array, use cupyx.scipy.ndimage
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cndi
        if isinstance(mask, cp.ndarray):
            diameter = 2 * radius - 1
            y, x = cp.ogrid[:diameter, :diameter]
            center = radius - 1
            structure = (x - center) ** 2 + (y - center) ** 2 < radius * radius
            return cndi.binary_dilation(mask, structure=structure)
    except ImportError:
        pass
    # Fallback to CPU (numpy/scipy)
    from scipy import ndimage as ndi_cpu
    diameter = 2 * radius - 1
    y, x = np.ogrid[:diameter, :diameter]
    center = radius - 1
    structure = ((x - center) ** 2 + (y - center) ** 2) < radius * radius
    return ndi_cpu.binary_dilation(mask, structure=structure)


# Cache for disk offsets to avoid recomputation
_disk_offsets_cache: Dict[int, np.ndarray] = {}

def get_disk_offsets(radius: int) -> np.ndarray:
    """Precompute circular disk offsets for efficient thick line drawing."""
    if radius in _disk_offsets_cache:
        return _disk_offsets_cache[radius]
    if radius <= 1:
        offsets = np.array([[0, 0]], dtype=np.int32)
    else:
        diameter = 2 * radius - 1
        y, x = np.ogrid[:diameter, :diameter]
        center = radius - 1
        disk = ((x - center) ** 2 + (y - center) ** 2) < radius * radius
        offsets = (np.argwhere(disk) - center).astype(np.int32)
    _disk_offsets_cache[radius] = offsets
    return offsets


def draw_thick_path_into_mask(
    mask: np.ndarray,
    path: Sequence[Coord],
    radius: int,
    smooth: bool = False,
    smooth_iterations: int = 0,
) -> None:
    """Draw thick path by stamping circles at each point. Much faster than dilation for sparse paths."""
    if not path or radius <= 0:
        return
    
    height, width = mask.shape
    offsets = get_disk_offsets(radius)
    
    # Get path points (apply smoothing if needed)
    if smooth and len(path) >= 4:
        compressed = compress_path(path)
        points = [(float(r), float(c)) for r, c in compressed]
        spline_points = catmull_rom_spline(points, num_segments=8)
        if smooth_iterations > 0:
            spline_points = chaikin_smooth(spline_points, smooth_iterations)
        path_points = [(int(round(p[0])), int(round(p[1]))) for p in spline_points]
    else:
        path_points = list(path)
    
    # Stamp circles at each path point
    for r, c in path_points:
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                mask[nr, nc] = True


def blend_overlay(base: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.75) -> np.ndarray:
    """Blend a semi-transparent color onto base image where mask is True."""
    result = base.copy().astype(np.float32)
    overlay_color = np.array(color[:3], dtype=np.float32)
    result[mask, :3] = result[mask, :3] * (1 - alpha) + overlay_color * alpha
    result[mask, 3] = 255  # Ensure full opacity in output
    return result.astype(np.uint8)


def blend_overlay_labels(
    base: np.ndarray,
    label_map: np.ndarray,
    colors: np.ndarray,
    alpha: float = 0.65,
) -> np.ndarray:
    """Vectorized multi-label overlay — paints all labels in a single pass.

    Parameters
    ----------
    base      : RGBA uint8 image (H, W, 4)
    label_map : int array (H, W); 0 = background (not painted)
    colors    : float32 array (max_label+1, 3); row 0 is unused
    alpha     : blend strength
    """
    result = base.astype(np.float32)
    fg = label_map > 0
    lbl_idx = label_map[fg]
    result[fg, :3] = result[fg, :3] * (1.0 - alpha) + colors[lbl_idx] * alpha
    result[fg, 3] = 255.0
    return result.astype(np.uint8)


def draw_path_into_mask(
    mask: np.ndarray,
    path: Sequence[Coord],
    smooth: bool,
    smooth_iterations: int,
) -> None:
    """Draw a path into an existing mask (no dilation, in-place modification)."""
    if not path:
        return
    
    shape = mask.shape
    
    # Skip smoothing for very short paths
    if smooth and len(path) >= 4:
        compressed = compress_path(path)
        points = [(float(r), float(c)) for r, c in compressed]
        spline_points = catmull_rom_spline(points, num_segments=8)
        if smooth_iterations > 0:
            spline_points = chaikin_smooth(spline_points, smooth_iterations)
        # Draw smoothed polyline
        for i in range(len(spline_points) - 1):
            r0, c0 = int(round(spline_points[i][0])), int(round(spline_points[i][1]))
            r1, c1 = int(round(spline_points[i + 1][0])), int(round(spline_points[i + 1][1]))
            # Bresenham line
            dr = abs(r1 - r0)
            dc = abs(c1 - c0)
            sr = 1 if r0 < r1 else -1
            sc = 1 if c0 < c1 else -1
            err = dr - dc
            r, c = r0, c0
            while True:
                if 0 <= r < shape[0] and 0 <= c < shape[1]:
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
    else:
        # No smoothing: draw original path
        for r, c in path:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True


def rasterize_path(
    path: Sequence[Coord],
    shape: Tuple[int, int],
    smooth: bool,
    smooth_iterations: int,
    bridge_width: int,
) -> np.ndarray:
    if not path:
        return np.zeros(shape, dtype=bool)
    
    # Skip smoothing for very short paths - smoothing doesn't improve tiny bridges
    # and can potentially eliminate them
    if smooth and len(path) >= 4:
        compressed = compress_path(path)
        points = [(float(r), float(c)) for r, c in compressed]
        # Use Catmull-Rom spline for organic curves through control points
        spline_points = catmull_rom_spline(points, num_segments=8)
        # Optional: apply Chaikin for additional smoothing
        if smooth_iterations > 0:
            spline_points = chaikin_smooth(spline_points, smooth_iterations)
        mask = rasterize_polyline(spline_points, shape)
    else:
        # No smoothing: directly rasterize the original path
        mask = np.zeros(shape, dtype=bool)
        for r, c in path:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
    return dilate_mask(mask, bridge_width)


def rasterize_path_gpu(
    path: Sequence[Coord],
    shape: Tuple[int, int],
    smooth: bool,
    smooth_iterations: int,
    bridge_width: int,
    cp_module,
    cndi_module,
):
    """GPU version of rasterize_path - keeps data on GPU."""
    if not path:
        return cp_module.zeros(shape, dtype=bool)
    
    # Skip smoothing for very short paths
    if smooth and len(path) >= 4:
        compressed = compress_path(path)
        points = [(float(r), float(c)) for r, c in compressed]
        spline_points = catmull_rom_spline(points, num_segments=8)
        if smooth_iterations > 0:
            spline_points = chaikin_smooth(spline_points, smooth_iterations)
        # Rasterize on CPU then transfer (spline math is CPU-bound anyway)
        mask_cpu = rasterize_polyline(spline_points, shape)
        mask = cp_module.asarray(mask_cpu)
    else:
        # No smoothing: set pixels directly on GPU
        mask = cp_module.zeros(shape, dtype=bool)
        for r, c in path:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
    
    # GPU dilation using cupyx.scipy.ndimage (much faster than CPU)
    if bridge_width > 1:
        # Build circular structuring element
        size = 2 * bridge_width - 1
        y, x = cp_module.ogrid[-bridge_width+1:bridge_width, -bridge_width+1:bridge_width]
        struct = (x*x + y*y) < bridge_width * bridge_width
        mask = cndi_module.binary_dilation(mask, structure=struct)
    
    return mask


def compute_bridge_path(
    boundary_black: Sequence[Coord],
    outer_black: Set[Coord],
    foreground: np.ndarray,
    avoid_mask: np.ndarray,
) -> Optional[List[Coord]]:
    """Find a bridge path from island boundary to outer boundary.
    
    Tries multiple strategies:
    1. 4-conn through foreground only (cleanest)
    2. 8-conn through foreground only (more options)
    3. 8-conn through any pixel (fallback for difficult cases)
    """
    if not boundary_black or not outer_black:
        return None
        
    allowed_fg = foreground & ~avoid_mask
    
    # Try 4-connectivity through foreground first (cleanest paths)
    path = bfs_path(boundary_black, outer_black, allowed_fg, use_8conn=False)
    if path is not None:
        return path
    
    # Try 8-connectivity through foreground (more routing options)
    path = bfs_path(boundary_black, outer_black, allowed_fg, use_8conn=True)
    if path is not None:
        return path
    
    # Fallback: allow path through any pixel (even background)
    # This handles cases where foreground doesn't form a continuous path
    allowed_any = np.ones_like(foreground, dtype=bool)  # Allow ALL pixels
    path = bfs_path(boundary_black, outer_black, allowed_any, use_8conn=True)
    return path


def generate_layers(
    foreground: np.ndarray,
    layers: int,
    smooth: bool = True,
    smooth_iterations: int = 2,
    bridge_width: int = 1,
    min_island_size: int = 1,
    erode_passes: int = 1,
    bridges_per_island: int = 1,
    max_cutout_size: int = None,
    clip_to_region: bool = False,
) -> Tuple[List[np.ndarray], List[str], np.ndarray, np.ndarray]:
    """
    Generate stencil layers that combine to form the original image.
    
    - foreground: True where original image is black (design to paint)
    - Output: True = stencil material (transparent), False = hole (black, paint passes)
    - min_island_size: Minimum number of pixels for an island to be considered
    - erode_passes: Erosion passes for aggressive island detection (handles thin/diagonal walls)
    
    Layer 1: Full design with ALL bridges (most bridges, blocks paint at bridge locations)
    Layers 2-N: Only the bridge areas as holes (to fill in the bridge gaps from layer 1)
    
    When all layers are painted in sequence, bridges get filled and you get the complete image.
    """
    height, width = foreground.shape
    background = ~foreground
    
    logging.info("Scanning for interior details (islands) with erode_passes=%d...", erode_passes)
    all_islands = find_hole_components(background, erode_passes=erode_passes)  # Enclosed white regions
    islands = [isl for isl in all_islands if len(isl) >= min_island_size]
    logging.info("Found %d interior detail(s) (islands), %d after size filter (min=%d px)", 
                 len(all_islands), len(islands), min_island_size)
    
    # Create islands mask for visualization
    islands_mask = np.zeros((height, width), dtype=bool)
    for island in islands:
        for r, c in island:
            islands_mask[r, c] = True
    
    logging.info("Computing outer boundary...")
    outer_black = compute_outer_boundary_black(foreground, background)
    
    # ALWAYS include border pixels as valid targets (for fallback paths through any pixel)
    border_targets: Set[Coord] = set()
    for c in range(width):
        border_targets.add((0, c))
        border_targets.add((height - 1, c))
    for r in range(height):
        border_targets.add((r, 0))
        border_targets.add((r, width - 1))
    
    # Combine foreground boundary and border pixels
    all_targets = outer_black | border_targets
    
    if not outer_black:
        logging.warning("No foreground pixels touch the border - using border only as targets")

    warnings: List[str] = []
    failed_count = 0
    
    # Compute all bridge paths first
    bridge_masks: List[np.ndarray] = []
    for island_index, island in enumerate(tqdm(islands, desc="Building bridges", unit="island")):
        boundary_black = list(compute_boundary_black(island, foreground))
        if not boundary_black:
            logging.debug("Island %d (size=%d) has no adjacent foreground pixels", island_index, len(island))
            # Use the island pixels themselves as starting points
            boundary_black = island
        
        # Compute multiple bridges per island for stronger support
        island_bridge_count = 0
        if bridges_per_island > 1 and len(boundary_black) >= bridges_per_island:
            # Compute island centroid
            island_rows = [r for r, c in island]
            island_cols = [c for r, c in island]
            centroid_r = sum(island_rows) / len(island_rows)
            centroid_c = sum(island_cols) / len(island_cols)
            
            # Compute angle for each boundary pixel from centroid
            import math
            boundary_with_angle = []
            for r, c in boundary_black:
                angle = math.atan2(r - centroid_r, c - centroid_c)
                boundary_with_angle.append((angle, (r, c)))
            boundary_with_angle.sort(key=lambda x: x[0])
            
            # Select boundary pixels at evenly-spaced angles for maximum spread
            # Target angles: evenly distributed around the circle
            target_angles = [(-math.pi + (2 * math.pi * i / bridges_per_island)) for i in range(bridges_per_island)]
            
            # For each target angle, find the closest boundary pixel
            selected_boundaries: List[List[Coord]] = []
            used_indices: Set[int] = set()
            
            for target_angle in target_angles:
                best_idx = -1
                best_diff = float('inf')
                for idx, (angle, _) in enumerate(boundary_with_angle):
                    if idx in used_indices:
                        continue
                    # Angular distance (handle wrap-around)
                    diff = abs(angle - target_angle)
                    if diff > math.pi:
                        diff = 2 * math.pi - diff
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = idx
                
                if best_idx >= 0:
                    used_indices.add(best_idx)
                    _, coord = boundary_with_angle[best_idx]
                    selected_boundaries.append([coord])
            
            used_targets: Set[Coord] = set()
            
            for group_boundary in selected_boundaries:
                if not group_boundary:
                    continue
                
                # Exclude already-used targets to encourage diverse bridges
                available_targets = all_targets - used_targets
                if not available_targets:
                    available_targets = all_targets
                
                path = compute_bridge_path(group_boundary, available_targets, foreground,
                                          np.zeros_like(foreground, dtype=bool))
                
                if path is not None:
                    island_bridge_count += 1
                    # Mark path endpoints as used
                    if path:
                        used_targets.add(path[-1])
                    
                    path_mask = rasterize_path(
                        path,
                        foreground.shape,
                        smooth=smooth,
                        smooth_iterations=smooth_iterations,
                        bridge_width=bridge_width,
                    )
                    bridge_masks.append(path_mask)
            
            if island_bridge_count > 0:
                logging.debug("Island %d (size=%d) bridged with %d paths", island_index, len(island), island_bridge_count)
            else:
                failed_count += 1
                logging.debug("Island %d (size=%d) could not find any bridge path", island_index, len(island))
                bridge_masks.append(np.zeros((height, width), dtype=bool))
        else:
            # Single bridge (original behavior)
            path = compute_bridge_path(boundary_black, all_targets, foreground,
                                      np.zeros_like(foreground, dtype=bool))
            
            if path is None:
                failed_count += 1
                logging.debug("Island %d (size=%d) could not find bridge path", island_index, len(island))
                bridge_masks.append(np.zeros((height, width), dtype=bool))
                continue
            
            logging.debug("Island %d (size=%d) bridged with path of length %d", island_index, len(island), len(path))
            path_mask = rasterize_path(
                path,
                foreground.shape,
                smooth=smooth,
                smooth_iterations=smooth_iterations,
                bridge_width=bridge_width,
            )
            bridge_masks.append(path_mask)
        
    # Assign regular bridges to fill-in layers (layers 2..N)
    fill_layers = max(1, layers)
    bridge_assignments: List[List[int]] = [[] for _ in range(fill_layers)]
    for bridge_index in range(len(bridge_masks)):
        fill_layer = bridge_index % fill_layers
        bridge_assignments[fill_layer].append(bridge_index)

    # All-regular-bridges union (used as input mask for structural calculation)
    all_regular_bridges = np.zeros((height, width), dtype=bool)
    for bm in bridge_masks:
        all_regular_bridges = all_regular_bridges | bm

    # Structural bridges: computed on foreground AFTER regular bridges are removed,
    # kept in their own list so they get a dedicated fill layer.
    structural_masks: List[np.ndarray] = []
    structural_mask = np.zeros((height, width), dtype=bool)
    if max_cutout_size is not None:
        remaining_fg = foreground & ~all_regular_bridges
        structural_masks = _compute_structural_bridges(remaining_fg, max_cutout_size, bridge_width, height, width, clip_to_region=clip_to_region)
        for sm in structural_masks:
            structural_mask = structural_mask | sm

    total_layers = 1 + fill_layers + (1 if structural_masks else 0)
    layer_images: List[np.ndarray] = []

    # Layer 1: full design + regular bridges + structural bridges (all solid)
    logging.info("Building layer 1/%d (main design with bridges)", total_layers)
    layer1 = ~foreground.copy()
    layer1 = layer1 | all_regular_bridges | structural_mask
    layer_images.append(layer1)

    # Layers 2..N: regular bridge fill-in (round-robin)
    for fill_idx in range(fill_layers):
        if layers == 1:
            break
        layer_num = fill_idx + 2
        logging.info("Building layer %d/%d (regular bridge fill-in)", layer_num, total_layers)
        layer_img = np.ones((height, width), dtype=bool)
        for bridge_index in bridge_assignments[fill_idx]:
            layer_img = layer_img & ~bridge_masks[bridge_index]
        layer_images.append(layer_img)

    # Dedicated structural fill layer (last)
    if structural_masks:
        logging.info("Building layer %d/%d (structural bridge fill-in)", total_layers, total_layers)
        struct_layer = np.ones((height, width), dtype=bool)
        for sm in structural_masks:
            struct_layer = struct_layer & ~sm
        layer_images.append(struct_layer)

    if failed_count > 0:
        warnings.append(f"{failed_count} island(s) could not get a bridge.")

    # Combine all bridge masks (regular + structural) for visualization
    all_bridges_mask = np.zeros((height, width), dtype=bool)
    for bridge_mask in bridge_masks:
        all_bridges_mask = all_bridges_mask | bridge_mask
    all_bridges_mask = all_bridges_mask | structural_mask

    return layer_images, warnings, islands_mask, all_bridges_mask, structural_mask


def _compute_structural_bridges(
    foreground_np: np.ndarray,
    max_cutout_size: int,
    bridge_width: int,
    height: int,
    width: int,
    clip_to_region: bool = False,
) -> List[np.ndarray]:
    """Iteratively split all foreground connected-components larger than
    *max_cutout_size* pixels by inserting bisecting chords.

    For each oversized region, scans 36 chord angles using a median-projection
    sweep to find the best chord regardless of region shape.  A chord is placed
    at the median of all pixel projections onto the chord's normal — this
    guarantees ~50/50 pixel balance and works for convex, concave, C/U/ring
    shapes without needing the centroid to lie inside the region.

    Regions are never permanently discarded: if no angle produces a clean split
    the shortest available chord is inserted anyway so the region keeps shrinking.

    Returns the ordered list of rasterized bridge masks (numpy bool arrays).
    """
    import math
    from scipy import ndimage as _ndi
    _structure_8 = np.ones((3, 3), dtype=bool)
    _structure_8_3 = np.ones((3, 3), dtype=bool)  # reused for cropped-box label checks
    working = foreground_np.copy()
    bridges: List[np.ndarray] = []
    max_iterations = 1000
    N_ANGLES = 36  # 5-degree steps

    # Precompute angle tables once
    _thetas = np.array([math.pi * i / N_ANGLES for i in range(N_ANGLES)])
    _cdrs = np.cos(_thetas)   # chord direction row component
    _cdcs = np.sin(_thetas)   # chord direction col component
    _ndrs = -np.sin(_thetas)  # normal direction row component
    _ndcs = np.cos(_thetas)   # normal direction col component

    def _best_chord(fg_coords: np.ndarray,
                    region_mask: np.ndarray) -> Optional[np.ndarray]:
        """Return the best bisecting chord mask for this region.

        All per-pixel work is done inside the region's bounding box (+ padding)
        so cost scales with region area, not image area.
        """
        # ---- bounding box (with padding for dilation) -------------------------
        pad = bridge_width + 1
        r0 = max(0, int(fg_coords[:, 0].min()) - pad)
        r1 = min(height, int(fg_coords[:, 0].max()) + pad + 1)
        c0 = max(0, int(fg_coords[:, 1].min()) - pad)
        c1 = min(width,  int(fg_coords[:, 1].max()) + pad + 1)
        bh, bw = r1 - r0, c1 - c0

        # Region mask cropped to bounding box
        rm_crop = region_mask[r0:r1, c0:c1]

        # Pixel coordinates relative to bounding box
        rs = fg_coords[:, 0].astype(float) - r0
        cs = fg_coords[:, 1].astype(float) - c0

        best_split: Optional[np.ndarray] = None
        best_split_len = float('inf')
        best_any: Optional[np.ndarray] = None
        best_any_len = float('inf')

        for i in range(N_ANGLES):
            cdr, cdc = _cdrs[i], _cdcs[i]
            ndr, ndc = _ndrs[i], _ndcs[i]

            # Project all pixels onto the normal; chord sits at the median
            projs_n = rs * ndr + cs * ndc
            median_n = float(np.median(projs_n))

            # Pixels nearest to the median projection line form the chord spine
            dist_to_median = np.abs(projs_n - median_n)
            tol = max(0.5, dist_to_median.min() + 0.5)
            near = dist_to_median <= tol
            near_rs = rs[near]
            near_cs = cs[near]

            # Endpoints: extremes along chord direction among near pixels
            projs_c = near_rs * cdr + near_cs * cdc
            imin, imax = int(np.argmin(projs_c)), int(np.argmax(projs_c))
            pt_a = (int(near_rs[imin]), int(near_cs[imin]))
            pt_b = (int(near_rs[imax]), int(near_cs[imax]))
            if pt_a == pt_b:
                continue

            # Rasterize and clip — all in cropped coords
            chord_thin = rasterize_polyline([pt_a, pt_b], (bh, bw)) & rm_crop
            if not chord_thin.any():
                continue

            # Dilate; optionally clip back to region
            if bridge_width > 1:
                from scipy import ndimage as _ndi2
                diam = 2 * bridge_width - 1
                cy, cx = np.ogrid[:diam, :diam]
                ctr = bridge_width - 1
                struct = ((cx - ctr) ** 2 + (cy - ctr) ** 2) < bridge_width * bridge_width
                mid_crop = _ndi2.binary_dilation(chord_thin, structure=struct)
            else:
                mid_crop = chord_thin
            if clip_to_region:
                mid_crop = mid_crop & rm_crop
            if not mid_crop.any():
                continue

            chord_len = math.hypot(pt_a[0] - pt_b[0], pt_a[1] - pt_b[1])

            # Split check — label only the small cropped region
            after_crop = rm_crop & ~mid_crop
            _, n_after = _ndi.label(after_crop, structure=_structure_8_3)
            if n_after >= 2 and chord_len < best_split_len:
                best_split_len = chord_len
                # Expand best candidate back to full image coords lazily
                best_split = (r0, r1, c0, c1, mid_crop.copy())
            if chord_len < best_any_len:
                best_any_len = chord_len
                best_any = (r0, r1, c0, c1, mid_crop.copy())

        # Materialise winner into a full-image mask
        winner = best_split if best_split is not None else best_any
        if winner is None:
            return None
        wr0, wr1, wc0, wc1, crop_mask = winner
        full = np.zeros((height, width), dtype=bool)
        full[wr0:wr1, wc0:wc1] = crop_mask
        return full

    for iteration in range(1, max_iterations + 1):
        labels, count = _ndi.label(working, structure=_structure_8)
        oversized = sorted(
            [np.argwhere(labels == lbl)
             for lbl in range(1, count + 1)
             if (labels == lbl).sum() > max_cutout_size],
            key=len, reverse=True,
        )
        if not oversized:
            logging.info(
                "max-cutout-size: done after %d iteration(s), all regions within threshold",
                iteration - 1,
            )
            break
        logging.info(
            "max-cutout-size iteration %d: %d region(s) above %d px",
            iteration, len(oversized), max_cutout_size,
        )
        # All oversized regions are disjoint connected components, so their
        # bridges (clipped to each region_mask) cannot affect one another.
        # Process all of them in a single pass and relabel only once per round.
        any_inserted = False
        for fg_coords in tqdm(oversized, desc=f"Structural bridges (iter {iteration})", unit="region", leave=False):
            region_mask = np.zeros((height, width), dtype=bool)
            region_mask[fg_coords[:, 0], fg_coords[:, 1]] = True
            mid_mask = _best_chord(fg_coords, region_mask)
            if mid_mask is None:
                logging.debug("  region size=%d -> no chord found, skipping", len(fg_coords))
                continue
            bridges.append(mid_mask)
            working &= ~mid_mask
            any_inserted = True
            logging.debug("  region size=%d -> bridge inserted", len(fg_coords))

        if not any_inserted:
            logging.warning(
                "max-cutout-size: no chord could be inserted for any oversized region; stopping"
            )
            break
    else:
        logging.warning(
            "max-cutout-size hit iteration cap (%d), some regions may still be oversized",
            max_iterations,
        )
    return bridges



def _gpu_compute_distance_field(targets_mask, foreground, cp_module, cndi_module):
    """Compute geodesic distance field from targets through foreground.

    Returns distance field where each foreground pixel has its shortest distance
    to any target pixel (only traveling through foreground).
    Uses 8-connectivity for diagonal paths.
    """
    height, width = foreground.shape
    INF = height * width + 1
    dist = cp_module.full(foreground.shape, INF, dtype=cp_module.int32)
    dist[targets_mask] = 0
    traversable = foreground | targets_mask
    changed = True
    while changed:
        changed = False
        
        # 4-connectivity (orthogonal) - distance +1
        # Up neighbor
        new_dist = cp_module.full_like(dist, INF)
        new_dist[1:, :] = dist[:-1, :] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Down neighbor
        new_dist = cp_module.full_like(dist, INF)
        new_dist[:-1, :] = dist[1:, :] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Left neighbor
        new_dist = cp_module.full_like(dist, INF)
        new_dist[:, 1:] = dist[:, :-1] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Right neighbor
        new_dist = cp_module.full_like(dist, INF)
        new_dist[:, :-1] = dist[:, 1:] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # 8-connectivity (diagonal) - also distance +1 (Chebyshev distance)
        # Up-left
        new_dist = cp_module.full_like(dist, INF)
        new_dist[1:, 1:] = dist[:-1, :-1] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Up-right
        new_dist = cp_module.full_like(dist, INF)
        new_dist[1:, :-1] = dist[:-1, 1:] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Down-left
        new_dist = cp_module.full_like(dist, INF)
        new_dist[:-1, 1:] = dist[1:, :-1] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
        
        # Down-right
        new_dist = cp_module.full_like(dist, INF)
        new_dist[:-1, :-1] = dist[1:, 1:] + 1
        improved = (new_dist < dist) & traversable
        if cp_module.any(improved):
            dist = cp_module.where(improved, new_dist, dist)
            changed = True
    
    return dist


def _trace_path_from_distance_cpu(start_r: int, start_c: int, dist_cpu: np.ndarray, targets_cpu: np.ndarray) -> List[Coord]:
    """Trace a path from start to targets by following decreasing distance values (CPU version)."""
    height, width = dist_cpu.shape
    INF = height * width + 1
    
    path = [(start_r, start_c)]
    r, c = start_r, start_c
    
    # 8-connectivity neighbors (including diagonals)
    neighbors_8 = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    # Follow gradient descent
    max_steps = height * width  # Safety limit
    for _ in range(max_steps):
        if targets_cpu[r, c]:
            break
        
        # Find neighbor with smallest distance (8-connectivity)
        best_dist = dist_cpu[r, c]
        best_r, best_c = r, c
        
        for dr, dc in neighbors_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                if dist_cpu[nr, nc] < best_dist:
                    best_dist = dist_cpu[nr, nc]
                    best_r, best_c = nr, nc
        
        if best_r == r and best_c == c:
            # Can't make progress - stuck
            break
        
        r, c = best_r, best_c
        path.append((r, c))
    
    return path


def _gpu_bfs_path(
    sources_mask,
    targets_mask,
    allowed_mask,
    use_8conn: bool = False,
):
    """GPU BFS to find path from sources to targets.
    
    Sources are always valid starting points regardless of allowed_mask.
    """
    import cupy as cp

    if not cp.any(sources_mask):
        return None
    if not cp.any(targets_mask):
        return None

    # Start from ALL source pixels (not just those in allowed_mask)
    frontier = sources_mask.copy()
    visited = frontier.copy()
    pred = cp.zeros(allowed_mask.shape, dtype=cp.int8)
    
    # Allow expansion to allowed pixels OR target pixels
    traversable = allowed_mask | targets_mask

    while True:
        if cp.any(frontier & targets_mask):
            break

        new_frontier = cp.zeros_like(frontier)

        # Up (from below)
        cand = cp.zeros_like(frontier)
        cand[:-1, :] = frontier[1:, :]
        new = cand & traversable & ~visited
        if cp.any(new):
            pred = cp.where(new, 2, pred)
            visited |= new
            new_frontier |= new

        # Down (from above)
        cand = cp.zeros_like(frontier)
        cand[1:, :] = frontier[:-1, :]
        new = cand & traversable & ~visited
        if cp.any(new):
            pred = cp.where(new, 1, pred)
            visited |= new
            new_frontier |= new

        # Left (from right)
        cand = cp.zeros_like(frontier)
        cand[:, :-1] = frontier[:, 1:]
        new = cand & traversable & ~visited
        if cp.any(new):
            pred = cp.where(new, 4, pred)
            visited |= new
            new_frontier |= new

        # Right (from left)
        cand = cp.zeros_like(frontier)
        cand[:, 1:] = frontier[:, :-1]
        new = cand & traversable & ~visited
        if cp.any(new):
            pred = cp.where(new, 3, pred)
            visited |= new
            new_frontier |= new

        # Diagonal directions (8-connectivity)
        if use_8conn:
            # Up-Left (from down-right)
            cand = cp.zeros_like(frontier)
            cand[:-1, :-1] = frontier[1:, 1:]
            new = cand & traversable & ~visited
            if cp.any(new):
                pred = cp.where(new, 5, pred)
                visited |= new
                new_frontier |= new

            # Up-Right (from down-left)
            cand = cp.zeros_like(frontier)
            cand[:-1, 1:] = frontier[1:, :-1]
            new = cand & traversable & ~visited
            if cp.any(new):
                pred = cp.where(new, 6, pred)
                visited |= new
                new_frontier |= new

            # Down-Left (from up-right)
            cand = cp.zeros_like(frontier)
            cand[1:, :-1] = frontier[:-1, 1:]
            new = cand & traversable & ~visited
            if cp.any(new):
                pred = cp.where(new, 7, pred)
                visited |= new
                new_frontier |= new

            # Down-Right (from up-left)
            cand = cp.zeros_like(frontier)
            cand[1:, 1:] = frontier[:-1, :-1]
            new = cand & traversable & ~visited
            if cp.any(new):
                pred = cp.where(new, 8, pred)
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
        elif direction == 5:  # Up-Left
            r -= 1
            c -= 1
        elif direction == 6:  # Up-Right
            r -= 1
            c += 1
        elif direction == 7:  # Down-Left
            r += 1
            c -= 1
        elif direction == 8:  # Down-Right
            r += 1
            c += 1
        path.append((r, c))

    path.reverse()
    return path


def generate_layers_gpu(
    foreground,
    layers: int,
    smooth: bool = True,
    smooth_iterations: int = 2,
    bridge_width: int = 1,
    min_island_size: int = 1,
    erode_passes: int = 1,
    bridges_per_island: int = 1,
    max_cutout_size: int = None,
    clip_to_region: bool = False,
) -> Tuple[List, List[str], Any, Any]:
    """
    GPU version: Generate stencil layers that combine to form the original image.
    
    - foreground: True where original image is black (design to paint)
    - Output: True = stencil material (transparent), False = hole (black, paint passes)
    - min_island_size: Minimum number of pixels for an island to be considered
    - erode_passes: Erosion passes for aggressive island detection
    
    Layer 1: Full design with ALL bridges (most bridges, blocks paint at bridge locations)
    Layers 2-N: Only the bridge areas as holes (to fill in the bridge gaps from layer 1)
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cndi

    # 4-connectivity for labeling - paint can't flow through diagonal gaps
    # This ensures diagonally-connected regions are separate islands
    structure_4conn = cp.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        dtype=cp.uint8,
    )
    # 8-connectivity for dilation (finding boundary pixels)
    structure_8conn = cp.array(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        dtype=cp.uint8,
    )

    height, width = foreground.shape
    background = ~foreground
    
    # Strategy 1: Baseline 4-conn labeling on original background
    result = cndi.label(background, structure=structure_4conn)
    labels_baseline, num_baseline = cast(Tuple[Any, int], result)
    
    if num_baseline == 0:
        islands_mask = cp.zeros(foreground.shape, dtype=bool)
        empty_bridges = cp.zeros(foreground.shape, dtype=bool)
        return [~foreground for _ in range(layers)], [], islands_mask, empty_bridges
    
    # Find which labels touch the border
    border_labels_baseline = cp.unique(
        cp.concatenate([labels_baseline[0, :], labels_baseline[-1, :], 
                       labels_baseline[:, 0], labels_baseline[:, -1]])
    )
    border_labels_baseline = border_labels_baseline[border_labels_baseline != 0]
    
    # Baseline islands = labels that don't touch border
    all_labels_baseline = cp.arange(1, num_baseline + 1, dtype=cp.int32)
    baseline_island_labels = all_labels_baseline[~cp.isin(all_labels_baseline, border_labels_baseline)]
    
    # Create baseline island mask
    islands_mask = cp.zeros(foreground.shape, dtype=bool)
    for label_id in cp.asnumpy(baseline_island_labels):
        islands_mask = islands_mask | (labels_baseline == label_id)
    
    # Strategy 2: Dilate foreground to find additional islands separated by thin/diagonal walls
    if erode_passes > 0:
        # Dilate foreground (this shrinks background)
        dilated_fg = foreground.copy()
        for _ in range(erode_passes):
            dilated_fg = cndi.binary_dilation(dilated_fg, structure=structure_8conn)
        
        shrunk_bg = background & ~dilated_fg
        
        # Label the shrunk background
        result = cndi.label(shrunk_bg, structure=structure_4conn)
        labels_shrunk, num_shrunk = cast(Tuple[Any, int], result)
        
        if num_shrunk > 0:
            # Find which shrunk labels touch the border
            border_labels_shrunk = cp.unique(
                cp.concatenate([labels_shrunk[0, :], labels_shrunk[-1, :],
                               labels_shrunk[:, 0], labels_shrunk[:, -1]])
            )
            border_labels_shrunk = border_labels_shrunk[border_labels_shrunk != 0]
            
            # Isolated shrunk regions = potential additional islands
            all_labels_shrunk = cp.arange(1, num_shrunk + 1, dtype=cp.int32)
            isolated_shrunk_labels = all_labels_shrunk[~cp.isin(all_labels_shrunk, border_labels_shrunk)]
            
            # Add isolated pixels from shrunk space to islands mask
            for label_id in cp.asnumpy(isolated_shrunk_labels):
                isolated_region = labels_shrunk == label_id
                islands_mask = islands_mask | isolated_region
    
    # Re-label the combined islands mask with 8-conn for grouping
    result = cndi.label(islands_mask, structure=structure_8conn)
    labels, num = cast(Tuple[Any, int], result)
    
    if num == 0:
        islands_mask = cp.zeros(foreground.shape, dtype=bool)
        empty_bridges = cp.zeros(foreground.shape, dtype=bool)
        return [~foreground for _ in range(layers)], [], islands_mask, empty_bridges
    
    # Filter by island size
    island_labels_filtered = []
    total_islands = num
    for label_id in range(1, num + 1):
        island_size = int(cp.sum(labels == label_id))
        if island_size >= min_island_size:
            island_labels_filtered.append(int(label_id))
    
    island_labels_list = island_labels_filtered
    islands_count = len(island_labels_list)
    
    # Update islands_mask to only include filtered islands
    islands_mask = cp.zeros(foreground.shape, dtype=bool)
    for label_id in island_labels_list:
        islands_mask = islands_mask | (labels == label_id)

    logging.info("Found %d interior detail(s) (islands), %d after size filter (min=%d px), erode_passes=%d (GPU)", 
                 total_islands, islands_count, min_island_size, erode_passes)

    # Warn if few islands - CPU may be faster due to reduced transfer overhead
    if islands_count > 0 and islands_count < 50:
        logging.warning("Few islands detected (%d). CPU mode (without --gpu) may be faster for small island counts.", islands_count)

    # Compute outer boundary: foreground pixels adjacent to background that touches border
    # First, find background pixels that can reach the border (outside region)
    outside_bg = background.copy()
    # Flood fill from border - use iterative dilation constrained to background
    border_mask = cp.zeros(foreground.shape, dtype=bool)
    border_mask[0, :] = background[0, :]
    border_mask[-1, :] = background[-1, :]
    border_mask[:, 0] = background[:, 0]
    border_mask[:, -1] = background[:, -1]
    
    # Expand from border through background
    outside_bg = border_mask.copy()
    prev_count = 0
    while True:
        expanded = cndi.binary_dilation(outside_bg, structure=structure_4conn) & background
        outside_bg = outside_bg | expanded
        curr_count = int(cp.sum(outside_bg))
        if curr_count == prev_count:
            break
        prev_count = curr_count
    
    outer_black = foreground & cndi.binary_dilation(outside_bg, structure=structure_8conn)
    
    # ALWAYS include border pixels as valid targets (for fallback paths)
    border_targets = cp.zeros_like(foreground, dtype=bool)
    border_targets[0, :] = True
    border_targets[-1, :] = True
    border_targets[:, 0] = True
    border_targets[:, -1] = True
    
    # Combine foreground boundary and border pixels
    all_targets = outer_black | border_targets
    
    if not cp.any(outer_black):
        logging.warning("No foreground pixels touch the border - using border only as targets (GPU)")

    warnings: List[str] = []
    failed_count = 0
    
    # Get labels on CPU for efficient iteration
    labels_cpu = cp.asnumpy(labels)
    foreground_cpu = cp.asnumpy(foreground)
    targets_cpu = cp.asnumpy(all_targets)
    height_img, width_img = foreground.shape
    INF = height_img * width_img + 1
    
    # Pre-compute bounding boxes for all islands (much faster than per-island == comparison)
    from scipy import ndimage as ndi_cpu
    logging.info("Pre-computing island bounding boxes...")
    island_slices = ndi_cpu.find_objects(labels_cpu)
    
    # Pre-allocate per-layer bridge accumulators
    fill_layers = max(1, layers)
    fill_layer_accums = [np.zeros((height_img, width_img), dtype=bool) for _ in range(fill_layers)]
    all_bridges_accum = np.zeros((height_img, width_img), dtype=bool)
    
    structure_8conn_cpu = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
    
    # Group islands by their fill layer assignment
    logging.info("Assigning islands to layers...")
    islands_by_layer: Dict[int, List[int]] = {i: [] for i in range(fill_layers)}
    for island_index, label_id in enumerate(island_labels_list):
        fill_layer_idx = island_index % fill_layers
        islands_by_layer[fill_layer_idx].append(label_id)
    
    # Process each fill layer separately to prevent intra-layer bridge crossings
    # By updating the "connected" network after each bridge, new bridges route around existing ones
    import math
    for fill_layer_idx in range(fill_layers):
        layer_islands = islands_by_layer[fill_layer_idx]
        if not layer_islands:
            continue
        
        logging.info("Bridge layer %d/%d (%d islands)...", fill_layer_idx + 1, fill_layers, len(layer_islands))
        
        # Initialize "connected" network for this layer: targets + border
        # As bridges are added, they become part of the connected network
        connected_cpu = targets_cpu.copy()
        connected_gpu = cp.asarray(connected_cpu)
        
        # Compute initial distance field on GPU (much faster)
        dist_field_gpu = _gpu_compute_distance_field(connected_gpu, foreground, cp, cndi)
        dist_field_cpu = cp.asnumpy(dist_field_gpu)
        del dist_field_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        bridges_since_recompute = 0
        recompute_interval = 100  # Recompute distance field every N bridges
        # batch_connected tracks bridges drawn in current batch - paths terminate on these too
        batch_connected = connected_cpu.copy()
        # batch_connected_dilated expands batch_connected by bridge width for termination checks
        # This ensures thin traced paths stop when they get close to thick bridges
        # Use full bridge_width (not radius) for more safety margin
        bridge_dilation = bridge_width
        if bridge_dilation > 0:
            batch_connected_dilated = ndi_cpu.binary_dilation(batch_connected, iterations=bridge_dilation)
        else:
            batch_connected_dilated = batch_connected
        
        for label_id in tqdm(layer_islands, desc=f"Building bridges (layer {fill_layer_idx + 2})", unit="island"):
            # Use precomputed bounding box for this island
            bbox = island_slices[label_id - 1]
            if bbox is None:
                continue
            
            # Add padding for dilation
            r_slice, c_slice = bbox
            pad = 1
            r_start = max(0, r_slice.start - pad)
            r_stop = min(height_img, r_slice.stop + pad)
            c_start = max(0, c_slice.start - pad)
            c_stop = min(width_img, c_slice.stop + pad)
            
            # Extract small region
            labels_region = labels_cpu[r_start:r_stop, c_start:c_stop]
            foreground_region = foreground_cpu[r_start:r_stop, c_start:c_stop]
            island_mask_region = labels_region == label_id
            
            # Dilate on small region
            dilated_region = ndi_cpu.binary_dilation(island_mask_region, structure=structure_8conn_cpu)
            boundary_black_region = foreground_region & dilated_region
            
            if not np.any(boundary_black_region):
                boundary_black_region = island_mask_region
            
            # Get coordinates in global space
            island_coords_local = np.argwhere(island_mask_region)
            island_coords_cpu = island_coords_local + np.array([r_start, c_start])
            
            boundary_coords_local = np.argwhere(boundary_black_region)
            boundary_coords_cpu = boundary_coords_local + np.array([r_start, c_start])
            
            island_bridge_count = 0
            
            if bridges_per_island > 1 and len(boundary_coords_cpu) >= bridges_per_island:
                # Compute centroid for multi-bridge placement
                centroid_r = float(island_coords_cpu[:, 0].mean())
                centroid_c = float(island_coords_cpu[:, 1].mean())
                
                # Compute angle for each boundary pixel
                boundary_with_angle = []
                for coord in boundary_coords_cpu:
                    r, c = int(coord[0]), int(coord[1])
                    angle = math.atan2(r - centroid_r, c - centroid_c)
                    boundary_with_angle.append((angle, (r, c)))
                boundary_with_angle.sort(key=lambda x: x[0])
                
                # Select boundary pixels at evenly-spaced angles
                target_angles = [(-math.pi + (2 * math.pi * i / bridges_per_island)) for i in range(bridges_per_island)]
                
                selected_boundaries: List[Tuple[int, int]] = []
                used_indices: Set[int] = set()
                
                for target_angle in target_angles:
                    best_idx = -1
                    best_diff = float('inf')
                    for idx, (angle, _) in enumerate(boundary_with_angle):
                        if idx in used_indices:
                            continue
                        diff = abs(angle - target_angle)
                        if diff > math.pi:
                            diff = 2 * math.pi - diff
                        if diff < best_diff:
                            best_diff = diff
                            best_idx = idx
                    
                    if best_idx >= 0:
                        used_indices.add(best_idx)
                        _, coord = boundary_with_angle[best_idx]
                        selected_boundaries.append(coord)
                
                for coord in selected_boundaries:
                    r, c = coord
                    if dist_field_cpu[r, c] >= INF:
                        continue
                    
                    # Check against dilated batch_connected (accounts for bridge width)
                    if batch_connected_dilated[r, c]:
                        island_bridge_count += 1
                        continue
                    
                    # Trace path - terminate on dilated batch_connected
                    path = _trace_path_from_distance_cpu(r, c, dist_field_cpu, batch_connected_dilated)
                    
                    if path and len(path) > 1:
                        island_bridge_count += 1
                        bridges_since_recompute += 1
                        
                        if bridge_width > 1:
                            draw_thick_path_into_mask(fill_layer_accums[fill_layer_idx], path, bridge_width, smooth, smooth_iterations)
                            draw_thick_path_into_mask(all_bridges_accum, path, bridge_width, smooth, smooth_iterations)
                            draw_thick_path_into_mask(connected_cpu, path, bridge_width, smooth, smooth_iterations)
                            draw_thick_path_into_mask(batch_connected, path, bridge_width, smooth, smooth_iterations)
                            # Update dilated version
                            draw_thick_path_into_mask(batch_connected_dilated, path, bridge_width + bridge_dilation * 2, smooth, smooth_iterations)
                        else:
                            draw_path_into_mask(fill_layer_accums[fill_layer_idx], path, smooth, smooth_iterations)
                            draw_path_into_mask(all_bridges_accum, path, smooth, smooth_iterations)
                            draw_path_into_mask(connected_cpu, path, smooth, smooth_iterations)
                            draw_path_into_mask(batch_connected, path, smooth, smooth_iterations)
                            batch_connected_dilated = batch_connected.copy()
                        
                        # Periodically recompute distance field to route around new bridges
                        if bridges_since_recompute >= recompute_interval:
                            connected_gpu = cp.asarray(connected_cpu)
                            dist_field_gpu = _gpu_compute_distance_field(connected_gpu, foreground, cp, cndi)
                            dist_field_cpu = cp.asnumpy(dist_field_gpu)
                            del dist_field_gpu, connected_gpu
                            cp.get_default_memory_pool().free_all_blocks()
                            bridges_since_recompute = 0
                            batch_connected = connected_cpu.copy()
                            if bridge_dilation > 0:
                                batch_connected_dilated = ndi_cpu.binary_dilation(batch_connected, iterations=bridge_dilation)
                            else:
                                batch_connected_dilated = batch_connected
                
                if island_bridge_count > 0:
                    logging.debug("Island %d (size=%d) bridged with %d paths", 
                                 label_id, len(island_coords_cpu), island_bridge_count)
                else:
                    failed_count += 1
                    logging.debug("Island %d (size=%d) could not find any bridge path", 
                                 label_id, len(island_coords_cpu))
            else:
                # Single bridge - find boundary pixel with minimum distance
                best_r, best_c = -1, -1
                best_dist = INF
                for coord in boundary_coords_cpu:
                    r, c = int(coord[0]), int(coord[1])
                    if dist_field_cpu[r, c] < best_dist:
                        best_dist = dist_field_cpu[r, c]
                        best_r, best_c = r, c
                
                if best_r < 0 or best_dist >= INF:
                    failed_count += 1
                    logging.debug("Island %d (size=%d) could not find bridge path", 
                                 label_id, len(island_coords_cpu))
                    continue
                
                # Check against dilated batch_connected (accounts for bridge width)
                if batch_connected_dilated[best_r, best_c]:
                    logging.debug("Island %d (size=%d) already connected to boundary", 
                                 label_id, len(island_coords_cpu))
                    continue
                
                # Trace path - terminate on dilated batch_connected
                path = _trace_path_from_distance_cpu(best_r, best_c, dist_field_cpu, batch_connected_dilated)
                
                if not path or len(path) < 2:
                    failed_count += 1
                    logging.debug("Island %d (size=%d) could not trace bridge path", 
                                 label_id, len(island_coords_cpu))
                    continue
                
                logging.debug("Island %d (size=%d) bridged with path of length %d", 
                             label_id, len(island_coords_cpu), len(path))
                
                bridges_since_recompute += 1
                
                if bridge_width > 1:
                    draw_thick_path_into_mask(fill_layer_accums[fill_layer_idx], path, bridge_width, smooth, smooth_iterations)
                    draw_thick_path_into_mask(all_bridges_accum, path, bridge_width, smooth, smooth_iterations)
                    draw_thick_path_into_mask(connected_cpu, path, bridge_width, smooth, smooth_iterations)
                    draw_thick_path_into_mask(batch_connected, path, bridge_width, smooth, smooth_iterations)
                    # Update dilated version
                    draw_thick_path_into_mask(batch_connected_dilated, path, bridge_width + bridge_dilation * 2, smooth, smooth_iterations)
                else:
                    draw_path_into_mask(fill_layer_accums[fill_layer_idx], path, smooth, smooth_iterations)
                    draw_path_into_mask(all_bridges_accum, path, smooth, smooth_iterations)
                    draw_path_into_mask(connected_cpu, path, smooth, smooth_iterations)
                    draw_path_into_mask(batch_connected, path, smooth, smooth_iterations)
                    batch_connected_dilated = batch_connected.copy()
                
                # Periodically recompute distance field to route around new bridges
                if bridges_since_recompute >= recompute_interval:
                    connected_gpu = cp.asarray(connected_cpu)
                    dist_field_gpu = _gpu_compute_distance_field(connected_gpu, foreground, cp, cndi)
                    dist_field_cpu = cp.asnumpy(dist_field_gpu)
                    del dist_field_gpu, connected_gpu
                    cp.get_default_memory_pool().free_all_blocks()
                    bridges_since_recompute = 0
                    batch_connected = connected_cpu.copy()
                    if bridge_dilation > 0:
                        batch_connected_dilated = ndi_cpu.binary_dilation(batch_connected, iterations=bridge_dilation)
                    else:
                        batch_connected_dilated = batch_connected
            
    # Structural bridges: computed on foreground AFTER regular bridges are removed.
    # They are NOT mixed into fill_layer_accums — they get their own dedicated layer.
    structural_accum = np.zeros((height_img, width_img), dtype=bool)
    if max_cutout_size is not None:
        remaining_fg_cpu = foreground_cpu & ~all_bridges_accum
        for mid_mask_cpu in _compute_structural_bridges(
            remaining_fg_cpu, max_cutout_size, bridge_width, height_img, width_img,
            clip_to_region=clip_to_region
        ):
            structural_accum |= mid_mask_cpu

    # Convert accumulated masks to GPU for layer assembly
    fill_layer_accums_gpu = [cp.asarray(acc) for acc in fill_layer_accums]
    all_bridges_gpu = cp.asarray(all_bridges_accum)
    structural_gpu = cp.asarray(structural_accum)
    has_structural = bool(structural_accum.any())

    total_layers = 1 + fill_layers + (1 if has_structural else 0)
    layer_images: List = []

    # Layer 1: full design + regular bridges + structural bridges (all solid)
    logging.info("Building layer 1/%d (main design with bridges) (GPU)", total_layers)
    layer1 = ~foreground.copy()
    layer1 = layer1 | all_bridges_gpu | structural_gpu
    layer_images.append(layer1)

    # Layers 2..N: regular bridge fill-in (round-robin)
    for fill_idx in range(fill_layers):
        if layers == 1:
            break
        layer_num = fill_idx + 2
        logging.info("Building layer %d/%d (regular bridge fill-in) (GPU)", layer_num, total_layers)

        layer_img = ~fill_layer_accums_gpu[fill_idx]

        # Verify no islands (enclosed stencil regions not connected to border)
        layer_img_cpu = cp.asnumpy(layer_img)
        structure_8conn = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool)
        result = ndi_cpu.label(layer_img_cpu, structure=structure_8conn)
        labeled_verify, num_verify = cast(Tuple[Any, int], result)
        border_labels_verify = set()
        border_labels_verify.update(labeled_verify[0, :].flatten())
        border_labels_verify.update(labeled_verify[-1, :].flatten())
        border_labels_verify.update(labeled_verify[:, 0].flatten())
        border_labels_verify.update(labeled_verify[:, -1].flatten())
        border_labels_verify.discard(0)
        remaining_islands = sum(1 for lbl in range(1, num_verify+1) if lbl not in border_labels_verify)
        if remaining_islands > 0:
            logging.warning("Fill layer %d has %d island(s) - this should not happen with per-layer routing!", layer_num, remaining_islands)

        layer_images.append(layer_img)

    # Dedicated structural fill layer (last)
    if has_structural:
        logging.info("Building layer %d/%d (structural bridge fill-in) (GPU)", total_layers, total_layers)
        layer_images.append(~structural_gpu)

    if failed_count > 0:
        warnings.append(f"{failed_count} island(s) could not get a bridge.")

    return layer_images, warnings, islands_mask, all_bridges_gpu | structural_gpu, structural_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multiple stencil layers to avoid holes while preserving the original silhouette when overlayed."
    )
    parser.add_argument("input", type=Path, help="Input image (PNG, BMP, JPEG, etc.)")
    parser.add_argument("--layers", type=int, default=2, help="Number of bridge fill layers (not counting the base layer and the structural fill layer)")
    parser.add_argument("--outdir", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument("--invert", action="store_true", help="Invert input if white is foreground")
    parser.add_argument("--threshold", type=int, default=128, help="Threshold for binarization (0-255)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration via CuPy")
    parser.add_argument("--no-smooth", action="store_true", help="Disable curved bridges")
    parser.add_argument("--smooth-iterations", type=int, default=0, help="Chaikin smoothing iterations")
    parser.add_argument("--bridge-width", type=int, default=1, help="Bridge width in pixels")
    parser.add_argument("--min-island-size", type=int, default=1, help="Minimum island size in pixels (smaller islands ignored)")
    parser.add_argument("--erode-passes", type=int, default=0, help="Erosion passes for island detection (higher = catches more diagonal/thin wall islands)")
    parser.add_argument("--bridges-per-island", type=int, default=1, help="Number of bridges per island (more = stronger stencil support)")
    parser.add_argument("--max-cutout-size", type=int, default=0, help="Maximum foreground region size in pixels; regions larger than this get structural bridges to limit open cutout area (0 = disabled)")
    parser.add_argument("--structural-bridges", action=argparse.BooleanOptionalAction, default=True, help="Enable structural bridge subdivision to limit large cutout regions (default: on)")
    parser.add_argument("--clip-structural", action="store_true", default=False, help="Clip dilated structural bridges back to their own region mask")
    parser.add_argument("--log", nargs='?', const="log.txt", default=None, metavar="FILE", help="Save console output to file (default: log.txt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Treat 0 (or negative) as "disabled" — same as omitting the flag
    if not args.structural_bridges:
        args.max_cutout_size = None
    elif args.max_cutout_size is not None and args.max_cutout_size <= 0:
        args.max_cutout_size = None

    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_format = "[%(levelname)s] %(message)s"
    handlers: list = [logging.StreamHandler()]
    
    if args.log:
        # Ensure output directory exists for log file
        args.outdir.mkdir(parents=True, exist_ok=True)
        log_path = args.outdir / args.log
        handlers.append(logging.FileHandler(log_path, mode='w'))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
    )
    if args.layers < 1:
        raise SystemExit("--layers must be >= 1")

    logging.info("Loading input image: %s", args.input)
    foreground = load_binary_image(args.input, invert=args.invert, threshold=args.threshold)
    logging.info("Generating %d layer(s)...", args.layers)

    # Load original image now so we can use it in the live struct_bridges progress callback.
    original_img = Image.open(args.input).convert("RGBA")
    original_rgba = np.array(original_img)

    smooth = not args.no_smooth and args.smooth_iterations > 0
    if args.gpu:
        try:
            import cupy as cp
        except Exception as exc:
            raise SystemExit(
                "GPU mode requires CuPy. Install with: pip install cupy-cuda12x (or cupy-cuda11x)."
            ) from exc

        foreground_gpu = cp.asarray(foreground)
        layers, warnings, islands_mask, bridges_mask, structural_mask = generate_layers_gpu(
            foreground_gpu,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
            min_island_size=args.min_island_size,
            erode_passes=args.erode_passes,
            bridges_per_island=args.bridges_per_island,
            max_cutout_size=args.max_cutout_size,
            clip_to_region=args.clip_structural,
        )
        layers = [cp.asnumpy(layer) for layer in layers]
        islands_mask = cp.asnumpy(islands_mask)
        bridges_mask = cp.asnumpy(bridges_mask)
        structural_mask = cp.asnumpy(structural_mask)
    else:
        layers, warnings, islands_mask, bridges_mask, structural_mask = generate_layers(
            foreground,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
            min_island_size=args.min_island_size,
            erode_passes=args.erode_passes,
            bridges_per_island=args.bridges_per_island,
            max_cutout_size=args.max_cutout_size,
            clip_to_region=args.clip_structural,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    # layers[0]  = base (full design + all bridges)
    # layers[1..-2] = bridge fill-in (round-robin, may be empty)
    # layers[-1] = structural fill (only when structural bridges were inserted)
    has_structural_layer = bool(structural_mask.any())
    save_binary_image(args.outdir / "layer_0_base.png", layers[0])
    logging.info("Wrote layer_0_base.png")
    if has_structural_layer and len(layers) > 1:
        save_binary_image(args.outdir / "layer_1_structural.png", layers[-1])
        logging.info("Wrote layer_1_structural.png")
        bridge_layers = layers[1:-1]
    else:
        bridge_layers = layers[1:]
    for i, layer in enumerate(bridge_layers, start=2):
        out_path = args.outdir / f"layer_{i}_bridges.png"
        save_binary_image(out_path, layer)
        logging.info("Wrote %s", out_path)

    islands_path = args.outdir / "islands.png"
    islands_rgba = blend_overlay(original_rgba, islands_mask, (0, 255, 0))  # Green for islands
    islands_img = Image.fromarray(islands_rgba, mode="RGBA")
    islands_img.save(islands_path)
    logging.info("Wrote %s (green = detected islands overlayed on original)", islands_path)

    # Save combined image (what you get when all layers are painted)
    # Holes in ANY layer = paint passes = black in final result
    # Combined = intersection of all layers (only stencil material where ALL layers have it)
    combined = layers[0].copy()
    for layer in layers[1:]:
        combined = combined & layer  # Only keep where ALL layers block paint
    # Invert for display: holes (paint through) become black, stencil material becomes white
    combined_result = ~combined  # Now True = painted (black), False = unpainted (white)
    combined_path = args.outdir / "combined.png"
    save_binary_image(combined_path, ~combined_result)  # Save as stencil format
    logging.info("Wrote %s (combined result of all layers)", combined_path)
    
    # Save diff image (difference between combined result and original) overlayed on original
    diff = foreground != combined_result
    diff_count = int(np.sum(diff))
    total_pixels = foreground.shape[0] * foreground.shape[1]
    diff_percent = (diff_count / total_pixels) * 100
    logging.info("Difference: %d pixels (%.4f%% of %d total)", diff_count, diff_percent, total_pixels)
    
    diff_path = args.outdir / "diff.png"
    # Overlay semi-transparent red pixels on original where there are differences
    diff_rgba = blend_overlay(original_rgba, diff, (255, 0, 0))  # Red for differences
    diff_img = Image.fromarray(diff_rgba, mode="RGBA")
    diff_img.save(diff_path)
    logging.info("Wrote %s (red = difference overlayed on original)", diff_path)

    # Save bridges image showing all bridges in blue overlayed on original
    bridges_path = args.outdir / "bridges.png"
    bridges_rgba = blend_overlay(original_rgba, bridges_mask, (0, 0, 255))  # Blue for bridges
    bridges_img = Image.fromarray(bridges_rgba, mode="RGBA")
    bridges_img.save(bridges_path)
    logging.info("Wrote %s (blue = bridges overlayed on original)", bridges_path)

    # Regular-only bridges (exclude structural) for pre-structural region view
    regular_bridges_mask = bridges_mask & ~structural_mask

    # cutouts.png: each foreground region AFTER regular bridges but BEFORE structural bridges
    from scipy import ndimage as _ndi_cutouts
    pre_structural_fg = foreground & ~regular_bridges_mask
    cut_labels, cut_count = _ndi_cutouts.label(pre_structural_fg, structure=np.ones((3, 3), dtype=bool))
    rng_cut = np.random.default_rng(42)
    _cut_colors = rng_cut.integers(60, 230, size=(cut_count + 1, 3)).astype(np.float32)
    _cut_colors[0] = 0
    cutouts_rgba = blend_overlay_labels(original_rgba, cut_labels, _cut_colors, alpha=0.65)
    cutouts_path = args.outdir / "cutouts.png"
    Image.fromarray(cutouts_rgba, mode="RGBA").save(cutouts_path)
    logging.info("Wrote %s (%d region(s) before structural bridges, each in a random colour)", cutouts_path, cut_count)

    # structural.png: oversized regions (before structural bridges) in random colours + structural bridges in pink
    structural_path = args.outdir / "structural.png"
    structural_rgba = original_rgba.copy()
    if args.structural_bridges and args.max_cutout_size is not None:
        # Compute per-label sizes in one vectorized pass
        label_sizes = np.bincount(cut_labels.ravel(), minlength=cut_count + 1)
        oversized_lbls = np.where(label_sizes > args.max_cutout_size)[0]
        oversized_lbls = oversized_lbls[oversized_lbls > 0]  # exclude background
        oversized_count = len(oversized_lbls)
        if oversized_count > 0:
            rng_str = np.random.default_rng(73)
            str_colors = rng_str.integers(60, 230, size=(cut_count + 1, 3)).astype(np.float32)
            str_colors[0] = 0
            # Zero out non-oversized labels so only oversized regions are painted
            keep = np.zeros(cut_count + 1, dtype=bool)
            keep[oversized_lbls] = True
            str_label_map = np.where(keep[cut_labels], cut_labels, 0)
            structural_rgba = blend_overlay_labels(original_rgba, str_label_map, str_colors, alpha=0.55)
        if structural_mask.any():
            structural_rgba = blend_overlay(structural_rgba, structural_mask, (255, 105, 180), alpha=0.85)
        logging.info("Wrote %s (%d oversized region(s) + structural bridges in pink)", structural_path, oversized_count)
    else:
        logging.info("Wrote %s (structural bridges disabled or no max-cutout-size set)", structural_path)
    Image.fromarray(structural_rgba, mode="RGBA").save(structural_path)

    if args.structural_bridges and args.max_cutout_size is not None:
        sb_path = args.outdir / "struct_bridges.png"
        if structural_mask.any():
            sb_rgba = blend_overlay(original_rgba, structural_mask, (255, 105, 180), alpha=0.85)
            Image.fromarray(sb_rgba, mode="RGBA").save(sb_path)
            logging.info("Wrote %s (structural bridges in pink)", sb_path)
        else:
            logging.info("No structural bridges were inserted (struct_bridges.png not created)")

    if warnings:
        logging.warning("Warnings:")
        for w in warnings:
            logging.warning("- %s", w)

    logging.info("Done. Wrote %d layer(s) to %s", len(layers), args.outdir)


if __name__ == "__main__":
    main()
