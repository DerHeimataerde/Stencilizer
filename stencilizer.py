import argparse
import logging
from collections import deque
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from PIL import Image

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
    height, width = mask.shape
    out = mask.copy()
    
    # Build circular kernel offsets
    offsets = []
    for dr in range(-radius + 1, radius):
        for dc in range(-radius + 1, radius):
            if dr * dr + dc * dc < radius * radius:
                offsets.append((dr, dc))
    
    for dr, dc in offsets:
        if dr == 0 and dc == 0:
            continue
        r0 = max(0, dr)
        r1 = min(height, height + dr)
        c0 = max(0, dc)
        c1 = min(width, width + dc)
        out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def blend_overlay(base: np.ndarray, mask: np.ndarray, color: tuple, alpha: float = 0.75) -> np.ndarray:
    """Blend a semi-transparent color onto base image where mask is True."""
    result = base.copy().astype(np.float32)
    overlay_color = np.array(color[:3], dtype=np.float32)
    result[mask, :3] = result[mask, :3] * (1 - alpha) + overlay_color * alpha
    result[mask, 3] = 255  # Ensure full opacity in output
    return result.astype(np.uint8)


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
    for island_index, island in enumerate(islands):
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
    
    # Assign bridges to fill-in layers (layers 2..N)
    # Layer 1 = main design with all bridges
    # Layers 2..N each fill in a portion of the bridges
    fill_layers = layers - 1 if layers > 1 else 1
    bridge_assignments: List[List[int]] = [[] for _ in range(fill_layers)]
    for bridge_index in range(len(bridge_masks)):
        fill_layer = bridge_index % fill_layers
        bridge_assignments[fill_layer].append(bridge_index)
    
    layer_images: List[np.ndarray] = []
    
    # Layer 1: Full design + all islands + ALL bridges
    logging.info("Building layer 1/%d (main design with bridges)", layers)
    layer1 = ~foreground.copy()  # Design as holes (False = paint through)
    # Islands are already stencil material (True) from ~foreground
    # Add all bridges as stencil material
    for bridge_mask in bridge_masks:
        layer1 = layer1 | bridge_mask
    layer_images.append(layer1)
    
    # Layers 2..N: Only bridge areas as holes (to fill in bridge gaps)
    for fill_idx in range(fill_layers):
        if layers == 1:
            break
        layer_num = fill_idx + 2
        logging.info("Building layer %d/%d (bridge fill-in)", layer_num, layers)
        
        # Start with all stencil material (True = blocks paint)
        layer_img = np.ones((height, width), dtype=bool)
        
        # Only the assigned bridges become holes (False = paint through)
        for bridge_index in bridge_assignments[fill_idx]:
            layer_img = layer_img & ~bridge_masks[bridge_index]
        
        layer_images.append(layer_img)

    if failed_count > 0:
        warnings.append(f"{failed_count} island(s) could not get a bridge.")

    # Combine all bridge masks into single mask for visualization
    all_bridges_mask = np.zeros((height, width), dtype=bool)
    for bridge_mask in bridge_masks:
        all_bridges_mask = all_bridges_mask | bridge_mask

    return layer_images, warnings, islands_mask, all_bridges_mask


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
    
    # Compute all bridge paths first
    import math
    bridge_masks: List = []
    for label_id in island_labels_list:
        island_mask = labels == label_id
        boundary_black = foreground & cndi.binary_dilation(island_mask, structure=structure_8conn)
        
        # Fallback: use island pixels directly if no adjacent foreground
        has_boundary = bool(cp.any(boundary_black))
        if not has_boundary:
            logging.debug("Island %d (GPU) has no adjacent foreground pixels", label_id)
            boundary_black = island_mask
        
        # Get island pixels for centroid calculation
        island_coords = cp.argwhere(island_mask)
        island_coords_cpu = cp.asnumpy(island_coords)
        
        # Get boundary pixels  
        boundary_coords = cp.argwhere(boundary_black)
        boundary_coords_cpu = cp.asnumpy(boundary_coords)
        
        island_bridge_count = 0
        
        if bridges_per_island > 1 and len(boundary_coords_cpu) >= bridges_per_island:
            # Compute centroid
            centroid_r = float(island_coords_cpu[:, 0].mean())
            centroid_c = float(island_coords_cpu[:, 1].mean())
            
            # Compute angle for each boundary pixel
            boundary_with_angle = []
            for coord in boundary_coords_cpu:
                r, c = int(coord[0]), int(coord[1])
                angle = math.atan2(r - centroid_r, c - centroid_c)
                boundary_with_angle.append((angle, (r, c)))
            boundary_with_angle.sort(key=lambda x: x[0])
            
            # Select boundary pixels at evenly-spaced angles for maximum spread
            target_angles = [(-math.pi + (2 * math.pi * i / bridges_per_island)) for i in range(bridges_per_island)]
            
            # For each target angle, find the closest boundary pixel
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
            
            used_targets_cpu: Set[Coord] = set()
            
            for coord in selected_boundaries:
                # Create boundary mask for this single pixel
                group_boundary_mask = cp.zeros_like(foreground, dtype=bool)
                group_boundary_mask[coord[0], coord[1]] = True
                
                # Exclude used targets
                available_targets = all_targets.copy()
                for r, c in used_targets_cpu:
                    available_targets[r, c] = False
                if not cp.any(available_targets):
                    available_targets = all_targets
                
                path = _gpu_bfs_path(group_boundary_mask, available_targets, foreground, use_8conn=False)
                if path is None:
                    path = _gpu_bfs_path(group_boundary_mask, available_targets, foreground, use_8conn=True)
                if path is None:
                    allowed_any = cp.ones_like(foreground, dtype=bool)
                    path = _gpu_bfs_path(group_boundary_mask, available_targets, allowed_any, use_8conn=True)
                
                if path is not None:
                    island_bridge_count += 1
                    if path:
                        used_targets_cpu.add(path[-1])
                    
                    path_mask_cpu = rasterize_path(
                        path,
                        foreground.shape,
                        smooth=smooth,
                        smooth_iterations=smooth_iterations,
                        bridge_width=bridge_width,
                    )
                    bridge_masks.append(cp.asarray(path_mask_cpu))
            
            if island_bridge_count > 0:
                logging.debug("Island %d (size=%d, GPU) bridged with %d paths", 
                             label_id, len(island_coords_cpu), island_bridge_count)
            else:
                failed_count += 1
                logging.debug("Island %d (size=%d, GPU) could not find any bridge path", 
                             label_id, len(island_coords_cpu))
                bridge_masks.append(cp.zeros_like(foreground, dtype=bool))
        else:
            # Single bridge (original behavior)
            path = _gpu_bfs_path(boundary_black, all_targets, foreground, use_8conn=False)
            if path is None:
                path = _gpu_bfs_path(boundary_black, all_targets, foreground, use_8conn=True)
            if path is None:
                allowed_any = cp.ones_like(foreground, dtype=bool)
                path = _gpu_bfs_path(boundary_black, all_targets, allowed_any, use_8conn=True)
            
            if path is None:
                failed_count += 1
                logging.debug("Island %d (size=%d, GPU) could not find bridge path", 
                             label_id, len(island_coords_cpu))
                bridge_masks.append(cp.zeros_like(foreground, dtype=bool))
                continue
            
            logging.debug("Island %d (size=%d, GPU) bridged with path of length %d", 
                         label_id, len(island_coords_cpu), len(path))
            path_mask_cpu = rasterize_path(
                path,
                foreground.shape,
                smooth=smooth,
                smooth_iterations=smooth_iterations,
                bridge_width=bridge_width,
            )
            bridge_masks.append(cp.asarray(path_mask_cpu))
    
    # Assign bridges to fill-in layers (layers 2..N)
    fill_layers = layers - 1 if layers > 1 else 1
    bridge_assignments: List[List[int]] = [[] for _ in range(fill_layers)]
    for bridge_index in range(len(bridge_masks)):
        fill_layer = bridge_index % fill_layers
        bridge_assignments[fill_layer].append(bridge_index)
    
    layer_images: List = []
    
    # Layer 1: Full design + all islands + ALL bridges
    logging.info("Building layer 1/%d (main design with bridges) (GPU)", layers)
    layer1 = ~foreground.copy()  # Design as holes (False = paint through)
    # Add all bridges as stencil material
    for bridge_mask in bridge_masks:
        layer1 = layer1 | bridge_mask
    layer_images.append(layer1)
    
    # Layers 2..N: Only bridge areas as holes
    for fill_idx in range(fill_layers):
        if layers == 1:
            break
        layer_num = fill_idx + 2
        logging.info("Building layer %d/%d (bridge fill-in) (GPU)", layer_num, layers)
        
        # Start with all stencil material (True = blocks paint)
        layer_img = cp.ones(foreground.shape, dtype=bool)
        
        # Only the assigned bridges become holes
        for bridge_index in bridge_assignments[fill_idx]:
            layer_img = layer_img & ~bridge_masks[bridge_index]
        
        layer_images.append(layer_img)

    if failed_count > 0:
        warnings.append(f"{failed_count} island(s) could not get a bridge.")

    # Combine all bridge masks into single mask for visualization
    all_bridges_mask = cp.zeros(foreground.shape, dtype=bool)
    for bridge_mask in bridge_masks:
        all_bridges_mask = all_bridges_mask | bridge_mask

    return layer_images, warnings, islands_mask, all_bridges_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multiple stencil layers to avoid holes while preserving the original silhouette when overlayed."
    )
    parser.add_argument("input", type=Path, help="Input image (PNG, BMP, JPEG, etc.)")
    parser.add_argument("--layers", type=int, default=2, help="Number of stencil layers to generate")
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
    parser.add_argument("--log", nargs='?', const="log.txt", default=None, metavar="FILE", help="Save console output to file (default: log.txt)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
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

    smooth = not args.no_smooth and args.smooth_iterations > 0
    if args.gpu:
        try:
            import cupy as cp
        except Exception as exc:
            raise SystemExit(
                "GPU mode requires CuPy. Install with: pip install cupy-cuda12x (or cupy-cuda11x)."
            ) from exc

        foreground_gpu = cp.asarray(foreground)
        layers, warnings, islands_mask, bridges_mask = generate_layers_gpu(
            foreground_gpu,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
            min_island_size=args.min_island_size,
            erode_passes=args.erode_passes,
            bridges_per_island=args.bridges_per_island,
        )
        layers = [cp.asnumpy(layer) for layer in layers]
        islands_mask = cp.asnumpy(islands_mask)
        bridges_mask = cp.asnumpy(bridges_mask)
    else:
        layers, warnings, islands_mask, bridges_mask = generate_layers(
            foreground,
            args.layers,
            smooth=smooth,
            smooth_iterations=args.smooth_iterations,
            bridge_width=args.bridge_width,
            min_island_size=args.min_island_size,
            erode_passes=args.erode_passes,
            bridges_per_island=args.bridges_per_island,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    for idx, layer in enumerate(layers, start=1):
        out_path = args.outdir / f"layer_{idx:02d}.png"
        save_binary_image(out_path, layer)
        logging.info("Wrote %s", out_path)

    # Load original image for overlays
    original_img = Image.open(args.input).convert("RGBA")
    original_rgba = np.array(original_img)

    # Save islands image showing detected islands overlayed on original
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

    if warnings:
        logging.warning("Warnings:")
        for w in warnings:
            logging.warning("- %s", w)

    logging.info("Done. Wrote %d layer(s) to %s", len(layers), args.outdir)


if __name__ == "__main__":
    main()
