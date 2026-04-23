"""Geometric Refinement Pipeline for Iris Segmentation.

Takes a raw TinyML binary mask (pixelated, staircased) and produces a
geometrically perfect circular annulus overlay on the original eye image.

CRITICAL CONSTRAINT: The outer fitted circle is strictly bounded by the
footprint of the original raw mask. It must NEVER expand beyond the mask
into eyelashes, eyelids, or sclera.

Pipeline:
    1. Smooth the raw mask (bilateral + morphological operations)
    2. Extract inner and outer boundary points from the mask itself
    3. Fit perfect circles with strict bounding on the outer radius
    4. Generate a clean annular mask from the bounded fitted circles
    5. Compose a translucent overlay on the original image
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FittedCircle:
    """A fitted circle with center (cx, cy) and radius r."""
    cx: float
    cy: float
    r: float

    @property
    def center(self) -> Tuple[int, int]:
        return (int(round(self.cx)), int(round(self.cy)))

    @property
    def radius(self) -> int:
        return int(round(self.r))


@dataclass
class RefinementResult:
    """Full result of geometric refinement."""
    inner_circle: FittedCircle        # Pupillary boundary
    outer_circle: FittedCircle        # Limbic boundary
    annular_mask: np.ndarray          # Clean annular mask (H, W), uint8 0-255
    overlay_image: np.ndarray         # Final RGBA composite (H, W, 4), uint8
    overlay_rgb: np.ndarray           # Final RGB composite (H, W, 3), uint8
    fitting_error_inner: float        # RMS error for inner circle fit (px)
    fitting_error_outer: float        # RMS error for outer circle fit (px)


# ---------------------------------------------------------------------------
# Circle Fitting (Algebraic + Geometric refinement)
# ---------------------------------------------------------------------------

def _algebraic_circle_fit(points: np.ndarray) -> Tuple[float, float, float]:
    """Kåsa algebraic circle fit (fast, closed-form).

    Args:
        points: (N, 2) array of (x, y) boundary points.

    Returns:
        (cx, cy, r) — center and radius.
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    n = len(x)

    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(n)])
    b = x**2 + y**2

    result, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, D = result
    r = np.sqrt(max(D + cx**2 + cy**2, 0.0))

    return float(cx), float(cy), float(r)


def _geometric_circle_fit(points: np.ndarray, cx0: float, cy0: float, r0: float,
                           max_iter: int = 100, tol: float = 1e-6
                           ) -> Tuple[float, float, float, float]:
    """Gauss–Newton geometric circle fit (minimizes geometric distance).

    Returns:
        (cx, cy, r, rms_error)
    """
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)

    cx, cy, r = cx0, cy0, r0

    for _ in range(max_iter):
        dx = x - cx
        dy = y - cy
        d = np.sqrt(dx**2 + dy**2)
        d = np.maximum(d, 1e-10)

        residuals = d - r
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)])

        JtJ = J.T @ J
        Jtr = J.T @ residuals

        try:
            delta = np.linalg.solve(JtJ, -Jtr)
        except np.linalg.LinAlgError:
            break

        cx += delta[0]
        cy += delta[1]
        r += delta[2]

        if np.linalg.norm(delta) < tol:
            break

    dx = x - cx
    dy = y - cy
    d = np.sqrt(dx**2 + dy**2)
    rms = float(np.sqrt(np.mean((d - r)**2)))

    return float(cx), float(cy), float(abs(r)), rms


def fit_circle(points: np.ndarray) -> Tuple[FittedCircle, float]:
    """Fit a circle via algebraic init + geometric refinement.

    Args:
        points: (N, 2) array of (x, y) boundary points.

    Returns:
        (FittedCircle, rms_error)
    """
    if len(points) < 3:
        raise ValueError(f"Need ≥3 points for circle fitting, got {len(points)}")

    cx0, cy0, r0 = _algebraic_circle_fit(points)
    cx, cy, r, rms = _geometric_circle_fit(points, cx0, cy0, r0)

    return FittedCircle(cx=cx, cy=cy, r=r), rms


# ---------------------------------------------------------------------------
# Strict Outer‑Radius Bounding
# ---------------------------------------------------------------------------

def _compute_bounded_outer_radius(
    cx: float, cy: float, outer_pts: np.ndarray,
    raw_mask: np.ndarray,
    percentile: float = 5.0,
) -> float:
    """Compute the maximum outer radius that stays within the raw mask.

    Strategy: from the fitted center, compute the distance to every outer
    boundary point.  The fitted circle's radius must be ≤ a conservative
    lower percentile of those distances so the circle is *inscribed* inside
    the mask footprint instead of circumscribing it.

    A low percentile (5th) is used to robustly ignore any boundary
    irregularities while ensuring the circle never protrudes.

    An additional check: for 360 radial directions, ray‑march from the
    center outward and record where the mask ends.  The outer radius is
    clamped to the minimum of those radial extents.

    Args:
        cx, cy: Fitted circle center.
        outer_pts: (N, 2) outer boundary points.
        raw_mask: (H, W) original binary mask, uint8 {0, 1}.
        percentile: Distance percentile to use (lower = more conservative).

    Returns:
        Bounded radius (px).
    """
    # --- Method 1: percentile of boundary‑point distances ---
    dx = outer_pts[:, 0] - cx
    dy = outer_pts[:, 1] - cy
    dists = np.sqrt(dx**2 + dy**2)
    r_percentile = float(np.percentile(dists, percentile))

    # --- Method 2: radial ray‑march on the raw mask ---
    h, w = raw_mask.shape
    n_rays = 360
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)
    radial_extents = []

    for theta in angles:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        # March outward from center
        max_r = max(h, w)
        extent = 0.0
        for step in range(1, int(max_r)):
            px = int(round(cx + cos_t * step))
            py = int(round(cy + sin_t * step))
            if 0 <= px < w and 0 <= py < h:
                if raw_mask[py, px] > 0:
                    extent = float(step)
                else:
                    break  # hit background — stop
            else:
                break  # out of image
        if extent > 0:
            radial_extents.append(extent)

    if radial_extents:
        r_raymin = float(np.min(radial_extents))
    else:
        r_raymin = r_percentile

    # Take the tighter (smaller) of the two bounds
    bounded_r = min(r_percentile, r_raymin)

    return max(bounded_r, 3.0)  # floor at 3px to avoid degenerate circles


# ---------------------------------------------------------------------------
# Mask Smoothing & Boundary Extraction
# ---------------------------------------------------------------------------

def _smooth_mask(mask: np.ndarray, upscale: int = 4) -> np.ndarray:
    """Smooth a binary mask using upscaling + morphological ops + bilateral filter.

    Args:
        mask: (H, W) binary mask, values in {0, 1}, dtype uint8.
        upscale: Upscale factor for supersampled smoothing.

    Returns:
        Smoothed binary mask at original resolution, (H, W), uint8 {0,1}.
    """
    h, w = mask.shape

    mask_up = cv2.resize(mask * 255, (w * upscale, h * upscale),
                         interpolation=cv2.INTER_LINEAR)

    kernel_size = max(5, upscale * 3) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_up = cv2.morphologyEx(mask_up, cv2.MORPH_OPEN, kernel, iterations=2)

    mask_up = cv2.bilateralFilter(mask_up, d=9, sigmaColor=75, sigmaSpace=75)

    blur_k = max(3, upscale * 2 + 1) | 1
    mask_up = cv2.GaussianBlur(mask_up, (blur_k, blur_k), 0)

    _, mask_up = cv2.threshold(mask_up, 127, 255, cv2.THRESH_BINARY)

    smoothed = cv2.resize(mask_up, (w, h), interpolation=cv2.INTER_AREA)
    _, smoothed = cv2.threshold(smoothed, 127, 1, cv2.THRESH_BINARY)

    return smoothed.astype(np.uint8)


def _extract_boundary_points(mask: np.ndarray, is_inner: bool = False) -> np.ndarray:
    """Extract boundary points from a binary mask.

    Relies strictly on the mask contours — no separate pupil detection.

    For outer: the outermost contour of the mask.
    For inner: the largest hole (child contour) inside the outer contour.

    Args:
        mask: (H, W) binary mask, uint8, values in {0, 1}.
        is_inner: If True, extract inner (hole) boundary.

    Returns:
        (N, 2) array of (x, y) boundary points.
    """
    mask_255 = (mask * 255).astype(np.uint8)

    contours, hierarchy = cv2.findContours(
        mask_255, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return np.empty((0, 2), dtype=np.float32)

    if hierarchy is None:
        largest = max(contours, key=cv2.contourArea)
        return largest.reshape(-1, 2).astype(np.float32)

    hierarchy = hierarchy[0]  # (N, 4): [next, prev, child, parent]

    if is_inner:
        # Find largest outer contour, then its largest child (the hole)
        outer_idx = max(range(len(contours)),
                        key=lambda i: cv2.contourArea(contours[i]))
        child_idx = hierarchy[outer_idx][2]

        if child_idx < 0:
            return np.empty((0, 2), dtype=np.float32)

        # Walk siblings to find the largest child
        best_child = child_idx
        best_area = cv2.contourArea(contours[child_idx])
        nxt = hierarchy[child_idx][0]
        while nxt >= 0:
            area = cv2.contourArea(contours[nxt])
            if area > best_area:
                best_area = area
                best_child = nxt
            nxt = hierarchy[nxt][0]

        return contours[best_child].reshape(-1, 2).astype(np.float32)
    else:
        largest_idx = max(range(len(contours)),
                          key=lambda i: cv2.contourArea(contours[i]))
        return contours[largest_idx].reshape(-1, 2).astype(np.float32)


# ---------------------------------------------------------------------------
# Annular Mask Generation
# ---------------------------------------------------------------------------

def _generate_annular_mask(h: int, w: int,
                           inner: FittedCircle,
                           outer: FittedCircle) -> np.ndarray:
    """Generate a clean annular mask from two fitted circles.

    Rendered at 4× supersampling for antialiased edges.

    Returns:
        (H, W) uint8, values 0–255 for smooth alpha blending.
    """
    scale = 4
    h_up, w_up = h * scale, w * scale

    outer_mask = np.zeros((h_up, w_up), dtype=np.uint8)
    cv2.circle(outer_mask,
               (int(round(outer.cx * scale)), int(round(outer.cy * scale))),
               int(round(outer.r * scale)),
               255, thickness=-1, lineType=cv2.LINE_AA)

    inner_mask = np.zeros((h_up, w_up), dtype=np.uint8)
    cv2.circle(inner_mask,
               (int(round(inner.cx * scale)), int(round(inner.cy * scale))),
               int(round(inner.r * scale)),
               255, thickness=-1, lineType=cv2.LINE_AA)

    annulus_up = cv2.subtract(outer_mask, inner_mask)

    annulus = cv2.resize(annulus_up, (w, h), interpolation=cv2.INTER_AREA)

    return annulus


# ---------------------------------------------------------------------------
# Overlay Composition
# ---------------------------------------------------------------------------

def _compose_overlay(original_rgb: np.ndarray,
                     annular_mask: np.ndarray,
                     overlay_color: Tuple[int, int, int] = (255, 170, 0),
                     alpha: float = 0.45) -> Tuple[np.ndarray, np.ndarray]:
    """Compose a translucent color overlay on the original image.

    Args:
        original_rgb: (H, W, 3) RGB image, uint8.
        annular_mask: (H, W) mask with values 0–255.
        overlay_color: RGB tuple for the overlay color.
        alpha: Overlay opacity (0–1).

    Returns:
        (overlay_rgba, overlay_rgb)
    """
    h, w = original_rgb.shape[:2]

    mask_float = annular_mask.astype(np.float32) / 255.0

    color_layer = np.zeros_like(original_rgb, dtype=np.float32)
    for c in range(3):
        color_layer[:, :, c] = overlay_color[c]

    effective_alpha = mask_float * alpha

    orig_f = original_rgb.astype(np.float32)
    blended = (orig_f * (1.0 - effective_alpha[:, :, np.newaxis]) +
               color_layer * effective_alpha[:, :, np.newaxis])

    blended_rgb = np.clip(blended, 0, 255).astype(np.uint8)

    blended_rgba = np.dstack([blended_rgb,
                              np.full((h, w), 255, dtype=np.uint8)])

    return blended_rgba, blended_rgb


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def refine_iris_segmentation(
    gray01: np.ndarray,
    raw_mask: np.ndarray,
    original_rgb: np.ndarray,
    overlay_color: Tuple[int, int, int] = (255, 170, 0),
    overlay_alpha: float = 0.45,
) -> RefinementResult:
    """Full geometric refinement pipeline with strict outer bounding.

    The outer fitted circle is constrained so its radius never exceeds
    the radial extent of the original raw mask in any direction.  This
    ensures the overlay cannot bleed into eyelashes, eyelids, or sclera.

    Args:
        gray01: (H, W) grayscale image in [0, 1].
        raw_mask: (H, W) binary mask from TinyUNet, uint8, {0, 1}.
        original_rgb: (H, W, 3) original eye image in RGB, uint8.
        overlay_color: RGB tuple for overlay tint.
        overlay_alpha: Opacity of the overlay (0.0–1.0).

    Returns:
        RefinementResult with all outputs.
    """
    h, w = raw_mask.shape

    # --- Phase 2.3: Smooth the raw mask ---
    smoothed = _smooth_mask(raw_mask, upscale=4)

    # --- Phase 2.4: Extract boundary points (mask only, no pupil detect) ---
    outer_pts = _extract_boundary_points(smoothed, is_inner=False)
    inner_pts = _extract_boundary_points(smoothed, is_inner=True)

    if len(outer_pts) < 10:
        raise ValueError(
            "Cannot extract outer boundary — mask may be empty or too small."
        )

    # --- Phase 2.5: Circle Fitting with strict bounding ---

    # Fit unconstrained circles first (for center estimation)
    outer_circle_raw, rms_outer = fit_circle(outer_pts)

    # *** CRITICAL: bound the outer radius to the raw mask footprint ***
    bounded_r = _compute_bounded_outer_radius(
        outer_circle_raw.cx, outer_circle_raw.cy,
        outer_pts, raw_mask, percentile=5.0,
    )

    outer_circle = FittedCircle(
        cx=outer_circle_raw.cx,
        cy=outer_circle_raw.cy,
        r=bounded_r,
    )

    # Re‑compute RMS for the bounded circle (informational)
    dx = outer_pts[:, 0] - outer_circle.cx
    dy = outer_pts[:, 1] - outer_circle.cy
    d = np.sqrt(dx**2 + dy**2)
    rms_outer = float(np.sqrt(np.mean((d - outer_circle.r)**2)))

    # Inner circle: fit from the mask's inner void
    if len(inner_pts) >= 10:
        inner_circle, rms_inner = fit_circle(inner_pts)
    else:
        # Fallback: estimate inner circle as ~30% of outer, concentric
        inner_circle = FittedCircle(
            cx=outer_circle.cx, cy=outer_circle.cy,
            r=outer_circle.r * 0.30,
        )
        rms_inner = 0.0

    # --- Sanity corrections ---

    # Inner must be inside outer
    if inner_circle.r >= outer_circle.r * 0.90:
        inner_circle = FittedCircle(
            cx=outer_circle.cx, cy=outer_circle.cy,
            r=outer_circle.r * 0.33,
        )

    # Centers shouldn't be too far apart
    dist_centers = np.sqrt((inner_circle.cx - outer_circle.cx)**2 +
                           (inner_circle.cy - outer_circle.cy)**2)
    if dist_centers > outer_circle.r * 0.4:
        inner_circle = FittedCircle(
            cx=outer_circle.cx, cy=outer_circle.cy,
            r=inner_circle.r,
        )

    # Minimum annulus width
    if outer_circle.r - inner_circle.r < 5:
        inner_circle = FittedCircle(
            cx=inner_circle.cx, cy=inner_circle.cy,
            r=max(3.0, outer_circle.r * 0.30),
        )

    # --- Phase 2.6: Generate refined annular mask ---
    annular_mask = _generate_annular_mask(h, w, inner_circle, outer_circle)

    # Final safety: AND with the raw mask so no pixel outside the original
    # segmentation can ever be colored.  This handles irregular eyelid edges
    # that even the bounded circle might touch.
    raw_mask_255 = (raw_mask * 255).astype(np.uint8)
    annular_mask = cv2.min(annular_mask, raw_mask_255)

    # --- Phase 3: Compose overlay ---
    overlay_rgba, overlay_rgb = _compose_overlay(
        original_rgb, annular_mask,
        overlay_color=overlay_color,
        alpha=overlay_alpha,
    )

    return RefinementResult(
        inner_circle=inner_circle,
        outer_circle=outer_circle,
        annular_mask=annular_mask,
        overlay_image=overlay_rgba,
        overlay_rgb=overlay_rgb,
        fitting_error_inner=rms_inner,
        fitting_error_outer=rms_outer,
    )
