"""Iris Segmentation Pipeline — Streamlit UI.

Single-page application:
    1. Upload a raw eye image.
    2. Run TinyUNet segmentation → raw binary mask.
    3. Run geometric refinement → smooth overlay on original RGB.
    4. Display Original | Raw Mask | Refined Overlay side-by-side.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.eye_feature_pipeline.tinyunet import load_tinyunet, segment_iris
from src.eye_feature_pipeline.geometric_refinement import (
    refine_iris_segmentation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _workspace_root() -> Path:
    return Path(__file__).resolve().parent


def _load_model():
    """Cache the TinyUNet model in session state."""
    if "unet_model" not in st.session_state:
        ws = _workspace_root()
        ckpt = ws / "outputs" / "models" / "tinyunet.pth"
        if not ckpt.exists():
            st.error(f"TinyUNet checkpoint not found at {ckpt}. Train first.")
            st.stop()
        st.session_state["unet_model"] = load_tinyunet(ckpt)
    return st.session_state["unet_model"]


def _colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Colorize a binary mask: 0 → black, 1 → orange."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[mask == 1] = [255, 170, 0]
    return rgb


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Iris Segmentation Pipeline",
        page_icon="🔬",
        layout="wide",
    )

    st.title("🔬 Iris Segmentation Pipeline")
    st.caption(
        "TinyUNet binary mask → Geometric refinement → "
        "Smoothly refined overlay on the original eye image"
    )

    # ── Sidebar controls ───────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Overlay Settings")
        overlay_alpha = st.slider(
            "Overlay opacity",
            min_value=0.10, max_value=0.90, value=0.45, step=0.05,
            key="overlay_alpha",
            help="Transparency of the iris highlight overlay.",
        )
        overlay_color_hex = st.color_picker(
            "Overlay color", "#FFAA00", key="overlay_color",
        )
        oc = overlay_color_hex.lstrip("#")
        overlay_color = tuple(int(oc[i:i + 2], 16) for i in (0, 2, 4))

    # ── File uploader ──────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload a raw eye image",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        key="eye_upload",
    )

    if uploaded is None:
        st.info("👆 Upload an eye image to begin.")
        return

    # Show the uploaded image immediately
    pil_preview = Image.open(uploaded).convert("RGB")
    st.image(pil_preview, caption="Uploaded Image", use_container_width=True)

    # ── Run button ─────────────────────────────────────────────────────
    if not st.button("🚀 Run Segmentation", type="primary", key="run_btn"):
        return

    # ── Load images ────────────────────────────────────────────────────
    model = _load_model()

    # Re-open the uploaded file (the seek pointer may have moved)
    uploaded.seek(0)
    pil_img = Image.open(uploaded)
    pil_rgb  = pil_img.convert("RGB")
    pil_gray = pil_img.convert("L")

    original_rgb = np.asarray(pil_rgb, dtype=np.uint8)
    gray01       = np.asarray(pil_gray, dtype=np.float32) / 255.0

    # ── Phase 1: TinyUNet inference ────────────────────────────────────
    with st.spinner("Running TinyUNet segmentation..."):
        raw_mask = segment_iris(gray01, model)

    iris_pixels = int(np.sum(raw_mask > 0))
    if iris_pixels < 20:
        st.warning(
            f"⚠️ Very few iris pixels detected ({iris_pixels}). "
            "The mask may be empty — try a different image."
        )

    # ── Phase 2: Geometric refinement ──────────────────────────────────
    refinement_ok = True
    with st.spinner("Running geometric refinement..."):
        try:
            result = refine_iris_segmentation(
                gray01=gray01,
                raw_mask=raw_mask,
                original_rgb=original_rgb,
                overlay_color=overlay_color,
                overlay_alpha=overlay_alpha,
            )
        except ValueError as e:
            refinement_ok = False
            st.warning(f"⚠️ Geometric refinement failed: {e}")
            # Build a simple fallback overlay
            overlay_img = original_rgb.copy().astype(np.float32)
            tint = np.zeros_like(original_rgb, dtype=np.float32)
            tint[raw_mask == 1] = list(overlay_color)
            mask_3ch = np.stack([raw_mask] * 3, axis=-1).astype(np.float32)
            blended = (overlay_img * (1 - overlay_alpha * mask_3ch) +
                       tint * overlay_alpha * mask_3ch)
            fallback_overlay = np.clip(blended, 0, 255).astype(np.uint8)

    # ── Display: three-column comparison ───────────────────────────────
    st.divider()
    st.markdown("### Pipeline Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(pil_rgb, caption="① Original Eye Image",
                 use_container_width=True)

    with col2:
        mask_colored = _colorize_mask(raw_mask)
        st.image(mask_colored, caption="② Raw TinyUNet Mask",
                 use_container_width=True)

    with col3:
        if refinement_ok:
            st.image(result.overlay_rgb,
                     caption="③ Geometrically Refined Overlay",
                     use_container_width=True)
        else:
            st.image(fallback_overlay,
                     caption="③ Overlay (fallback — refinement failed)",
                     use_container_width=True)

    # ── Diagnostics (only if refinement succeeded) ─────────────────────
    if refinement_ok:
        st.divider()
        st.markdown("### 📊 Circle Fitting Diagnostics")

        d1, d2, d3, d4 = st.columns(4)

        with d1:
            st.metric(
                "Outer Circle (Limbic)",
                f"r = {result.outer_circle.radius}px",
            )
            st.caption(
                f"Center: ({result.outer_circle.center[0]}, "
                f"{result.outer_circle.center[1]})"
            )

        with d2:
            st.metric(
                "Inner Circle (Pupillary)",
                f"r = {result.inner_circle.radius}px",
            )
            st.caption(
                f"Center: ({result.inner_circle.center[0]}, "
                f"{result.inner_circle.center[1]})"
            )

        with d3:
            st.metric(
                "Outer Fit Error",
                f"{result.fitting_error_outer:.2f}px",
            )

        with d4:
            st.metric(
                "Inner Fit Error",
                f"{result.fitting_error_inner:.2f}px",
            )

        # Fitted circle visualization
        st.markdown("### 🔵 Fitted Circle Boundaries")
        circle_vis = original_rgb.copy()
        cv2.circle(circle_vis,
                   result.outer_circle.center,
                   result.outer_circle.radius,
                   (0, 255, 0), 2, cv2.LINE_AA)
        cv2.circle(circle_vis,
                   result.inner_circle.center,
                   result.inner_circle.radius,
                   (255, 0, 0), 2, cv2.LINE_AA)
        cv2.drawMarker(circle_vis, result.outer_circle.center,
                       (0, 255, 0), cv2.MARKER_CROSS, 10, 1)
        cv2.drawMarker(circle_vis, result.inner_circle.center,
                       (255, 0, 0), cv2.MARKER_CROSS, 10, 1)

        st.image(circle_vis,
                 caption="Fitted circles: 🟢 Limbic (outer)  🔴 Pupillary (inner)",
                 use_container_width=True)

    # ── Download buttons ───────────────────────────────────────────────
    st.divider()
    st.markdown("### 📥 Downloads")

    dl1, dl2 = st.columns(2)

    with dl1:
        # Raw mask download
        mask_buf = BytesIO()
        Image.fromarray((raw_mask * 255).astype(np.uint8), mode="L").save(
            mask_buf, format="PNG")
        st.download_button(
            label="📥 Download Raw Mask",
            data=mask_buf.getvalue(),
            file_name=f"mask_{uploaded.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            key="dl_mask",
        )

    with dl2:
        # Overlay download
        overlay_buf = BytesIO()
        if refinement_ok:
            Image.fromarray(result.overlay_rgb).save(overlay_buf, format="PNG")
        else:
            Image.fromarray(fallback_overlay).save(overlay_buf, format="PNG")
        st.download_button(
            label="📥 Download Refined Overlay",
            data=overlay_buf.getvalue(),
            file_name=f"overlay_{uploaded.name.rsplit('.', 1)[0]}.png",
            mime="image/png",
            key="dl_overlay",
        )


if __name__ == "__main__":
    main()
