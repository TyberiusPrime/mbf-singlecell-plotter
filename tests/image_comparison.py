"""
Image comparison helpers for visual regression testing.

Usage:
    from image_comparison import assert_image_matches

    def test_something(tmp_path):
        fig = make_my_plot()
        assert_image_matches(fig, "test_something")

On first run (no reference image exists), the image is saved as the reference and
the test passes. On subsequent runs the rendered image is compared to the reference;
on mismatch both images plus a per-pixel diff are written to tests/failures/ and
the test fails with a descriptive message.

Set REGENERATE_REFS=1 in the environment to overwrite all reference images.
"""

import os
import io
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageChops, ImageEnhance

TESTS_DIR = Path(__file__).parent
REFERENCE_DIR = TESTS_DIR / "reference_images"
FAILURES_DIR = TESTS_DIR / "failures"
REGENERATE = os.environ.get("REGENERATE_REFS", "").strip() not in ("", "0")

# Pixel-level tolerance: fraction of pixels allowed to differ
DEFAULT_TOLERANCE = 0.001   # 0.1 %
# Per-pixel difference magnitude threshold (0-255) before a pixel counts as "different"
DEFAULT_PIXEL_THRESHOLD = 5


def _fig_to_array(fig, dpi=150) -> np.ndarray:
    """Render a matplotlib figure to an RGBA uint8 numpy array."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return np.array(img)


def _plotnine_to_array(p, dpi=100) -> np.ndarray:
    """Render a plotnine ggplot to an RGB uint8 numpy array."""
    buf = io.BytesIO()
    p.save(buf, format="png", dpi=dpi, verbose=False)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _load_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _build_diff_image(ref: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Returns an RGB uint8 array highlighting differences.
    Matching pixels are grey-washed; differing pixels are shown in red.
    """
    if ref.shape != actual.shape:
        # If shapes differ, just stack them side by side
        max_h = max(ref.shape[0], actual.shape[0])
        pad_ref = np.full((max_h, ref.shape[1], 3), 200, dtype=np.uint8)
        pad_act = np.full((max_h, actual.shape[1], 3), 200, dtype=np.uint8)
        pad_ref[: ref.shape[0], :, :] = ref
        pad_act[: actual.shape[0], :, :] = actual
        diff_raw = np.abs(pad_ref.astype(int) - pad_act.astype(int)).astype(np.uint8)
        return diff_raw

    diff_raw = np.abs(ref.astype(int) - actual.astype(int)).astype(np.uint8)
    # Amplify for visibility
    amplified = np.clip(diff_raw * 10, 0, 255).astype(np.uint8)
    # Build a red-channel-dominant diff
    diff_rgb = np.zeros_like(ref)
    diff_rgb[:, :, 0] = amplified.max(axis=2)  # R = max channel diff
    diff_rgb[:, :, 1] = 0
    diff_rgb[:, :, 2] = 0
    # Background: desaturated actual
    grey = (actual.astype(float).mean(axis=2, keepdims=True) * 0.4).astype(np.uint8)
    background = np.repeat(grey, 3, axis=2)
    # Overlay diff in red where there are differences
    has_diff = diff_raw.max(axis=2) > 5
    result = background.copy()
    result[has_diff] = diff_rgb[has_diff]
    result[~has_diff] = background[~has_diff]
    return result


def assert_image_matches(
    fig,
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    dpi: int = 150,
    close_fig: bool = True,
) -> None:
    """
    Assert that `fig` renders the same as the stored reference image `name`.

    Parameters
    ----------
    fig:
        A matplotlib Figure.  Will be closed after rendering unless close_fig=False.
    name:
        Stem for the reference PNG file (no extension, no path).
    tolerance:
        Maximum fraction of pixels that may differ.
    pixel_threshold:
        A pixel is considered different only if any channel differs by more than this.
    dpi:
        Render resolution.
    close_fig:
        Close the figure after rendering (default True).
    """
    actual = _fig_to_array(fig, dpi=dpi)

    if close_fig:
        plt.close(fig)

    ref_path = REFERENCE_DIR / f"{name}.png"
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    if REGENERATE or not ref_path.exists():
        _save_png(actual, ref_path)
        print(f"  [image] Reference saved: {ref_path}")
        return  # first run always passes

    ref = _load_png(ref_path)

    if ref.shape != actual.shape:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        diff = _build_diff_image(ref, actual)
        _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image shape mismatch for '{name}': "
            f"reference={ref.shape}, actual={actual.shape}\n"
            f"  reference: {ref_path}\n"
            f"  actual:    {FAILURES_DIR / (name + '_actual.png')}\n"
            f"  diff:      {FAILURES_DIR / (name + '_diff.png')}"
        )

    diff_mask = np.abs(ref.astype(int) - actual.astype(int)).max(axis=2) > pixel_threshold
    bad_fraction = diff_mask.mean()

    if bad_fraction > tolerance:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        import shutil
        shutil.copy(ref_path, FAILURES_DIR / f"{name}_reference.png")
        diff = _build_diff_image(ref, actual)
        _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image mismatch for '{name}': "
            f"{bad_fraction:.2%} pixels differ (tolerance {tolerance:.2%})\n"
            f"  reference: {FAILURES_DIR / (name + '_reference.png')}\n"
            f"  actual:    {FAILURES_DIR / (name + '_actual.png')}\n"
            f"  diff:      {FAILURES_DIR / (name + '_diff.png')}"
        )


def assert_array_matches(
    actual: np.ndarray,
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
) -> None:
    """Same as assert_image_matches but accepts a pre-rendered RGB uint8 array."""
    import shutil

    ref_path = REFERENCE_DIR / f"{name}.png"
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    if REGENERATE or not ref_path.exists():
        _save_png(actual, ref_path)
        print(f"  [image] Reference saved: {ref_path}")
        return

    ref = _load_png(ref_path)
    if ref.shape != actual.shape:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        _save_png(_build_diff_image(ref, actual), FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image shape mismatch for '{name}': "
            f"reference={ref.shape}, actual={actual.shape}"
        )

    diff_mask = np.abs(ref.astype(int) - actual.astype(int)).max(axis=2) > pixel_threshold
    bad_fraction = diff_mask.mean()
    if bad_fraction > tolerance:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        shutil.copy(ref_path, FAILURES_DIR / f"{name}_reference.png")
        _save_png(_build_diff_image(ref, actual), FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image mismatch for '{name}': "
            f"{bad_fraction:.2%} pixels differ (tolerance {tolerance:.2%})\n"
            f"  reference: {FAILURES_DIR / (name + '_reference.png')}\n"
            f"  actual:    {FAILURES_DIR / (name + '_actual.png')}\n"
            f"  diff:      {FAILURES_DIR / (name + '_diff.png')}"
        )


def assert_plotnine_matches(
    p,
    name: str,
    tolerance: float = DEFAULT_TOLERANCE,
    pixel_threshold: int = DEFAULT_PIXEL_THRESHOLD,
    dpi: int = 100,
    width: float = None,
    height: float = None,
) -> None:
    """Same as assert_image_matches but for a plotnine plot object."""
    import io
    from PIL import Image

    save_kwargs = dict(format="png", dpi=dpi, verbose=False)
    if width is not None:
        save_kwargs["width"] = width
    if height is not None:
        save_kwargs["height"] = height

    buf = io.BytesIO()
    p.save(buf, **save_kwargs)
    buf.seek(0)
    actual = np.array(Image.open(buf).convert("RGB"))

    ref_path = REFERENCE_DIR / f"{name}.png"
    FAILURES_DIR.mkdir(parents=True, exist_ok=True)

    if REGENERATE or not ref_path.exists():
        _save_png(actual, ref_path)
        print(f"  [image] Reference saved: {ref_path}")
        return

    ref = _load_png(ref_path)
    if ref.shape != actual.shape:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        diff = _build_diff_image(ref, actual)
        _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image shape mismatch for '{name}': "
            f"reference={ref.shape}, actual={actual.shape}"
        )

    import shutil
    diff_mask = np.abs(ref.astype(int) - actual.astype(int)).max(axis=2) > pixel_threshold
    bad_fraction = diff_mask.mean()
    if bad_fraction > tolerance:
        _save_png(actual, FAILURES_DIR / f"{name}_actual.png")
        shutil.copy(ref_path, FAILURES_DIR / f"{name}_reference.png")
        diff = _build_diff_image(ref, actual)
        _save_png(diff, FAILURES_DIR / f"{name}_diff.png")
        raise AssertionError(
            f"Image mismatch for '{name}': "
            f"{bad_fraction:.2%} pixels differ (tolerance {tolerance:.2%})\n"
            f"  reference: {FAILURES_DIR / (name + '_reference.png')}\n"
            f"  actual:    {FAILURES_DIR / (name + '_actual.png')}\n"
            f"  diff:      {FAILURES_DIR / (name + '_diff.png')}"
        )
