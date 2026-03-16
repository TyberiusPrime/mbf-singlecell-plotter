#!/usr/bin/env python3
"""
Review image test failures in the terminal.

Usage:
    python dev/review-image-changes.py [--failures-dir PATH] [--filter PATTERN]

Finds all sets of (reference, actual, diff) PNGs in tests/failures/ and
displays them side-by-side in the terminal using the Kitty graphics protocol.
Falls back to printing file paths + basic PIL stats if Kitty is not available.

Controls (Kitty mode):
    Press Enter / n  → next failure
    Press q          → quit
"""

import argparse
import os
import sys
import struct
import zlib
import base64
import termios
import tty
from pathlib import Path
from typing import Optional

TESTS_DIR = Path(__file__).parent.parent / "tests"
DEFAULT_FAILURES = TESTS_DIR / "failures"
DEFAULT_REFERENCE = TESTS_DIR / "reference_images"


# ---------------------------------------------------------------------------
# Kitty graphics protocol helpers
# ---------------------------------------------------------------------------

def _kitty_supported() -> bool:
    """Heuristic: check if TERM or TERM_PROGRAM suggests Kitty."""
    term = os.environ.get("TERM", "")
    term_program = os.environ.get("TERM_PROGRAM", "")
    return "kitty" in term.lower() or "kitty" in term_program.lower()


def _encode_image_kitty(path: Path, max_width: int = 400) -> None:
    """
    Display an image using the Kitty terminal graphics protocol (APC escape).
    Reads the PNG, sends it in base64-encoded chunks.
    """
    try:
        from PIL import Image
        import io

        img = Image.open(path).convert("RGB")
        # Scale down if too wide
        w, h = img.size
        if w > max_width:
            scale = max_width / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue())

        chunk_size = 4096
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            m_value = 0 if is_last else 1
            if i == 0:
                header = f"a=T,f=100,m={m_value}"
            else:
                header = f"m={m_value}"
            sys.stdout.buffer.write(
                b"\x1b_G" + header.encode() + b";" + chunk + b"\x1b\\"
            )
        sys.stdout.buffer.flush()
        print()  # newline after image
    except Exception as e:
        print(f"  [kitty error: {e}] {path}")


def _iterm2_encode(path: Path, max_width: int = 400) -> None:
    """Display via iTerm2 inline images protocol (fallback for iTerm2)."""
    try:
        from PIL import Image
        import io
        img = Image.open(path).convert("RGB")
        w, h = img.size
        if w > max_width:
            scale = max_width / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        data = base64.standard_b64encode(buf.getvalue()).decode()
        sys.stdout.write(f"\x1b]1337;File=inline=1;width=400px:{data}\x07\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"  [iterm2 error: {e}] {path}")


def _detect_protocol() -> str:
    """Detect which image protocol to use. Returns 'kitty', 'iterm2', or 'none'."""
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if "kitty" in term:
        return "kitty"
    if "iterm" in term_program:
        return "iterm2"
    # Try sixel? For now, fall through to none.
    return "none"


def _show_image(path: Path, protocol: str, label: str = "", max_width: int = 380) -> None:
    if label:
        print(f"\n  {label}: {path.name}")
    if not path.exists():
        print(f"    (file not found)")
        return
    if protocol == "kitty":
        _encode_image_kitty(path, max_width=max_width)
    elif protocol == "iterm2":
        _iterm2_encode(path, max_width=max_width)
    else:
        _print_image_stats(path)


def _print_image_stats(path: Path) -> None:
    """Print basic image statistics when no graphics protocol is available."""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        print(f"    size={img.size}  mean={arr.mean():.1f}  path={path}")
    except Exception:
        print(f"    path={path}")


# ---------------------------------------------------------------------------
# Keyboard helper
# ---------------------------------------------------------------------------

def _getch() -> str:
    """Read a single character from stdin without echo."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _wait_for_next(interactive: bool) -> bool:
    """Returns True to continue, False to quit."""
    if not interactive:
        return True
    print("\n  [Enter/n = next | q = quit] ", end="", flush=True)
    try:
        ch = _getch()
    except Exception:
        return False
    print()
    return ch.lower() not in ("q",)


# ---------------------------------------------------------------------------
# Failure discovery
# ---------------------------------------------------------------------------

def find_failures(failures_dir: Path, pattern: Optional[str] = None):
    """
    Yield dicts with keys: name, reference, actual, diff.
    Looks for *_actual.png files and resolves their siblings.
    """
    if not failures_dir.exists():
        return

    for actual_path in sorted(failures_dir.glob("*_actual.png")):
        name = actual_path.stem[: -len("_actual")]
        if pattern and pattern.lower() not in name.lower():
            continue
        ref_in_failures = failures_dir / f"{name}_reference.png"
        ref_in_refs = DEFAULT_REFERENCE / f"{name}.png"
        ref_path = ref_in_failures if ref_in_failures.exists() else ref_in_refs
        diff_path = failures_dir / f"{name}_diff.png"
        yield {
            "name": name,
            "reference": ref_path,
            "actual": actual_path,
            "diff": diff_path,
        }


# ---------------------------------------------------------------------------
# Main display logic
# ---------------------------------------------------------------------------

def review(failures_dir: Path, pattern: Optional[str], protocol: str, interactive: bool) -> None:
    failures = list(find_failures(failures_dir, pattern))
    if not failures:
        print(f"No failures found in {failures_dir}")
        if pattern:
            print(f"  (filter: '{pattern}')")
        return

    print(f"Found {len(failures)} failure(s) in {failures_dir}\n")
    print("=" * 60)

    for i, f in enumerate(failures, 1):
        print(f"\n{'='*60}")
        print(f"  [{i}/{len(failures)}]  {f['name']}")
        print(f"{'='*60}")

        if protocol != "none":
            # Show reference, actual, diff side by side (vertically in terminal)
            _show_image(f["reference"], protocol, label="REFERENCE", max_width=380)
            _show_image(f["actual"],    protocol, label="ACTUAL",    max_width=380)
            _show_image(f["diff"],      protocol, label="DIFF",      max_width=380)
        else:
            # Text-only fallback
            print(f"  reference : {f['reference']}")
            print(f"  actual    : {f['actual']}")
            print(f"  diff      : {f['diff']}")
            for label, path in [("reference", f["reference"]), ("actual", f["actual"]), ("diff", f["diff"])]:
                _print_image_stats(path)

        if not _wait_for_next(interactive):
            print("Quit.")
            return

    print(f"\nAll {len(failures)} failure(s) reviewed.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Review image test failures in the terminal."
    )
    parser.add_argument(
        "--failures-dir",
        type=Path,
        default=DEFAULT_FAILURES,
        help=f"Directory containing failure images (default: {DEFAULT_FAILURES})",
    )
    parser.add_argument(
        "--filter",
        metavar="PATTERN",
        help="Only show failures whose name contains PATTERN (case-insensitive)",
    )
    parser.add_argument(
        "--protocol",
        choices=["auto", "kitty", "iterm2", "none"],
        default="auto",
        help="Image display protocol (default: auto-detect)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Print all failures without waiting for keypress",
    )
    args = parser.parse_args()

    if args.protocol == "auto":
        protocol = _detect_protocol()
        if protocol == "none":
            print("No graphics protocol detected (not Kitty or iTerm2).")
            print("Falling back to text output.  Set --protocol kitty/iterm2 to force.")
    else:
        protocol = args.protocol

    interactive = not args.no_interactive and sys.stdin.isatty()

    review(
        failures_dir=args.failures_dir,
        pattern=args.filter,
        protocol=protocol,
        interactive=interactive,
    )


if __name__ == "__main__":
    main()
