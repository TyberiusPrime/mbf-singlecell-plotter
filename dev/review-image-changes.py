#!/usr/bin/env python3
"""
Review image test failures in the terminal.

Usage:
    python dev/review-image-changes.py [--failures-dir PATH] [--filter PATTERN]

Finds all *_actual.png files in tests/failures/ and shows the reference and
actual images side by side.

Controls:
    Space        → toggle: cycle between full-size reference / full-size actual
                   (for flicker comparison; side-by-side is the resting view)
    a            → accept: copy actual → canonical reference, advance to next
    Enter / n    → next failure
    q            → quit
"""

import argparse
import os
import shutil
import sys
import base64
import termios
import tty
from pathlib import Path
from typing import Optional

TESTS_DIR = Path(__file__).parent.parent / "tests"
DEFAULT_FAILURES = TESTS_DIR / "failures"
DEFAULT_REFERENCE = TESTS_DIR / "reference_images"


# ---------------------------------------------------------------------------
# Image display helpers
# ---------------------------------------------------------------------------

def _send_kitty(path: Path, max_width: int) -> None:
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
        data = base64.standard_b64encode(buf.getvalue())
        chunk = 4096
        chunks = [data[i:i + chunk] for i in range(0, len(data), chunk)]
        for i, c in enumerate(chunks):
            m = 0 if i == len(chunks) - 1 else 1
            hdr = f"a=T,f=100,m={m}" if i == 0 else f"m={m}"
            sys.stdout.buffer.write(b"\x1b_G" + hdr.encode() + b";" + c + b"\x1b\\")
        sys.stdout.buffer.flush()
        print()
    except Exception as e:
        print(f"  [kitty error: {e}] {path}")


def _send_iterm2(path: Path, max_width: int) -> None:
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
        sys.stdout.write(f"\x1b]1337;File=inline=1;width={max_width}px:{data}\x07\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"  [iterm2 error: {e}] {path}")


def _composite_side_by_side(left: Path, right: Path, gap: int = 20) -> "Image":
    from PIL import Image
    a = Image.open(left).convert("RGB")
    b = Image.open(right).convert("RGB")
    h = max(a.height, b.height)
    w = a.width + gap + b.width
    out = Image.new("RGB", (w, h), (220, 220, 220))
    out.paste(a, (0, (h - a.height) // 2))
    out.paste(b, (a.width + gap, (h - b.height) // 2))
    return out


def _terminal_pixel_width() -> int:
    """Best-effort terminal width in pixels."""
    # Try TIOCGWINSZ ioctl (gives both cell and pixel dimensions)
    try:
        import fcntl, struct
        packed = fcntl.ioctl(sys.stdout.fileno(), 0x5413,  # TIOCGWINSZ
                             b"\x00" * 8)
        _rows, _cols, px_w, _px_h = struct.unpack("HHHH", packed)
        if px_w > 0:
            return px_w
    except Exception:
        pass
    # Fallback: assume ~9 pixels per column
    cols = shutil.get_terminal_size((120, 40)).columns
    return cols * 9


def _show_composite(left: Path, right: Path, protocol: str, max_width: Optional[int] = None) -> None:
    if max_width is None:
        max_width = _terminal_pixel_width()
    if protocol == "none":
        _print_stats(left, "REFERENCE")
        _print_stats(right, "ACTUAL")
        return
    try:
        from PIL import Image
        import io
        img = _composite_side_by_side(left, right)
        w, h = img.size
        if w > max_width:
            scale = max_width / w
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tmp = Path("/tmp/_review_composite.png")
        img.save(tmp)
        if protocol == "kitty":
            _send_kitty(tmp, max_width)
        else:
            _send_iterm2(tmp, max_width)
    except Exception as e:
        print(f"  [composite error: {e}]")
        _print_stats(left, "REFERENCE")
        _print_stats(right, "ACTUAL")



def _print_stats(path: Path, label: str = "") -> None:
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = __import__("numpy").array(img)
        tag = f"  {label:<12}" if label else "  "
        print(f"{tag} size={img.size}  mean={arr.mean():.1f}  {path}")
    except Exception:
        print(f"  {path}")


# ---------------------------------------------------------------------------
# Keyboard
# ---------------------------------------------------------------------------

def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


# ---------------------------------------------------------------------------
# Failure discovery
# ---------------------------------------------------------------------------

def find_failures(failures_dir: Path, pattern: Optional[str] = None):
    if not failures_dir.exists():
        return
    for actual_path in sorted(failures_dir.glob("*_actual.png")):
        name = actual_path.stem[: -len("_actual")]
        if pattern and pattern.lower() not in name.lower():
            continue
        ref_in_failures = failures_dir / f"{name}_reference.png"
        ref_canonical = DEFAULT_REFERENCE / f"{name}.png"
        ref_path = ref_in_failures if ref_in_failures.exists() else ref_canonical
        yield {"name": name, "reference": ref_path, "actual": actual_path}


# ---------------------------------------------------------------------------
# Accept helper
# ---------------------------------------------------------------------------

def _accept(f: dict) -> None:
    """Copy actual → canonical reference and clean up failures dir."""
    dest = DEFAULT_REFERENCE / f"{f['name']}.png"
    shutil.copy2(f["actual"], dest)
    # Remove artefacts from failures dir
    for suffix in ("_actual.png", "_reference.png", "_diff.png"):
        p = DEFAULT_FAILURES / f"{f['name']}{suffix}"
        if p.exists():
            p.unlink()
    print(f"  ✓ accepted  →  {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _clear() -> None:
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


def review(failures_dir: Path, pattern: Optional[str], protocol: str, interactive: bool) -> None:
    failures = list(find_failures(failures_dir, pattern))
    if not failures:
        print(f"No failures found in {failures_dir}")
        if pattern:
            print(f"  (filter: '{pattern}')")
        return

    print(f"Found {len(failures)} failure(s)\n")

    i = 0
    while i < len(failures):
        f = failures[i]
        # flipped=False: reference left, actual right
        # flipped=True:  actual left, reference right
        flipped = False

        while True:
            _clear()
            print(f"  [{i+1}/{len(failures)}]  {f['name']}")
            print(f"  ref:    file://{f['reference'].resolve()}")
            print(f"  actual: file://{f['actual'].resolve()}")
            if flipped:
                left, right = f["actual"], f["reference"]
                print("  [actual (left)  |  reference (right)]")
            else:
                left, right = f["reference"], f["actual"]
                print("  [reference (left)  |  actual (right)]")
            print()
            _show_composite(left, right, protocol)

            if not interactive:
                i += 1
                break

            print("\n  Space=swap  a=accept  Enter/n=next  q=quit  ", end="", flush=True)
            try:
                ch = _getch()
            except Exception:
                i += 1
                break
            print()

            if ch == " ":
                flipped = not flipped
            elif ch == "a":
                _accept(f)
                i += 1
                break
            elif ch.lower() == "q":
                print("Quit.")
                return
            elif ch in ("\r", "\n", "n"):
                i += 1
                break
            # any other key: redraw

    print(f"\nDone.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _detect_protocol() -> str:
    term = os.environ.get("TERM", "").lower()
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    if "kitty" in term:
        return "kitty"
    if "iterm" in term_program:
        return "iterm2"
    return "none"


def main():
    parser = argparse.ArgumentParser(description="Review image test failures.")
    parser.add_argument("--failures-dir", type=Path, default=DEFAULT_FAILURES)
    parser.add_argument("--filter", metavar="PATTERN")
    parser.add_argument(
        "--protocol", choices=["auto", "kitty", "iterm2", "none"], default="auto"
    )
    parser.add_argument("--no-interactive", action="store_true")
    args = parser.parse_args()

    protocol = _detect_protocol() if args.protocol == "auto" else args.protocol
    if protocol == "none" and args.protocol == "auto":
        print("No graphics protocol detected; falling back to text.  Use --protocol to force.")

    interactive = not args.no_interactive and sys.stdin.isatty()
    review(
        failures_dir=args.failures_dir,
        pattern=args.filter,
        protocol=protocol,
        interactive=interactive,
    )


if __name__ == "__main__":
    main()
