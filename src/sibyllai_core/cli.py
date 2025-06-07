import argparse, pathlib, os
from .pipeline import analyse

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"  # repo/outputs

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-spotting CLI")
    p.add_argument("src", help="Audio or video file (input)")
    p.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help=f"Output folder (default: {DEFAULT_OUT})",
    )
    p.add_argument("--fps", type=int, default=25, help="Time-code FPS")
    p.add_argument("--thr", type=float, default=0.5, help="Mood prob threshold")
    return p

def main(argv=None):
    print("CLI started")
    args = build_parser().parse_args(argv)
    analyse(pathlib.Path(args.src), pathlib.Path(args.out), args.thr, args.fps)
    print(f"Analysis complete. Output should be in: {args.out}")
