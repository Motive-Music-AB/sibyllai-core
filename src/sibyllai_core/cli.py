import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # (Optional) Suppress some additional TF logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='keras')

print("=== CLI MODULE LOADED (DEBUG) ===")

import argparse, pathlib, os
from .pipeline import analyse

DEFAULT_OUT = pathlib.Path(__file__).resolve().parents[2] / "outputs"  # repo/outputs

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Auto-spotting CLI")
    p.add_argument("src", help="Audio or video file (input)")
    p.add_argument("--fps", type=int, default=25, help="Time-code FPS")
    p.add_argument("--thr", type=float, default=0.5, help="Mood prob threshold")
    return p

def main(argv=None):
    print("CLI started")
    args = build_parser().parse_args(argv)
    analyse(pathlib.Path(args.src), DEFAULT_OUT, args.thr, args.fps)
    print(f"Analysis complete. Output should be in: {DEFAULT_OUT}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
