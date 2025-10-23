import argparse, glob, os
from src.pipeline import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, help="Path to one .mat file")
    parser.add_argument("--data_dir", type=str, default="data", help="Folder with .mat files")
    args = parser.parse_args()

    files = []
    if args.mat:
        files = [args.mat]
    else:
        # 这里按你的命名做通配
        files = sorted(glob.glob(os.path.join(args.data_dir, "example*kinect.mat")))

    if not files:
        raise SystemExit("No .mat files found. Check path/pattern.")

    for f in files:
        print(f"\n=== Running on {f} ===")
        run(f)
