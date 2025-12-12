#!/usr/bin/env python3
"""
Run MLX vs PyTorch searchsorted benchmarks across multiple sizes.
This script invokes the existing bench_mlx.py and bench_torch.py scripts
and prints their stdout for each size.

Usage: python run_searchsorted_compare.py --sizes 100 1000 10000 --cpu
"""
import argparse
import shutil
import subprocess
import sys

HERE = "$(pwd)"


def which(cmd):
    return shutil.which(cmd) is not None


def run_command(cmd):
    print("+ " + " ".join(cmd))
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        print(out)
    except subprocess.CalledProcessError as e:
        print(e.output)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        required=False,
        default=[100, 1000, 10000, 100000],
        help="Sizes to test for the sorted axis",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU runs")
    # bench_mlx.py and bench_torch.py use internal iteration counts
    # so we don't pass an --iters flag to them.
    args = parser.parse_args()

    # ensure scripts exist in this folder
    import os

    base = os.path.dirname(__file__)
    bench_mlx_path = os.path.join(base, "bench_mlx.py")
    bench_torch_path = os.path.join(base, "bench_torch.py")
    # MLX provides three implementations; we'll run all three and one PyTorch run
    bench_mlx_cmd = [sys.executable, bench_mlx_path]
    mlx_variants = [
        "searchsorted_integrated",
        "searchsorted_linear",
        "searchsorted_binary",
    ]
    bench_torch_cmd = [sys.executable, bench_torch_path]
    torch_variant = "searchsorted"
    if not os.path.exists(bench_mlx_path) or not os.path.exists(bench_torch_path):
        print("Error: expected bench_mlx.py and bench_torch.py in", base)
        sys.exit(2)

    for size in args.sizes:
        print("\n=== SIZE {} ===".format(size))

        # Run MLX variants
        for variant in mlx_variants:
            print(f"\n-- MLX ({variant}) --")
            run_command(
                bench_mlx_cmd
                + [variant, "--size", str(size)]
                + (["--cpu"] if args.cpu else [])
            )

        # Run PyTorch (single implementation)
        print("\n-- PyTorch (torch.searchsorted) --")
        run_command(
            bench_torch_cmd
            + [torch_variant, "--size", str(size)]
            + (["--cpu"] if args.cpu else [])
        )


if __name__ == "__main__":
    main()
