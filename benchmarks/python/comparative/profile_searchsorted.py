"""
Profile MLX searchsorted variants.

Usage examples:

# Quick timing of variants
python benchmarks/python/comparative/profile_searchsorted.py --mode time --a-size 100000 --v-size 1000 --iters 100

# Run under cProfile and write stats to file
python benchmarks/python/comparative/profile_searchsorted.py --mode cprofile --a-size 100000 --v-size 1000 --iters 20 --out stats.prof

# Print PID and wait so you can attach perf (recommended for C++ profiling)
python benchmarks/python/comparative/profile_searchsorted.py --mode wait --a-size 100000 --v-size 1000

"""

import argparse
import cProfile
import pstats
import sys
import time

import numpy as np

try:
    import mlx.core as mx
except Exception:
    mx = None


def make_data(a_size, v_size, dtype=np.float32):
    a = np.sort(np.random.rand(a_size).astype(dtype))
    v = np.random.rand(v_size).astype(dtype)
    return a, v


def time_fn(fn, iters=10):
    # Warmup
    fn()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


def run_integrated(a_np, v_np, side="left"):
    a = mx.array(a_np)
    v = mx.array(v_np)

    def _run():
        out = mx.searchsorted(a, v, side=side)
        mx.eval(out)

    return _run


def run_linear(a_np, v_np, side="left"):
    a = mx.array(a_np)
    v = mx.array(v_np)

    def _run():
        a_row = mx.reshape(a, (1, -1))
        v_col = mx.reshape(v, (-1, 1))
        if side == "left":
            mask = a_row < v_col
        else:
            mask = a_row <= v_col
        idx = mx.sum(mask, axis=1)
        mx.eval(idx)

    return _run


def run_binary(a_np, v_np, side="left"):
    a = mx.array(a_np)
    v = mx.array(v_np)

    def _run():
        N = int(a.shape[-1])
        M = int(v.shape[0])
        lo = mx.zeros((M,), dtype=mx.uint32)
        hi = mx.full((M,), N, dtype=mx.uint32)
        loops = (N + 1).bit_length()
        for _ in range(loops):
            mid = (lo + hi) // 2
            mid_idx = mx.reshape(mid, (-1, 1))
            mid_val = mx.take_along_axis(mx.reshape(a, (1, -1)), mid_idx, axis=1)
            mid_val = mx.reshape(mid_val, (-1,))
            if side == "left":
                cmp = mid_val < v
            else:
                cmp = mid_val <= v
            lo = mx.where(cmp, mid + 1, lo)
            hi = mx.where(cmp, hi, mid)
        mx.eval(lo)

    return _run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["time", "cprofile", "wait"], default="time")
    parser.add_argument("--a-size", type=int, default=100000)
    parser.add_argument("--v-size", type=int, default=1000)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument(
        "--out", type=str, default="stats.prof", help="cProfile output file"
    )
    args = parser.parse_args()

    if mx is None:
        print("mlx.core not importable. Activate your MLX Python environment.")
        sys.exit(1)

    mx.set_default_device(mx.cpu)

    a_np, v_np = make_data(args.a_size, args.v_size)

    integrated = run_integrated(a_np, v_np, args.side)
    linear = run_linear(a_np, v_np, args.side)
    binary = run_binary(a_np, v_np, args.side)

    if args.mode == "time":
        print("Timing (avg seconds per run):")
        print("integrated:", time_fn(integrated, iters=args.iters))
        print("linear:", time_fn(linear, iters=args.iters))
        print("binary:", time_fn(binary, iters=args.iters))

    elif args.mode == "cprofile":
        print("Running cProfile on integrated variant...")
        pr = cProfile.Profile()
        pr.enable()
        integrated()
        pr.disable()
        pr.dump_stats(args.out)
        print(
            f"Wrote cProfile stats to {args.out}. Use 'python -m pstats {args.out}' to view."
        )

    elif args.mode == "wait":
        # Print PID so user can attach perf or other native profilers.
        import os

        pid = os.getpid()
        print(
            f"PID: {pid}. Attach your profiler now (e.g. 'perf record -p {pid} -g'), then press Enter to continue."
        )
        input()
        # After attaching, run a few iterations to capture native-level activity
        t = time_fn(integrated, iters=args.iters)
        print("integrated (avg s):", t)


if __name__ == "__main__":
    main()
