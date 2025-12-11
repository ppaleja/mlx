import argparse
import time
import numpy as np

try:
    import mlx.core as mx
except Exception as e:
    mx = None

def time_fn(fn, iters: int = 10):
    # Simple timing helper: run fn() iters times and return average seconds
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    end = time.perf_counter()
    return (end - start) / iters


def bench_searchsorted_mx(a_sizes, v_sizes, side, dtype):
    if mx is None:
        raise RuntimeError("mlx.core not available. Install MLX Python first.")

    results = []
    for n in a_sizes:
        for m in v_sizes:
            # Create sorted array 'a' and values 'v'
            a_np = np.sort(np.random.rand(n).astype(dtype))
            v_np = np.random.rand(m).astype(dtype)

            a = mx.array(a_np)
            v = mx.array(v_np)

            # Warm-up
            idx = mx.searchsorted(a, v, side=side)
            mx.eval(idx)

            def _run():
                out = mx.searchsorted(a, v, side=side)
                mx.eval(out)
                return out

            t = time_fn(_run)
            results.append((n, m, t))
    return results


def bench_searchsorted_numpy(a_sizes, v_sizes, side, dtype):
    results = []
    for n in a_sizes:
        for m in v_sizes:
            a = np.sort(np.random.rand(n).astype(dtype))
            v = np.random.rand(m).astype(dtype)

            # Warm-up
            _ = np.searchsorted(a, v, side=side)

            def _run():
                return np.searchsorted(a, v, side=side)

            t = time_fn(_run)
            results.append((n, m, t))
    return results


def fmt_results(tag, results):
    print(f"\n{tag} results (a_size, v_size, time_ms):")
    for n, m, t in results:
        print(f"{n:>8} {m:>8} {t*1e3:8.3f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark searchsorted for MLX vs NumPy")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--a-sizes", type=int, nargs="*", default=[1_000, 10_000, 100_000, 1_000_000])
    parser.add_argument("--v-sizes", type=int, nargs="*", default=[10, 100, 1_000, 10_000])
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    np_results = bench_searchsorted_numpy(args.a_sizes, args.v_sizes, args.side, dtype)
    fmt_results("NumPy", np_results)

    try:
        mx_results = bench_searchsorted_mx(args.a_sizes, args.v_sizes, args.side, dtype)
        fmt_results("MLX", mx_results)
    except Exception as e:
        print(f"\nMLX benchmark skipped: {e}")


if __name__ == "__main__":
    main()
