import argparse

import numpy as np

from .time_utils import time_fn


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
    parser = argparse.ArgumentParser(description="Benchmark NumPy searchsorted")
    parser.add_argument("--side", choices=["left", "right"], default="left")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument(
        "--a-sizes", type=int, nargs="*", default=[1_000, 10_000, 100_000, 1_000_000]
    )
    parser.add_argument(
        "--v-sizes", type=int, nargs="*", default=[10, 100, 1_000, 1_000]
    )
    args = parser.parse_args()

    dtype = np.float32 if args.dtype == "float32" else np.float64

    np_results = bench_searchsorted_numpy(args.a_sizes, args.v_sizes, args.side, dtype)
    fmt_results("NumPy", np_results)


if __name__ == "__main__":
    main()
