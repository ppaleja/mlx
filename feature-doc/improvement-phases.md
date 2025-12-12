Establish baseline performance metrics — Run profile_searchsorted.py with --mode time --a-size 100000 --v-size 1000 --iters 100 and compare.py --filter searchsorted; record timings as baseline; verify all C++ tests pass with ctest -R searchsorted.

Phase 1: Move allocation inside dispatch (LOC: -1, Impact: 10-15%) — Move out.set_data(allocator::malloc(...)) from line 503 into the dispatch lambda after line 517; rebuild; run C++ tests to confirm correctness; re-run benchmarks and compare to baseline; commit if speedup confirmed.

Phase 2: Add contiguous fast-path (LOC: +60, Impact: 3-5x for contiguous) — Insert new code block after line 571 in search_sorted<T> checking a.flags().contiguous && a.strides()[axis] == 1 and using raw pointer arithmetic with std::lower_bound/std::upper_bound on const T* instead of vector copy; rebuild; validate C++ tests pass (especially contiguous cases); benchmark and confirm 3-5x speedup on contiguous inputs; commit.

Phase 3: Eliminate vector allocation in slow path (LOC: -8, Impact: 2-3x for strided) — Replace lines 578-591 vector allocation/copy loop with direct StridedIterator<const T> passed to std::lower_bound; rebuild; verify C++ tests (especially strided/broadcast cases); benchmark strided inputs and confirm 2-3x improvement on slow path; commit.

Phase 4: Inline iterator steps (LOC: +10-15, Impact: 5-10%) — Replace a_it.step() and v_it.step() calls with direct offset arithmetic for simple broadcast patterns; rebuild; run full test suite including edge cases; benchmark and validate 5-10% gain; commit if successful (skip if complexity too high or gains minimal).

Phase 5: Hoist NaN checks out of comparator (LOC: +25, Impact: 5-10% for floats) — Add pre-scan for NaN presence before binary search; use std::less<T> when no NaNs detected instead of nan_aware_less<T>; rebuild; carefully test float16/float32/float64 with NaN arrays from C++ tests; benchmark floating-point cases; commit if validated.