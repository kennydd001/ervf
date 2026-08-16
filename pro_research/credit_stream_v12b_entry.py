"""Entrypoint for V12B.

The experiment module intentionally imports CuPy inside the hot-path helper so
module import remains cheap. Cleanup also needs the same module-global symbol;
install it explicitly here rather than introducing another target-code change.
"""
import cupy as cp
import credit_stream_v12b as experiment

experiment.cp = cp

if __name__ == "__main__":
    raise SystemExit(experiment.main())
