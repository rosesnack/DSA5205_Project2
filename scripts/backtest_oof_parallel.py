"""Process execution adapter for the unchanged step-9 model implementation.

Reuse the registered design and exact monthly job functions. Transfer prepared
monthly arrays once per process/month, rather than copying them with every fit.
This changes execution only: seeds, candidate grids, losses and forecasts are
unchanged. The normal completed-cache path needs no worker processes.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import pickle
import time
import uuid

if __package__:
    from . import backtest_training_oof as implementation
else:
    import backtest_training_oof as implementation

_CACHE_PATH = None
_CACHE_DATA = None
_NUMERICAL_LIMITS = None
_PAYLOAD_FOLDER = None


def _initialize_worker():
    global _NUMERICAL_LIMITS
    _NUMERICAL_LIMITS = implementation.threadpool_limits(limits=1)


def _execute_job(job):
    global _CACHE_PATH, _CACHE_DATA
    function, short_task, path = job
    if _CACHE_PATH != path:
        with open(path, "rb") as stream:
            _CACHE_DATA = pickle.load(stream)
        _CACHE_PATH = path
    return function((*short_task, _CACHE_DATA))


class MonthlyProcessPool:
    def __init__(self, max_workers):
        self.pool = ProcessPoolExecutor(max_workers=max_workers, initializer=_initialize_worker)

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.pool.shutdown(wait=True, cancel_futures=True)

    def map(self, function, tasks):
        tasks = list(tasks)
        if not tasks:
            return iter(())
        prepared = tasks[0][-1]
        assert all(t[-1] is prepared for t in tasks)
        _PAYLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        path = _PAYLOAD_FOLDER / f"monthly_{uuid.uuid4().hex}.pkl"
        with path.open("wb") as stream:
            pickle.dump(prepared, stream, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            results = list(self.pool.map(_execute_job, [(function, t[:-1], str(path)) for t in tasks], chunksize=3))
        finally:
            # Only this explicitly created temporary payload is removed.
            path.unlink(missing_ok=True)
        return iter(results)


def run_training_oof(project_root, workers=4):
    global _PAYLOAD_FOLDER
    root = Path(project_root).resolve()
    out = root / "results/backtest/training_oof"
    out.mkdir(parents=True, exist_ok=True)
    _PAYLOAD_FOLDER = out / "checkpoints/executor_payloads"
    execution_path = out / "execution_config.json"
    execution = dict(backend="persistent ProcessPoolExecutor", workers=workers, numerical_threads=1,
        purpose="Execution speed only; exact registered monthly model functions and checkpoints reused",
        model_implementation_sha256=implementation.sha256(Path(implementation.__file__)),
        execution_adapter_sha256=implementation.sha256(Path(__file__)),
        registered_before_formal_process_fits=True)
    if execution_path.exists():
        assert json.loads(execution_path.read_text(encoding="utf-8")) == execution
    else:
        implementation.json_write(execution_path, execution)
    original = implementation.ThreadPoolExecutor
    implementation.ThreadPoolExecutor = MonthlyProcessPool
    try:
        return implementation.run_training_oof(root, workers)
    finally:
        implementation.ThreadPoolExecutor = original


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.benchmark:
        _PAYLOAD_FOLDER = args.project_root / "results/backtest/training_oof/checkpoints/executor_payloads"
        with implementation.threadpool_limits(limits=1):
            data, schedule = implementation.load_data(args.project_root)
            prepared, _ = implementation.month_inputs(data, schedule, "2012-01")
            tasks = [("elastic_net", ("XLB", "S2"), p, prepared) for p in implementation.EN_GRID]
            tasks += [("gbdt", ("XLB", "S2"), p, prepared) for p in implementation.GB_GRID]
            start = time.perf_counter()
            with MonthlyProcessPool(4) as pool:
                actual = list(pool.map(implementation.calibration_job, tasks))
            print("Process pilot:", len(actual), "fits; seconds", round(time.perf_counter()-start, 2), flush=True)
            # Verify the exact original job result, not only an approximate MSE.
            expected = implementation.calibration_job(tasks[0])
            assert actual[0] == expected
            print("PASS: process and original numerical result/audit are identical", flush=True)
    else:
        run_training_oof(args.project_root)
