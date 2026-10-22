"""Build step for docker/Dockerfile: optional PLE round-trip timing, switched on by VLLM_PLE_TIMING=1.

Decode's disk PLE path runs between CUDA graphs: the host waits for the step's graph (the ids copy is
stream-ordered behind it), reads the table rows, and only then can the next graph launch. While
VLLM_PLE_TIMING=1 is set, two wall clocks record that span so the GPU idle share of a decode step can be
computed without a profiler:

- vllm's b12x_ple.py: total host time inside prepare_disk (the whole between-graphs span).
- flashinfer's _shared/disk_table.py: that span split into the ids synchronize and the row read.

Usage: python3 ple_timing.py B12X_PLE_PATH SHA256 DISK_TABLE_PATH SHA256
Stops without writing when a file's sha256 differs: a patch made against other code could change the
model's numerics. With the environment variable unset the inserted code does nothing.
"""
import ast
import hashlib
import pathlib
import subprocess
import sys

B12X_ANCHOR = (
    "        self._disk_prepared_tokens = -1\n"
    "        self._disk_prepared = False\n"
    "        self._run_embedding(input_ids, query_start_loc, ngram_context)\n"
)
B12X_BODY = (
    "        self._disk_prepared_tokens = -1\n"
    "        self._disk_prepared = False\n"
    "        _ple_enter()\n"
    "        self._run_embedding(input_ids, query_start_loc, ngram_context)\n"
    "        _ple_exit()\n"
)
B12X_INJECT_ANCHOR = "from vllm.utils.torch_utils import direct_register_custom_op\n"
B12X_INJECT = B12X_INJECT_ANCHOR + """
# VLLM_PLE_TIMING=1 (recipe build step): host wall time inside prepare_disk, printed every 100 calls.
import os as _ple_os
import sys as _ple_sys
import time as _ple_time
_PLE_T = {"n": 0, "ms": 0.0, "t0": 0.0}


def _ple_enter() -> None:
    if _ple_os.environ.get("VLLM_PLE_TIMING") == "1":
        _PLE_T["t0"] = _ple_time.perf_counter()


def _ple_exit() -> None:
    if not _PLE_T["t0"]:
        return
    _PLE_T["ms"] += (_ple_time.perf_counter() - _PLE_T["t0"]) * 1e3
    _PLE_T["n"] += 1
    _PLE_T["t0"] = 0.0
    if _PLE_T["n"] % 100 == 0:
        _ple_sys.stderr.write(
            f"PLE_TIMING prepare_disk calls={_PLE_T['n']} total_ms={_PLE_T['ms']:.0f} "
            f"per_call_ms={_PLE_T['ms'] / _PLE_T['n']:.3f}\\n")
        _ple_sys.stderr.flush()
"""

DISK_ANCHOR = (
    "    def _read_staged(self, count: int) -> None:\n"
    "        # Completes ID production and all prior cache readers before host writes.\n"
    "        with torch.cuda.device(self.device):\n"
    "            self._ids_ready.synchronize()\n"
    "            if self._gds is not None:\n"
    "                self._gds.read(self._ids_buffer, count, self._transaction_stream)\n"
    "                return\n"
    "            self._native.ple_reader_run(\n"
    "                self._reader,\n"
    "                self._ids_buffer,\n"
    "                self._weight_buffer,\n"
    "                self._scale_buffer,\n"
    "                count,\n"
    "            )\n"
)
DISK_BODY = (
    "    def _read_staged(self, count: int) -> None:\n"
    "        # Completes ID production and all prior cache readers before host writes.\n"
    "        with torch.cuda.device(self.device):\n"
    "            _ple_t = _ple_now()\n"
    "            self._ids_ready.synchronize()\n"
    "            _ple_add(\"sync_ms\", _ple_t)\n"
    "            _ple_t = _ple_now()\n"
    "            if self._gds is not None:\n"
    "                self._gds.read(self._ids_buffer, count, self._transaction_stream)\n"
    "                _ple_add(\"read_ms\", _ple_t)\n"
    "                return\n"
    "            self._native.ple_reader_run(\n"
    "                self._reader,\n"
    "                self._ids_buffer,\n"
    "                self._weight_buffer,\n"
    "                self._scale_buffer,\n"
    "                count,\n"
    "            )\n"
    "            _ple_add(\"read_ms\", _ple_t)\n"
)
DISK_INJECT_ANCHOR = "import torch\n"
DISK_INJECT = DISK_INJECT_ANCHOR + """
# VLLM_PLE_TIMING=1 (recipe build step): the disk reader's phases, printed every 100 reads.
import os as _ple_os
import sys as _ple_sys
import time as _ple_time
_PLE_R = {"reads": 0, "sync_ms": 0.0, "read_ms": 0.0}


def _ple_now() -> float:
    return _ple_time.perf_counter() if _ple_os.environ.get("VLLM_PLE_TIMING") == "1" else 0.0


def _ple_add(key: str, t0: float) -> None:
    if not t0:
        return
    _PLE_R[key] += (_ple_time.perf_counter() - t0) * 1e3
    if key == "read_ms":
        _PLE_R["reads"] += 1
        if _PLE_R["reads"] % 100 == 0:
            _ple_sys.stderr.write(
                f"PLE_TIMING reads={_PLE_R['reads']} sync_ms={_PLE_R['sync_ms']:.0f} "
                f"read_ms={_PLE_R['read_ms']:.0f} per_read_sync_ms={_PLE_R['sync_ms'] / _PLE_R['reads']:.3f} "
                f"per_read_read_ms={_PLE_R['read_ms'] / _PLE_R['reads']:.3f}\\n")
            _ple_sys.stderr.flush()
"""


def patch(path: pathlib.Path, want: str, replacements: tuple[tuple[str, str], ...]) -> None:
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        sys.exit(f"{path}: sha256 {got}, expected {want}. This base image ships other code; "
                 "set IMAGE to the base image itself to serve it without the timing change.")
    src = raw.decode()
    for anchor, body in replacements:
        if src.count(anchor) != 1:
            sys.exit(f"{path}: expected exactly one {anchor[:60]!r}")
        src = src.replace(anchor, body, 1)
    ast.parse(src)
    path.write_text(src)
    for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        pyc.unlink()
    print(f"patched {path.name}")


def main() -> None:
    b12x = pathlib.Path(sys.argv[1])
    disk = pathlib.Path(sys.argv[3])
    patch(b12x, sys.argv[2], ((B12X_ANCHOR, B12X_BODY), (B12X_INJECT_ANCHOR, B12X_INJECT)))
    patch(disk, sys.argv[4], ((DISK_ANCHOR, DISK_BODY), (DISK_INJECT_ANCHOR, DISK_INJECT)))
    probe = ("import importlib; m = importlib.import_module"
             "('flashinfer.experimental.b12x.sequence._shared.disk_table'); "
             "assert callable(m._ple_now) and callable(m._ple_add)")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"disk_table import check failed: {out.stderr.strip()[-1500:]}")


if __name__ == "__main__":
    main()
