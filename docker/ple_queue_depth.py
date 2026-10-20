"""Build step for docker/Dockerfile: let PLE_QUEUE_DEPTH set DiskTable's default io_uring queue depth.

FlashInfer's b12x PLE disk table (flashinfer-python 0.7.1, Apache License 2.0) keeps at most 64 row reads
in flight, a fixed default with no setting. This rewrites that one default to read PLE_QUEUE_DEPTH when
the module is imported, 256 when unset, and changes nothing else in the file.

Usage: python3 ple_queue_depth.py PATH SHA256
Stops without writing when PATH's sha256 is not SHA256: a patch made against other code could change
the model's numerics. After writing it imports DiskTable twice and checks the default it got.
"""
import hashlib
import os
import pathlib
import subprocess
import sys

OLD_DEFAULT = "queue_depth: int = 64"
NEW_DEFAULT = 'queue_depth: int = int(os.environ.get("PLE_QUEUE_DEPTH", "256"))'
FUTURE = "from __future__ import annotations\n"
MODULE = "flashinfer.experimental.b12x.sequence.ple_embedding._disk"


def main() -> None:
    path = pathlib.Path(sys.argv[1])
    want = sys.argv[2]
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        sys.exit(f"{path}: sha256 {got}, expected {want}. This base image ships another _disk.py; "
                 "set IMAGE to the base image itself to serve it without the queue-depth change.")
    src = raw.decode()
    for needle in (OLD_DEFAULT, FUTURE):
        if src.count(needle) != 1:
            sys.exit(f"{path}: expected exactly one {needle!r}")
    src = src.replace(FUTURE, FUTURE + "\nimport os\n", 1).replace(OLD_DEFAULT, NEW_DEFAULT, 1)
    path.write_text(src)
    # Drop the image's compiled copy so Python cannot load the unpatched bytecode.
    for pyc in (path.parent / "__pycache__").glob(path.stem + ".*.pyc"):
        pyc.unlink()
    for env, depth in (({}, 256), ({"PLE_QUEUE_DEPTH": "64"}, 64)):
        probe = (f"import inspect; from {MODULE} import DiskTable; "
                 "print(inspect.signature(DiskTable.__init__).parameters['queue_depth'].default)")
        out = subprocess.run([sys.executable, "-c", probe], env={**os.environ, **env},
                             capture_output=True, text=True)
        if out.returncode != 0 or out.stdout.strip().splitlines()[-1:] != [str(depth)]:
            sys.exit(f"DiskTable default with {env or 'PLE_QUEUE_DEPTH unset'}: {out.stdout.strip()!r} "
                     f"{out.stderr.strip()[-2000:]}, expected {depth}")
        print(f"DiskTable queue_depth default {depth} with {env or 'PLE_QUEUE_DEPTH unset'}")


if __name__ == "__main__":
    main()
