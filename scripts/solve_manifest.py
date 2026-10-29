#!/usr/bin/env python3
"""Content identities and atomic completion records for the long-running solve wrapper.

Full input hashes are cached against device/inode/size/mtime/ctime. Regular files
up to 8 MiB also receive a bounded content hash on each signature check, because
same-tick rewrites can preserve every recorded stat field. The initial identity
reads large graph/index/chain inputs once; unchanged runs reuse that cache.
Completed large outputs use exact sizes and stat signatures, avoiding an extra
read of hundreds of GiB of policy/value output on every PI round.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile

VERSION = 2
SMALL_SIGNATURE_BYTES = 8 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def stat_fields(st):
    return {"device": st.st_dev, "inode": st.st_ino, "size": st.st_size,
            "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns}


def signature_metadata(path):
    resolved = path.resolve()
    st = path.stat()
    result = {"path": str(path), **stat_fields(st)}
    if resolved != path:
        result["resolved_path"] = str(resolved)
    return result, st


def signature(path):
    # Keep the name the solver will open. Recording only its resolved target
    # would miss a symlink being repointed while that old target stays intact.
    path = Path(path).absolute()
    result, before = signature_metadata(path)
    if stat.S_ISREG(before.st_mode) and before.st_size <= SMALL_SIGNATURE_BYTES:
        with path.open("rb") as stream:
            if stat_fields(os.fstat(stream.fileno())) != stat_fields(before):
                raise ValueError(f"input changed while fingerprinting: {path}")
            # Bound the read even if a concurrent writer grows the file.
            data = stream.read(SMALL_SIGNATURE_BYTES + 1)
            digest = hashlib.sha256(data).hexdigest()
            if (len(data) != before.st_size
                    or stat_fields(os.fstat(stream.fileno())) != stat_fields(before)):
                raise ValueError(f"input changed while fingerprinting: {path}")
        if signature_metadata(path)[0] != result:
            raise ValueError(f"input changed while fingerprinting: {path}")
        result["small_sha256"] = digest
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".pending.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def content_identity(path, cache):
    before = signature(path)
    cache_path = Path(cache) / (hashlib.sha256(before["path"].encode()).hexdigest() + ".json")
    try:
        saved = json.loads(cache_path.read_text())
    except (OSError, ValueError):
        saved = {}
    cached_hash = saved.get("sha256") if isinstance(saved, dict) else None
    if (isinstance(saved, dict) and saved.get("signature") == before
            and isinstance(cached_hash, str) and len(cached_hash) == 64
            and all(c in "0123456789abcdef" for c in cached_hash)):
        return {"path": before.get("resolved_path", before["path"]), "sha256": cached_hash}
    full_hash = before.get("small_sha256")
    if full_hash is None:
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
        if signature(path) != before:
            raise ValueError(f"input changed while hashing: {path}")
        full_hash = digest.hexdigest()
    saved = {"signature": before, "sha256": full_hash}
    atomic_json(cache_path, saved)
    return {"path": before.get("resolved_path", before["path"]), "sha256": saved["sha256"]}


def layer_states(pruned, layer):
    path = Path(pruned) / f"layer{layer}q_pruned.field_index.bin"
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"FCHNIDX1":
        raise ValueError(f"invalid pruned index header: {path}")
    version, blocks, states = struct.unpack("<IIQ", header[8:])
    if version != 1 or path.stat().st_size != 24 + blocks * 16:
        raise ValueError(f"invalid pruned index size/version: {path}")
    return states


def expected_outputs(pruned, kind, paths):
    if kind in ("seed", "value"):
        return [(paths[0], layer_states(pruned, 0) * 4, True)]
    if kind == "policy":
        return [(str(Path(paths[0]) / f"layer{k}.policy.u16.bin"), layer_states(pruned, k) * 2, False) for k in range(10)]
    if kind == "full-boundary":
        return (expected_outputs(pruned, "full", paths[:2])
                + expected_outputs(pruned, "value", paths[2:]))
    if kind == "full":
        values, policies = paths
        return ([(str(Path(values) / f"layer{k}.values.f32.bin"), layer_states(pruned, k) * 4, False) for k in range(10)]
                + expected_outputs(pruned, "policy", [policies]))
    if kind == "stable":
        boundary, policy, marker, round_marker = paths
        return (expected_outputs(pruned, "value", [boundary])
                + expected_outputs(pruned, "policy", [policy])
                + [(marker, None, False), (round_marker, None, False)])
    raise ValueError(f"unknown output kind: {kind}")


def finite_values(path):
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            if len(chunk) % 4 or any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", chunk)):
                raise ValueError(f"nonfinite or malformed boundary value: {path}")


def input_paths(graph, pruned, solver):
    paths = [graph, solver]
    for k in range(10):
        paths += [str(Path(pruned) / f"layer{k}q_pruned.{suffix}")
                  for suffix in ("field_index.bin", "chains.u24.bin")]
    return paths


def guard_snapshot(snapshot):
    records = json.loads(snapshot)
    if not isinstance(records, list) or not records:
        raise ValueError("empty or malformed input snapshot")
    for record in records:
        if not isinstance(record, dict) or signature(record["path"]) != record:
            raise ValueError(f"input changed during solve: {record.get('path') if isinstance(record, dict) else record}")


def main():
    cmd, *args = sys.argv[1:]
    if cmd == "context":
        graph, pruned, solver, cache = args
        inputs = [content_identity(path, cache) for path in input_paths(graph, pruned, solver)]
        configuration = {key: value for key, value in os.environ.items()
                         if key.startswith(("FULL_MDP_", "ONE_LOOP_MDP_", "PI_"))}
        print(hashlib.sha256(canonical([VERSION, inputs, configuration])).hexdigest())
    elif cmd == "snapshot-inputs":
        print(canonical([signature(path) for path in input_paths(*args)]).decode())
    elif cmd == "snapshot":
        print(canonical([signature(path) for path in args]).decode())
    elif cmd == "guard":
        guard_snapshot(args[0])
    elif cmd == "identity":
        path, cache = args
        print(hashlib.sha256(canonical(content_identity(path, cache))).hexdigest())
    elif cmd == "key":
        print(hashlib.sha256(canonical([VERSION, args])).hexdigest())
    elif cmd == "check":
        path, context, key = args
        try:
            record = json.loads(Path(path).read_text())
            valid = (record["version"] == VERSION and record["context"] == context
                     and record["record_path"] == str(Path(path).resolve())
                     and (key == "-" or record["key"] == key)
                     and bool(record["outputs"])
                     and all(signature(out["path"]) == out for out in record["outputs"]))
        except (OSError, ValueError, KeyError, TypeError):
            valid = False
        sys.exit(0 if valid else 1)
    elif cmd == "record":
        path, context, key, pruned, kind, *paths = args
        outputs = []
        for output, expected, finite in expected_outputs(pruned, kind, paths):
            before = signature(output)
            if expected is not None and before["size"] != expected:
                raise ValueError(f"incomplete output {output}: {before['size']} bytes, expected {expected}")
            if finite:
                finite_values(output)
            if signature(output) != before:
                raise ValueError(f"output changed while checking: {output}")
            outputs.append(before)
        atomic_json(path, {"version": VERSION, "record_path": str(Path(path).resolve()),
                           "context": context, "key": key, "outputs": outputs})
    elif cmd == "states":
        print(layer_states(args[0], int(args[1])))
    elif cmd == "copy":
        source, target = map(Path, args)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=target.name + ".pending.", dir=target.parent)
        try:
            with source.open("rb") as inp, os.fdopen(fd, "wb") as out:
                while chunk := inp.read(8 * 1024 * 1024):
                    out.write(chunk)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    else:
        raise ValueError(f"unknown manifest command: {cmd}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, struct.error) as error:
        print(f"solve completion check: {error}", file=sys.stderr)
        sys.exit(1)
