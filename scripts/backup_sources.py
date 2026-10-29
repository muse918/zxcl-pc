#!/usr/bin/env python3
"""Verified, append-only local source snapshots. No generated data or retention deletion."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

MIB = 1 << 20
GENERATED_ROOTS = {"target", "work", "data"}
LEGACY_SCOPE = "tracked and non-ignored working-tree files plus Git metadata; excludes generated datasets/build/scratch"
LEGACY_PUBLICATION_KEYS = {"archive", "archive_sha256", "archive_bytes", "verified"}


class BackupError(RuntimeError):
    pass


class SourceChanged(BackupError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(MIB):
            digest.update(chunk)
    return digest.hexdigest()


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path, value, *, replace=True):
    descriptor, pending = tempfile.mkstemp(prefix=".pending-json-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(pending, path)
        else:
            os.link(pending, path)
            os.unlink(pending)
        sync_directory(path.parent)
    finally:
        if os.path.exists(pending):
            os.unlink(pending)


def git_files(source):
    result = subprocess.run(["git", "-C", str(source), "ls-files", "--cached", "--others",
                             "--exclude-standard", "-z"], capture_output=True, check=False)
    if result.returncode:
        raise BackupError("cannot read the source Git inventory")
    return {os.fsdecode(value) for value in result.stdout.split(b"\0") if value}


def safe_relative(value):
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or not value:
        raise BackupError("unsafe path in source inventory")
    return relative


def metadata(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def inventory_record(source, relative, max_file_bytes):
    path = source / relative
    try:
        before = path.lstat()
        record = {"path": relative, "mode": stat.S_IMODE(before.st_mode)}
        if stat.S_ISLNK(before.st_mode):
            target = os.readlink(path)
            payload = os.fsencode(target)
            record.update(kind="symlink", target=target, size=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        elif stat.S_ISDIR(before.st_mode):
            record.update(kind="directory")
        elif stat.S_ISREG(before.st_mode):
            if before.st_size > max_file_bytes:
                raise BackupError(f"source file exceeds the backup size limit: {relative}")
            digest = hashlib.sha256()
            descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(descriptor, "rb") as stream:
                if metadata(os.fstat(stream.fileno())) != metadata(before):
                    raise SourceChanged(f"source changed while opening: {relative}")
                remaining = before.st_size
                while remaining:
                    data = stream.read(min(MIB, remaining))
                    if not data:
                        raise SourceChanged(f"source was truncated: {relative}")
                    digest.update(data)
                    remaining -= len(data)
                if stream.read(1) or metadata(os.fstat(stream.fileno())) != metadata(before):
                    raise SourceChanged(f"source changed while hashing: {relative}")
            record.update(kind="file", size=before.st_size, sha256=digest.hexdigest())
        else:
            raise BackupError(f"unsupported special source file: {relative}")
        if metadata(path.lstat()) != metadata(before):
            raise SourceChanged(f"source changed while inspecting: {relative}")
        return record
    except (FileNotFoundError, NotADirectoryError) as error:
        raise SourceChanged(f"source disappeared while inspecting: {relative}") from error


def capture_inventory(source, max_file_bytes, max_total_bytes):
    names = git_files(source)
    selected = {".", ".git"}
    missing = []
    for name in names:
        relative = safe_relative(name)
        if relative.parts[0] in GENERATED_ROOTS:
            continue
        path = source / name
        if not path.exists() and not path.is_symlink():
            missing.append(name)  # Keep staged-but-deleted paths deleted on restore.
            continue
        if path.is_dir() and not path.is_symlink():
            raise BackupError(f"nested Git repositories/submodules are unsupported: {name}")
        selected.add(name)
        selected.update(str(parent) for parent in relative.parents)
    for directory, subdirs, files in os.walk(source / ".git", followlinks=False):
        for name in subdirs + files:
            path = Path(directory) / name
            if name.endswith(".lock"):
                raise SourceChanged("Git metadata is locked by another writer")
            selected.add(path.relative_to(source).as_posix())
    if len(selected) > 100_000:
        raise BackupError("source inventory exceeds the backup entry limit")
    records = []
    total = 0
    for name in sorted(selected):
        record = inventory_record(source, name, max_file_bytes)
        total += record.get("size", 0)
        if total > max_total_bytes:
            raise BackupError("source inventory exceeds the total backup size limit")
        records.append(record)
    payload = {"root_name": source.name, "files": records, "missing_tracked": sorted(missing)}
    payload["content_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def write_archive(path, source, inventory):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(mode="w|", fileobj=compressed, format=tarfile.PAX_FORMAT) as archive:
                for record in inventory["files"]:
                    relative = record["path"]
                    name = source.name if relative == "." else source.name + "/" + relative
                    member = tarfile.TarInfo(name)
                    member.mode = record["mode"]
                    member.mtime = 0
                    if record["kind"] == "directory":
                        member.type = tarfile.DIRTYPE
                        archive.addfile(member)
                    elif record["kind"] == "symlink":
                        member.type = tarfile.SYMTYPE
                        member.linkname = record["target"]
                        archive.addfile(member)
                    else:
                        member.size = record["size"]
                        descriptor = os.open(source / relative, os.O_RDONLY | os.O_NOFOLLOW)
                        with os.fdopen(descriptor, "rb") as stream:
                            archive.addfile(member, stream)
        raw.flush()
        os.fsync(raw.fileno())


def verify_archive(path, manifest):
    """Read every member without extraction, checking modes, types and SHA-256."""
    root = manifest.get("root_name", Path(manifest["source"]).name)
    expected = {}
    for record in manifest["files"]:
        relative = record["path"]
        safe_relative(relative)
        name = root if relative == "." else root + "/" + relative
        if name in expected:
            raise BackupError("duplicate manifest member")
        expected[name] = record
    seen = set()
    legacy = ("version" not in manifest and manifest.get("includes_git_metadata") is True
              and manifest.get("scope") == LEGACY_SCOPE)
    legacy_payload = {key: value for key, value in manifest.items() if key not in LEGACY_PUBLICATION_KEYS}
    if legacy and LEGACY_PUBLICATION_KEYS.intersection(manifest):
        if (not LEGACY_PUBLICATION_KEYS.issubset(manifest) or manifest["verified"] is not True
                or Path(path).stat().st_size != manifest["archive_bytes"]
                or sha_file(path) != manifest["archive_sha256"]):
            raise BackupError("invalid legacy archive publication metadata")
    legacy_manifest_seen = False
    with gzip.open(path, "rb") as compressed:
        # Use the seekable gzip interface so tar's private streaming read-ahead
        # cannot hide unconsumed bytes after the end-of-archive marker.
        with tarfile.open(mode="r:", fileobj=compressed) as archive:
            for member in archive:
                name = member.name.rstrip("/")
                if name == "BACKUP_MANIFEST.json" and legacy:
                    maximum = min(8 * MIB, max(4096, 4 * len(canonical(manifest))))
                    if legacy_manifest_seen or not member.isfile() or member.size > maximum:
                        raise BackupError("invalid embedded legacy backup manifest")
                    with archive.extractfile(member) as stream:
                        embedded = json.loads(stream.read(maximum + 1))
                    if embedded != legacy_payload:
                        raise BackupError("embedded legacy backup manifest differs from its sidecar")
                    legacy_manifest_seen = True
                    continue
                if name not in expected or name in seen:
                    raise BackupError("archive inventory differs from its manifest")
                seen.add(name)
                record = expected[name]
                if member.mode != record["mode"]:
                    raise BackupError("archive file mode differs from its manifest")
                if record["kind"] == "directory":
                    valid = member.isdir()
                elif record["kind"] == "symlink":
                    target = record["target"]
                    valid = (member.issym() and member.linkname == target
                             and hashlib.sha256(os.fsencode(target)).hexdigest() == record["sha256"])
                else:
                    valid = member.isfile() and member.size == record["size"]
                    if valid:
                        digest = hashlib.sha256()
                        with archive.extractfile(member) as stream:
                            while data := stream.read(MIB):
                                digest.update(data)
                        valid = digest.hexdigest() == record["sha256"]
                if not valid:
                    raise BackupError(f"archive verification failed: {name}")
        # Reach gzip EOF, including concatenated gzip members. Only bounded
        # zero tar padding is valid after the end marker; no payload is skipped.
        padding = 0
        while tail := compressed.read(64 * 1024):
            padding += len(tail)
            if any(tail) or padding > MIB:
                raise BackupError("unexpected data after the tar archive")
    if seen != set(expected):
        raise BackupError("archive is missing manifest members")
    if legacy and not legacy_manifest_seen:
        raise BackupError("archive is missing its embedded legacy manifest")


def snapshot_file(destination, value):
    path = Path(value)
    if not path.is_absolute():
        path = destination / path
    path = path.resolve(strict=True)
    if path.parent != destination or not path.is_file():
        raise BackupError("LATEST points outside the backup directory")
    return path


def verified_latest(destination):
    latest_path = destination / "LATEST.json"
    if not latest_path.exists():
        return None, None
    latest = json.loads(latest_path.read_text())
    archive = snapshot_file(destination, latest["archive"])
    sidecar = snapshot_file(destination, latest["manifest"])
    if (latest.get("verified") is not True or archive.stat().st_size != latest["archive_bytes"]
            or sha_file(archive) != latest["archive_sha256"]
            or (latest.get("manifest_sha256") and sha_file(sidecar) != latest["manifest_sha256"])):
        raise BackupError("existing LATEST snapshot is not verifiable; it was left untouched")
    manifest = json.loads(sidecar.read_text())
    if "version" not in manifest and LEGACY_PUBLICATION_KEYS.intersection(manifest):
        if any(manifest.get(key) != latest.get(key) for key in LEGACY_PUBLICATION_KEYS):
            raise BackupError("legacy sidecar publication metadata differs from LATEST")
    verify_archive(archive, manifest)
    return latest, manifest


@contextmanager
def backup_lock(destination):
    descriptor = os.open(destination / ".backup.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise BackupError("another source backup is already running") from error
        yield
    finally:
        os.close(descriptor)


def backup(source, destination, *, allow_same_filesystem=False, retries=3,
           max_file_bytes=64 * MIB, max_total_bytes=512 * MIB):
    source = Path(source).resolve(strict=True)
    if not source.is_dir() or not (source / ".git").is_dir() or (source / ".git").is_symlink():
        raise BackupError("source must have its own .git directory; linked worktrees are unsupported")
    for relative in (".git/objects", ".git/refs", ".git/index"):
        if (source / relative).is_symlink():
            raise BackupError("external Git object/ref/index links cannot be backed up safely")
    alternates = source / ".git/objects/info/alternates"
    if alternates.exists() and alternates.stat().st_size:
        raise BackupError("external Git object stores cannot be backed up safely")
    destination = Path(destination).expanduser().resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise BackupError("source and backup directories must not contain one another")
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = destination.resolve(strict=True)
    if not allow_same_filesystem and source.stat().st_dev == destination.stat().st_dev:
        raise BackupError("backup destination must be on a different filesystem from the source")
    with backup_lock(destination):
        latest, previous_manifest = verified_latest(destination)
        for attempt in range(retries):
            pending = None
            try:
                inventory = capture_inventory(source, max_file_bytes, max_total_bytes)
                if previous_manifest and previous_manifest.get("content_sha256") == inventory["content_sha256"]:
                    if capture_inventory(source, max_file_bytes, max_total_bytes) != inventory:
                        raise SourceChanged("source changed while checking the existing snapshot")
                    return {"status": "unchanged", **latest}
                descriptor, pending = tempfile.mkstemp(prefix=".pending-archive-", dir=destination)
                os.close(descriptor)
                write_archive(pending, source, inventory)
                if capture_inventory(source, max_file_bytes, max_total_bytes) != inventory:
                    raise SourceChanged("source changed while creating the archive")
                created = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
                manifest = {"version": 1, "created_utc": created, "source": str(source),
                            "source_device": source.stat().st_dev, "backup_device": destination.stat().st_dev,
                            **inventory}
                verify_archive(pending, manifest)
                archive_hash = sha_file(pending)
                stem = source.name + "-" + created + "-" + uuid.uuid4().hex[:8]
                archive_path = destination / (stem + ".tar.gz")
                manifest_path = destination / (stem + ".manifest.json")
                # Link to a unique final name without ever replacing an old archive.
                os.link(pending, archive_path)
                os.unlink(pending)
                pending = None
                atomic_json(manifest_path, manifest, replace=False)
                sync_directory(destination)
                record = {"created_utc": created, "source": str(source), "archive": str(archive_path),
                          "archive_sha256": archive_hash, "archive_bytes": archive_path.stat().st_size,
                          "manifest": str(manifest_path), "manifest_sha256": sha_file(manifest_path),
                          "file_count": sum(item["kind"] != "directory" for item in inventory["files"]),
                          "entry_count": len(inventory["files"]), "content_sha256": inventory["content_sha256"],
                          "verified": True}
                atomic_json(destination / "LATEST.json", record)
                return {"status": "created", **record}
            except SourceChanged:
                if attempt + 1 == retries:
                    raise
                time.sleep(0.2)
            finally:
                if pending is not None and os.path.exists(pending):
                    os.unlink(pending)
    raise BackupError("no snapshot was created")


def positive_int(value):
    number = int(value)
    if not 1 <= number <= 1024:
        raise argparse.ArgumentTypeError("value must be between 1 and 1024")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, default=Path.home() / "pcfinder-source-backups/zxcl-pc")
    parser.add_argument("--retries", type=positive_int, default=3)
    parser.add_argument("--max-file-mib", type=positive_int, default=64)
    parser.add_argument("--max-total-mib", type=positive_int, default=512)
    parser.add_argument("--allow-same-filesystem-for-tests", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        result = backup(args.source, args.destination, retries=args.retries,
                        allow_same_filesystem=args.allow_same_filesystem_for_tests,
                        max_file_bytes=args.max_file_mib * MIB, max_total_bytes=args.max_total_mib * MIB)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, EOFError, BackupError, ValueError, KeyError, tarfile.TarError) as error:
        print(json.dumps({"status": "failed", "error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
