"""Small Git fixtures; all snapshots remain under the caller's temporary directory."""
import fcntl
import gzip
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("source_backup", ROOT / "scripts/backup_sources.py")
BACKUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKUP)


class SourceBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zxcl-source-backup-tests-")
        self.path = Path(self.temporary.name)
        self.source = self.path / "repo"
        self.destination = self.path / "backups"
        self.source.mkdir()
        self.git("init", "-b", "main")
        (self.source / ".gitignore").write_text("/target/\n/work/\n/data/\n*.pyc\n")
        (self.source / "Cargo.lock").write_text("version = 3\n")
        (self.source / "source.py").write_text("print('staged')\n")
        (self.source / "source.py").chmod(0o751)
        self.git("add", ".gitignore", "Cargo.lock", "source.py")
        (self.source / "source.py").write_text("print('unstaged')\n")
        (self.source / "new.py").write_text("print('untracked')\n")
        (self.source / "alias").symlink_to("missing-target")
        (self.source / "target").mkdir()
        (self.source / "target/generated").write_bytes(b"ignored generated output")

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.source), *args], check=True, capture_output=True,
                              env=dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull))

    def create(self, **options):
        return BACKUP.backup(self.source, self.destination, allow_same_filesystem=True, **options)

    def latest_bytes(self):
        return (self.destination / "LATEST.json").read_bytes()

    def test_snapshot_preserves_unborn_index_unstaged_untracked_links_and_modes(self):
        index = (self.source / ".git/index").read_bytes()
        result = self.create()
        self.assertEqual(result["status"], "created")
        self.assertTrue(result["verified"])
        latest, manifest = BACKUP.verified_latest(self.destination)
        self.assertEqual(latest["archive_sha256"], result["archive_sha256"])
        with tarfile.open(result["archive"], "r:gz") as archive:
            self.assertEqual(archive.extractfile("repo/.git/index").read(), index)
            self.assertEqual(archive.extractfile("repo/source.py").read(), b"print('unstaged')\n")
            self.assertEqual(archive.extractfile("repo/new.py").read(), b"print('untracked')\n")
            self.assertEqual(archive.extractfile("repo/Cargo.lock").read(), b"version = 3\n")
            self.assertEqual(archive.getmember("repo/source.py").mode, 0o751)
            self.assertEqual(archive.getmember("repo/alias").linkname, "missing-target")
            self.assertTrue(archive.getmember("repo/.git/refs").isdir())
            self.assertNotIn("repo/target/generated", archive.getnames())
            restored = self.path / "restored"
            archive.extractall(restored)
        recovered = restored / "repo"
        recovered_stage = subprocess.run(["git", "-C", str(recovered), "ls-files", "--stage", "-z"],
                                         check=True, capture_output=True).stdout
        self.assertEqual(recovered_stage, self.git("ls-files", "--stage", "-z").stdout)
        recovered_lock = subprocess.run(["git", "-C", str(recovered), "show", ":Cargo.lock"],
                                        check=True, capture_output=True).stdout
        self.assertEqual(recovered_lock, b"version = 3\n")
        self.assertEqual((self.source / ".git/index").read_bytes(), index)
        self.assertTrue(any(record["path"] == "new.py" for record in manifest["files"]))

    def test_unchanged_sources_reuse_verified_snapshot_and_keep_latest_bytes(self):
        first = self.create()
        before = self.latest_bytes()
        second = self.create()
        self.assertEqual(second["status"], "unchanged")
        self.assertEqual(first["archive"], second["archive"])
        self.assertEqual(before, self.latest_bytes())
        self.assertEqual(len(list(self.destination.glob("*.tar.gz"))), 1)

    def test_changed_sources_publish_new_snapshot_without_deleting_old(self):
        first = self.create()
        (self.source / "new.py").write_text("print('changed')\n")
        second = self.create()
        self.assertNotEqual(first["archive"], second["archive"])
        self.assertTrue(Path(first["archive"]).is_file())
        self.assertEqual(BACKUP.sha_file(first["archive"]), first["archive_sha256"])
        self.assertEqual(len(list(self.destination.glob("*.tar.gz"))), 2)

    def test_deleted_staged_file_stays_absent_in_snapshot(self):
        (self.source / "source.py").unlink()
        result = self.create()
        manifest = json.loads(Path(result["manifest"]).read_text())
        self.assertIn("source.py", manifest["missing_tracked"])
        with tarfile.open(result["archive"], "r:gz") as archive:
            self.assertNotIn("repo/source.py", archive.getnames())
            self.assertIn("repo/.git/index", archive.getnames())

    def test_source_loss_preserves_last_verified_snapshot(self):
        self.create()
        before = self.latest_bytes()
        self.source.rename(self.path / "source-formatted-away")
        with self.assertRaises(FileNotFoundError):
            self.create()
        self.assertEqual(self.latest_bytes(), before)
        BACKUP.verified_latest(self.destination)

    def test_failed_bounded_retries_preserve_latest_and_remove_only_pending_files(self):
        first = self.create()
        before = self.latest_bytes()
        (self.source / "new.py").write_text("changed\n")
        with mock.patch.object(BACKUP, "write_archive", side_effect=BACKUP.SourceChanged("writer active")) as writer, \
                mock.patch.object(BACKUP.time, "sleep"):
            with self.assertRaises(BACKUP.SourceChanged):
                self.create(retries=2)
        self.assertEqual(writer.call_count, 2)
        self.assertEqual(self.latest_bytes(), before)
        self.assertTrue(Path(first["archive"]).exists())
        self.assertFalse(list(self.destination.glob(".pending-*")))

    def test_archive_corruption_is_detected_without_changing_latest(self):
        first = self.create()
        before = self.latest_bytes()
        with Path(first["archive"]).open("ab") as stream:
            stream.write(b"corrupt")
        with self.assertRaisesRegex(BACKUP.BackupError, "not verifiable"):
            self.create()
        self.assertEqual(self.latest_bytes(), before)
        self.assertTrue(Path(first["archive"]).exists())

    def test_same_filesystem_is_rejected_without_test_override(self):
        with self.assertRaisesRegex(BACKUP.BackupError, "different filesystem"):
            BACKUP.backup(self.source, self.destination)
        self.assertFalse((self.destination / "LATEST.json").exists())

    def test_concurrent_invocation_cannot_replace_snapshot(self):
        self.create()
        before = self.latest_bytes()
        with (self.destination / ".backup.lock").open("rb") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(BACKUP.BackupError, "already running"):
                self.create()
        self.assertEqual(self.latest_bytes(), before)

    def test_size_limit_failure_preserves_latest(self):
        self.create()
        before = self.latest_bytes()
        (self.source / "unexpected-large.bin").write_bytes(b"x" * 4096)
        with self.assertRaisesRegex(BACKUP.BackupError, "size limit"):
            self.create(max_file_bytes=2048)
        self.assertEqual(self.latest_bytes(), before)

    def legacy_archive(self, *, corrupt_embedded=False, extra_member=False):
        records = [BACKUP.inventory_record(self.source, path, BACKUP.MIB)
                   for path in (".git/HEAD", "Cargo.lock")]
        manifest = {"created_utc": "20261029T093759Z", "source": str(self.source),
                    "source_device": 1, "backup_device": 2, "files": records,
                    "includes_git_metadata": True, "scope": BACKUP.LEGACY_SCOPE}
        embedded = dict(manifest)
        if corrupt_embedded:
            embedded["source"] = "/different/source"
        path = self.path / "legacy.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            for record in records:
                archive.add(self.source / record["path"], arcname="repo/" + record["path"], recursive=False)
            payload = json.dumps(embedded, indent=2).encode()
            member = tarfile.TarInfo("BACKUP_MANIFEST.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
            if extra_member:
                member = tarfile.TarInfo("unrecognized-extra")
                archive.addfile(member, io.BytesIO())
        # The real first snapshot added these fields to its sidecar only after
        # serializing the embedded manifest into the archive.
        manifest.update(archive=str(path), archive_sha256=BACKUP.sha_file(path),
                        archive_bytes=path.stat().st_size, verified=True)
        return path, manifest

    def test_first_snapshot_legacy_embedded_manifest_is_verified(self):
        archive, manifest = self.legacy_archive()
        BACKUP.verify_archive(archive, manifest)
        self.destination.mkdir()
        sidecar = self.destination / "first.manifest.json"
        final_archive = self.destination / "first.tar.gz"
        archive.rename(final_archive)
        manifest["archive"] = str(final_archive)
        sidecar.write_text(json.dumps(manifest))
        latest = {"created_utc": manifest["created_utc"], "source": str(self.source),
                  "archive": str(final_archive), "manifest": str(sidecar), "verified": True,
                  "archive_sha256": BACKUP.sha_file(final_archive), "archive_bytes": final_archive.stat().st_size,
                  "file_count": len(manifest["files"])}
        (self.destination / "LATEST.json").write_text(json.dumps(latest))
        accepted, recovered = BACKUP.verified_latest(self.destination)
        self.assertEqual(accepted, latest)
        self.assertEqual(recovered, manifest)
        complete = self.create()
        self.assertEqual(complete["status"], "created")
        self.assertTrue(final_archive.exists())

    def test_legacy_embedded_manifest_corruption_is_rejected(self):
        archive, manifest = self.legacy_archive(corrupt_embedded=True)
        with self.assertRaisesRegex(BACKUP.BackupError, "differs from its sidecar"):
            BACKUP.verify_archive(archive, manifest)

    def test_legacy_manifest_does_not_allow_unrecognized_extra_members(self):
        archive, manifest = self.legacy_archive(extra_member=True)
        with self.assertRaisesRegex(BACKUP.BackupError, "inventory differs"):
            BACKUP.verify_archive(archive, manifest)

    def test_legacy_publication_fields_are_verified_before_projection(self):
        archive, manifest = self.legacy_archive()
        manifest["archive_sha256"] = "0" * 64
        with self.assertRaisesRegex(BACKUP.BackupError, "publication metadata"):
            BACKUP.verify_archive(archive, manifest)

    def test_direct_verification_rejects_a_nonzero_concatenated_gzip_tail(self):
        result = self.create()
        manifest = json.loads(Path(result["manifest"]).read_text())
        with Path(result["archive"]).open("ab") as stream:
            stream.write(gzip.compress(b"x"))
        with self.assertRaisesRegex(BACKUP.BackupError, "unexpected data"):
            BACKUP.verify_archive(result["archive"], manifest)

    def test_direct_verification_allows_bounded_zero_tar_padding(self):
        result = self.create()
        manifest = json.loads(Path(result["manifest"]).read_text())
        with Path(result["archive"]).open("ab") as stream:
            stream.write(gzip.compress(b"\0" * 512))
        BACKUP.verify_archive(result["archive"], manifest)


if __name__ == "__main__":
    unittest.main()
