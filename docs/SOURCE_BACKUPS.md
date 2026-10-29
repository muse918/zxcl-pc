# Source backups on the home disk

`scripts/backup_sources.py` takes verified local snapshots of Git tracked and
nonignored files, together with the complete local `.git` directory. This keeps
the current working files, untracked source, staged index, unborn branch state,
symlinks, file modes and empty Git metadata directories. `target`, `work` and
`data` are excluded. A default 64 MiB per-file limit and 512 MiB total source
limit reject accidental generated-data capture; exceeding a limit fails the
backup rather than silently omitting another source file.

```sh
python3 scripts/backup_sources.py \
  --source /mnt/fast2/muse918/pcfinder/zxcl-pc \
  --destination /home/muse918/pcfinder-source-backups/zxcl-pc
```

The source and destination must be on different filesystems. The same-filesystem
override exists only for small temporary test fixtures. Linked worktrees,
external Git object stores and external object/ref/index links fail explicitly.

Every source file is hashed before and after archive creation. The archive is
read back to check its inventory, types, modes, symlink targets and per-file
SHA-256 values. The gzip trailer is also read and checked. Only then are the
archive, JSON manifest and `LATEST.json` published with file and directory
`fsync`. `LATEST.json` includes the archive and manifest SHA-256 values.
Concurrent runs are locked; unchanged source reuses the existing verified
snapshot. Changing source is retried up to three times. Failed attempts and a
missing source leave the accepted snapshot untouched. Old backups are never
deleted by this tool.

The first verified snapshot's manifest format remains readable. A subsequent
snapshot adds directory records and a content identity for deduplication.

For scheduled backups, copy the script itself to
`/home/muse918/.local/libexec/pcfinder-backup-sources.py`, rather than linking to
the `/mnt` checkout. Install the accompanying service and timer templates under
`~/.config/systemd/user/`, then reload the user manager and enable the timer.
The timer runs every 15 minutes and catches up after a missed scheduled run.
The installed tool and existing archives remain on the home disk if `/mnt` is
formatted. A missing source then produces a failed service without replacing
the last snapshot. The templates limit the small backup job to 256 MiB RAM,
one CPU's affinity and a three-minute runtime.

To inspect the current snapshot, read
`/home/muse918/pcfinder-source-backups/zxcl-pc/LATEST.json`. Verify the stored
archive and manifest hashes before restoring, and extract into a new empty
directory so existing source is not overwritten. The archive contains one
`zxcl-pc/` root directory, including `.git` metadata. Do not extract a backup
over an active checkout.
