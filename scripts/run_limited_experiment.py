#!/usr/bin/env python3
"""Run a trusted experiment with aggregate cgroup limits, failing closed by default.

Defaults: 32 GiB charged cgroup memory, 4 CPUs, 32 GiB monitored disk, unlimited time.
Shared file mappings can have resident pages charged to a different cgroup. A
supplemental 0.05-second monitor therefore sums process RSS in this service and
its subgroups, conservatively counting shared pages once per process. It stops
at 28 GiB RSS for the default 32 GiB budget. This RSS guard is monitored, can
overshoot between samples, and is not a hard kernel RSS limit.
The command must direct every output into --output-dir or an explicit --write-root.
TMPDIR, CARGO_TARGET_DIR and XDG_CACHE_HOME default below the fresh output directory.
This is resource control, not a filesystem/security sandbox. Disk usage is monitored
every 0.5 seconds, counts sparse files conservatively, and can briefly overshoot;
it is NOT a filesystem quota. The disk monitor stops below the requested budget,
reserving max(1 MiB, min(1 GiB, budget / 32)) as headroom. Existing files under
--write-root count in full.

--allow-process-limits explicitly permits a weaker fallback when user systemd is
unavailable: address-space/file-size limits are per process, not aggregate memory
or disk caps. Do not use that fallback for large experiments.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

GIB = 1 << 30
POLL_SECONDS = 0.05
DISK_POLL_SECONDS = 0.5
MAX_SCAN_ENTRIES = 200_000
CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_ROOT = Path("/proc")
PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def disk_stop_threshold(budget):
    headroom = max(1 << 20, min(GIB, budget // 32))
    return max(0, min(budget, budget - headroom))


def rss_stop_threshold(budget):
    return max(0, budget - max(1 << 20, min(4 * GIB, budget // 8)))


def size_gib(value):
    number = float(value)
    if not math.isfinite(number) or not 1 / 1024 <= number <= 1024:
        raise argparse.ArgumentTypeError("size must be finite and between 1 MiB and 1024 GiB")
    return math.ceil(number * GIB)


def bounded_int(value, maximum):
    number = int(value)
    if not 1 <= number <= maximum:
        raise argparse.ArgumentTypeError(f"must be between 1 and {maximum}")
    return number


def seconds_limit(value):
    number = int(value)
    if not 0 <= number <= 86400:
        raise argparse.ArgumentTypeError("seconds must be between 0 (unlimited) and 86400")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--mem-gib", type=size_gib, default=32 * GIB)
    result.add_argument("--cpus", type=lambda s: bounded_int(s, 256), default=4)
    result.add_argument("--seconds", type=seconds_limit, default=0, help="wall time cap; 0 means unlimited (default)")
    result.add_argument("--disk-gib", type=size_gib, default=32 * GIB)
    result.add_argument("--output-dir", type=Path, required=True, help="new directory owned by this run")
    result.add_argument("--write-root", type=Path, action="append", default=[], help="additional output directory; all existing contents count")
    result.add_argument("--cwd", type=Path, default=Path(__file__).resolve().parents[1])
    result.add_argument("--tag", default="run")
    result.add_argument("--allow-process-limits", action="store_true")
    result.add_argument("command", nargs=argparse.REMAINDER)
    return result


def nonoverlapping_roots(paths):
    roots = sorted({path.resolve() for path in paths}, key=lambda p: (len(p.parts), str(p)))
    return [path for i, path in enumerate(roots) if not any(parent in path.parents for parent in roots[:i])]


def disk_bytes(roots):
    """No symlink traversal; count every name, conservatively including hard links."""
    total = entries = 0
    pending = list(roots)
    while pending:
        path = pending.pop()
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        entries += 1
        if entries > MAX_SCAN_ENTRIES:
            raise RuntimeError("disk monitor exceeded its bounded directory-entry scan")
        total += max(info.st_size, info.st_blocks * 512)
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(path) as iterator:
                    for entry in iterator:
                        pending.append(Path(entry.path))
                        if entries + len(pending) > MAX_SCAN_ENTRIES:
                            raise RuntimeError("disk monitor exceeded its bounded directory-entry scan")
            except FileNotFoundError:
                continue
    return total


def write_json(path, value):
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    temporary.replace(path)


def worker_environment(config):
    # No dump or wholesale transfer of the caller's potentially private environment.
    output = Path(config["output_dir"])
    env = {"PATH": os.environ.get("PATH", os.defpath), "HOME": str(Path.home()),
           "LANG": "C.UTF-8", "RAYON_NUM_THREADS": str(len(config["cpu_ids"])),
           "CARGO_BUILD_JOBS": str(len(config["cpu_ids"])), "PYTHONDONTWRITEBYTECODE": "1"}
    for key, directory in (("TMPDIR", "tmp"), ("CARGO_TARGET_DIR", "target"), ("XDG_CACHE_HOME", "cache")):
        path = output / directory
        path.mkdir(exist_ok=True)
        env[key] = str(path)
    return env


def cgroup_path():
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = Path(line[3:].lstrip("/"))
            if ".." in relative.parts:
                break
            return CGROUP_ROOT / relative
    raise RuntimeError("cgroup v2 membership unavailable")


def validated_group_path(value, unit):
    group = Path(value)
    relative = group.relative_to(CGROUP_ROOT)
    if not relative.parts or ".." in relative.parts or group.name != unit:
        raise RuntimeError("worker reported an unexpected service cgroup")
    return group


def read_worker_group(output, unit):
    try:
        ready = json.loads((output / "worker-ready.json").read_text())
    except FileNotFoundError:
        return None
    if ready.get("unit") != unit:
        raise RuntimeError("worker readiness unit does not match this run")
    return validated_group_path(ready["cgroup"], unit)


def cgroup_process_ids(group):
    """cgroup.procs contains process IDs, avoiding per-thread RSS double counts."""
    result = set()
    pending = [group]
    groups = 0
    while pending:
        directory = pending.pop()
        groups += 1
        if groups > 1024:
            raise RuntimeError("RSS monitor exceeded its bounded cgroup scan")
        try:
            result.update(int(value) for value in (directory / "cgroup.procs").read_text().split())
            with os.scandir(directory) as entries:
                pending.extend(Path(entry.path) for entry in entries if entry.is_dir(follow_symlinks=False))
        except FileNotFoundError:
            continue  # An exited service/subgroup may disappear during a sample.
    return result


def status_rss(status):
    rss = next((line.split() for line in status.splitlines() if line.startswith("VmRSS:")), None)
    if rss is not None:
        if len(rss) != 3 or rss[2] != "kB" or int(rss[1]) < 0:
            raise RuntimeError("malformed process RSS counter")
        return int(rss[1]) * 1024
    if any(line.startswith("State:") and line.split()[1:2] in (["Z"], ["X"], ["x"])
           for line in status.splitlines()):
        return 0
    return None


def process_identity_rss(directory):
    raw = (directory / "stat").read_text()
    fields = raw[raw.rfind(")") + 2:].split()
    if len(fields) < 22 or int(fields[21]) < 0:
        raise RuntimeError("malformed process identity/RSS counter")
    return int(fields[19]), fields[0], int(fields[21]) * PAGE_SIZE


def process_rss_bytes(pid, proc_root):
    directory = proc_root / str(pid)
    try:
        status = (directory / "status").read_text()
        rss = status_rss(status)
        if rss is not None:
            return rss
        # exec/exit can temporarily expose a live State with no VmRSS. Retry,
        # then use statm only when the PID's start time is stable across reads.
        for _attempt in range(2):
            before = process_identity_rss(directory)
            rss = status_rss((directory / "status").read_text())
            if rss is not None:
                return rss
            values = (directory / "statm").read_text().split()
            if len(values) < 2 or int(values[1]) < 0:
                raise RuntimeError("malformed process statm RSS counter")
            after = process_identity_rss(directory)
            if before[0] == after[0]:
                if after[1] in ("Z", "X", "x"):
                    return 0
                # A valid zero statm counter is legitimate while exec/exit has
                # no installed mm; unreadable counters still fail closed.
                return max(int(values[1]) * PAGE_SIZE, before[2], after[2])
        raise RuntimeError("process identity changed repeatedly during RSS sampling")
    except (FileNotFoundError, ProcessLookupError):
        try:
            remaining = status_rss((directory / "status").read_text())
        except (FileNotFoundError, ProcessLookupError):
            return 0
        if remaining is not None:
            return remaining
        raise RuntimeError("live experiment process has no RSS counter")


def cgroup_rss_bytes(group, proc_root=PROC_ROOT):
    return sum(process_rss_bytes(pid, proc_root) for pid in cgroup_process_ids(group))


def pid_in_group(pid, group):
    relative = "/" + str(group.relative_to(CGROUP_ROOT))
    try:
        return any(line.startswith("0::") and (line[3:] == relative or line[3:].startswith(relative + "/"))
                   for line in (PROC_ROOT / str(pid) / "cgroup").read_text().splitlines())
    except (FileNotFoundError, ProcessLookupError):
        return False


def kill_cgroup_processes(group, exclude_pid=None):
    if exclude_pid is None:
        try:
            (group / "cgroup.kill").write_text("1")
            return
        except OSError:
            pass
    for pid in cgroup_process_ids(group):
        if pid == exclude_pid:
            continue
        descriptor = None
        try:
            if hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"):
                descriptor = os.pidfd_open(pid)
            if pid_in_group(pid, group):  # Recheck membership before signaling a possibly reused PID.
                if descriptor is not None:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def verify_cgroup(config, group):
    def value(name):
        return (group / name).read_text().strip()
    maximum = value("memory.max")
    if maximum == "max" or int(maximum) > config["memory_bytes"]:
        raise RuntimeError("aggregate memory.max was not applied")
    if value("memory.swap.max") != "0" or value("memory.oom.group") != "1":
        raise RuntimeError("swap/whole-group OOM limits were not applied")
    if not set(os.sched_getaffinity(0)).issubset(config["cpu_ids"]):
        raise RuntimeError("CPU affinity was not applied")
    try:
        quota, period = value("cpu.max").split()
    except FileNotFoundError:
        # Some user managers cannot delegate the CPU controller. The service's
        # CPUAffinity still bounds this trusted command and its inheriting tasks.
        cpu_mode = "affinity; CPU quota controller unavailable"
    else:
        if quota == "max" or int(quota) > len(config["cpu_ids"]) * int(period):
            raise RuntimeError("aggregate CPU quota was not applied")
        cpu_mode = "quota and affinity"
    tasks = value("pids.max")
    if tasks == "max" or int(tasks) > 128:
        raise RuntimeError("aggregate task limit was not applied")
    return cpu_mode


def read_cgroup_stats(group):
    result = {}
    for filename in ("memory.peak", "memory.current", "memory.events", "cpu.stat", "pids.peak"):
        try:
            raw = (group / filename).read_text().strip()
            result[filename] = int(raw) if raw.isdigit() else dict(line.split() for line in raw.splitlines())
        except (OSError, ValueError):
            pass
    return result


def set_process_limit(kind, amount):
    _soft, hard = resource.getrlimit(kind)
    cap = amount if hard == resource.RLIM_INFINITY else min(amount, hard)
    resource.setrlimit(kind, (cap, cap))


def worker(config_path, fallback):
    config = json.loads(Path(config_path).read_text())
    result_path = Path(config["output_dir"]) / "worker-result.json"
    result = {"status": "setup-failed", "returncode": 125}
    group = None
    child = None
    rss_peak = 0
    started = time.monotonic()
    try:
        if not fallback:
            group = cgroup_path()
            result["cpu_enforcement"] = verify_cgroup(config, group)  # Gate before the command.
            write_json(Path(config["output_dir"]) / "worker-ready.json",
                       {"unit": config["unit"], "cgroup": str(group), "pid": os.getpid()})
        os.sched_setaffinity(0, config["cpu_ids"])
        set_process_limit(resource.RLIMIT_FSIZE, config["disk_bytes"])
        set_process_limit(resource.RLIMIT_CORE, 0)
        set_process_limit(resource.RLIMIT_NOFILE, 4096)
        if fallback:
            set_process_limit(resource.RLIMIT_AS, config["memory_bytes"])
            if config["seconds"]:
                set_process_limit(resource.RLIMIT_CPU, config["seconds"] * len(config["cpu_ids"]))
        child = subprocess.Popen(config["command"], cwd=config["cwd"], env=worker_environment(config))
        rss_stop = config.get("rss_stop_bytes", rss_stop_threshold(config["memory_bytes"]))
        while child.poll() is None:
            if group is not None:
                rss_peak = max(rss_peak, cgroup_rss_bytes(group))
                if rss_peak >= rss_stop:
                    result.update(status="rss-limit", returncode=124, rss_peak_bytes=rss_peak)
                    write_json(result_path, result)
                    kill_cgroup_processes(group, exclude_pid=os.getpid())
                    child.wait(timeout=3)
                    break
            time.sleep(POLL_SECONDS)
        else:
            code = child.returncode if child.returncode >= 0 else 128 - child.returncode
            result.update(status="completed" if code == 0 else "command-failed", returncode=code)
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        if child is not None and child.poll() is None:
            try:
                if group is not None:
                    kill_cgroup_processes(group, exclude_pid=os.getpid())
                else:
                    child.kill()
                child.wait(timeout=3)
            except Exception as error:
                result.update(status="termination-failed", returncode=125, termination_error=str(error))
        result["elapsed_seconds"] = time.monotonic() - started
        result["rss_peak_bytes"] = rss_peak
        if group is not None:
            result["cgroup"] = read_cgroup_stats(group)
        write_json(result_path, result)
    return result["returncode"]


def service_command(config, config_path):
    properties = {
        "MemoryMax": config["memory_bytes"], "MemorySwapMax": 0,
        "OOMPolicy": "kill", "MemoryAccounting": "yes", "CPUAccounting": "yes",
        "CPUQuota": f"{len(config['cpu_ids']) * 100}%", "AllowedCPUs": ",".join(map(str, config["cpu_ids"])),
        "CPUAffinity": " ".join(map(str, config["cpu_ids"])),
        "TasksMax": 128, "TimeoutStopSec": 3,
        "KillMode": "control-group", "WorkingDirectory": config["cwd"],
        "LimitFSIZE": config["disk_bytes"],
        "LimitCORE": 0, "LimitNOFILE": 4096,
    }
    if config["seconds"]:
        properties["RuntimeMaxSec"] = config["seconds"]
    command = ["systemd-run", "--user", "--wait", "--pipe", "--collect", "--quiet", "--unit=" + config["unit"]]
    command += [f"--property={key}={value}" for key, value in properties.items()]
    command += ["--setenv=PATH=" + os.environ.get("PATH", os.defpath), "--setenv=HOME=" + str(Path.home())]
    return command + [sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path)]


def user_systemd_available():
    if not shutil.which("systemd-run") or not shutil.which("systemctl"):
        return False
    try:
        return subprocess.run(["systemctl", "--user", "show-environment"], stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _stop_experiment(process, unit, fallback, group=None):
    """Do not equate a dead systemd-run launcher with a terminated experiment."""
    errors = []
    if not fallback and group is None:
        try:
            found = subprocess.run(["systemctl", "--user", "show", "--property=ControlGroup", "--value", unit],
                                   capture_output=True, text=True, timeout=3)
            if found.returncode == 0 and found.stdout.strip():
                group = validated_group_path(str(CGROUP_ROOT) + found.stdout.strip(), unit)
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    if fallback:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    else:
        if group is not None:
            try:
                kill_cgroup_processes(group)
            except OSError as error:
                errors.append(str(error))
        try:
            killed = subprocess.run(["systemctl", "--user", "kill", "--kill-who=all", "--signal=KILL", unit],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)
            stopped = subprocess.run(["systemctl", "--user", "stop", unit], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)
            if killed.returncode and stopped.returncode:
                errors.append("systemctl kill and stop both failed")
        except (OSError, subprocess.TimeoutExpired) as error:
            errors.append(str(error))
    try:
        code = process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        process.kill()
        code = process.wait(timeout=3)
    deadline = time.monotonic() + 3
    if group is not None:
        while cgroup_process_ids(group) and time.monotonic() < deadline:
            kill_cgroup_processes(group)
            time.sleep(POLL_SECONDS)
        confirmed = not cgroup_process_ids(group)
    elif fallback:
        try:
            os.killpg(process.pid, 0)
            confirmed = False
        except ProcessLookupError:
            confirmed = True
    else:
        try:
            active = subprocess.run(["systemctl", "--user", "is-active", unit], capture_output=True, text=True, timeout=3)
            confirmed = active.stdout.strip() in ("inactive", "failed", "unknown") and active.returncode in (0, 3, 4)
        except (OSError, subprocess.TimeoutExpired):
            confirmed = False
    return {"confirmed": confirmed, "launcher_returncode": code, "errors": errors}


def stop_experiment(process, unit, fallback, group=None):
    # A repeated interrupt must not abort cleanup while the service is still live.
    previous = {kind: signal.signal(kind, signal.SIG_IGN) for kind in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        return _stop_experiment(process, unit, fallback, group)
    finally:
        for kind, handler in previous.items():
            signal.signal(kind, handler)


def terminate(summary, process, unit, fallback, group):
    try:
        summary["termination"] = stop_experiment(process, unit, fallback, group)
    except Exception as error:
        summary["termination"] = {"confirmed": False, "error": str(error)}
    if not summary["termination"]["confirmed"]:
        summary.update(status="termination-failed", returncode=125)


def run(args):
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("a command is required after --")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", args.tag):
        raise ValueError("tag must contain 1–40 letters, digits, underscores or hyphens")
    cpu_ids = sorted(os.sched_getaffinity(0))[:args.cpus]
    if len(cpu_ids) != args.cpus:
        raise ValueError("requested CPUs exceed the current allowed affinity")
    cwd = args.cwd.resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("working directory must be a directory")
    output = args.output_dir.absolute()
    if output.exists() or output.is_symlink():
        raise ValueError("output directory must not already exist")
    roots = [path.resolve(strict=True) for path in args.write_root]
    if any(not path.is_dir() for path in roots):
        raise ValueError("additional write roots must be existing directories")
    fallback = not user_systemd_available()
    if fallback and not args.allow_process_limits:
        raise RuntimeError("user systemd is unavailable; refusing to run without aggregate cgroup limits")
    # The disk monitor and launcher share the same CPUs as the service, rather
    # than consuming a fifth CPU outside the experimental command's affinity.
    os.sched_setaffinity(0, cpu_ids)
    output.mkdir(mode=0o700, parents=True)
    output = output.resolve()
    roots = nonoverlapping_roots([output, *roots])
    disk_stop = disk_stop_threshold(args.disk_gib)
    rss_stop = rss_stop_threshold(args.mem_gib)
    config = {"memory_bytes": args.mem_gib, "cpu_ids": cpu_ids, "seconds": args.seconds,
              "disk_bytes": args.disk_gib, "disk_stop_bytes": disk_stop, "rss_stop_bytes": rss_stop,
              "output_dir": str(output), "cwd": str(cwd),
              "command": command, "unit": f"zxcl-experiment-{args.tag}-{uuid.uuid4().hex[:12]}.service"}
    config_path = output / "run-config.json"
    write_json(config_path, config)
    summary = {"status": "preflight-failed", "returncode": 125, "limits": {key: config[key] for key in
               ("memory_bytes", "rss_stop_bytes", "cpu_ids", "seconds", "disk_bytes", "disk_stop_bytes")}, "unit": config["unit"],
               "enforcement": "process-only-fallback" if fallback else "aggregate-cgroup-v2",
               "rss_enforcement": "unavailable in process-only fallback" if fallback else "summed process RSS monitored every 0.05 s; can overshoot; shared pages counted per process",
               "disk_enforcement": "monitored; not a filesystem quota", "write_roots": list(map(str, roots))}
    started = time.monotonic()
    process = None
    disk_peak = 0
    rss_peak = 0
    group = None
    try:
        disk_peak = disk_bytes(roots)
        if disk_peak >= disk_stop:
            raise RuntimeError("write roots already reach the conservative disk stop threshold")
        invocation = ([sys.executable, str(Path(__file__).resolve()), "--worker", str(config_path), "--fallback"]
                      if fallback else service_command(config, config_path))
        with (output / "output.log").open("wb") as log:
            process = subprocess.Popen(invocation, cwd=cwd, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            next_disk_sample = 0
            while process.poll() is None:
                if not fallback:
                    group = group or read_worker_group(output, config["unit"])
                    if group is not None:
                        rss_peak = max(rss_peak, cgroup_rss_bytes(group))
                        if rss_peak >= rss_stop:
                            summary.update(status="rss-limit", returncode=124)
                            terminate(summary, process, config["unit"], fallback, group)
                            break
                    elif time.monotonic() - started >= 5 and not (output / "worker-result.json").exists():
                        raise RuntimeError("worker did not publish its verified cgroup before the startup deadline")
                if time.monotonic() >= next_disk_sample:
                    disk_peak = max(disk_peak, disk_bytes(roots))
                    next_disk_sample = time.monotonic() + DISK_POLL_SECONDS
                if disk_peak >= disk_stop:
                    summary.update(status="disk-limit", returncode=124)
                    terminate(summary, process, config["unit"], fallback, group)
                    break
                if args.seconds and time.monotonic() - started >= args.seconds:
                    summary.update(status="time-limit", returncode=124)
                    terminate(summary, process, config["unit"], fallback, group)
                    break
                time.sleep(POLL_SECONDS)
            else:
                worker_path = output / "worker-result.json"
                if worker_path.exists():
                    summary.update(json.loads(worker_path.read_text()))
                else:
                    summary.update(status="service-failed", returncode=process.returncode or 125)
                if group is not None and cgroup_process_ids(group):
                    summary.update(status="lingering-processes", returncode=125)
                    terminate(summary, process, config["unit"], fallback, group)
        disk_peak = max(disk_peak, disk_bytes(roots))
        if disk_peak >= disk_stop and summary["status"] == "completed":
            summary.update(status="disk-limit", returncode=124)
    except KeyboardInterrupt:
        summary.update(status="interrupted", returncode=130)
        if process is not None:
            terminate(summary, process, config["unit"], fallback, group)
    except Exception as error:
        summary.update(status="runner-failed", returncode=125, error=f"{type(error).__name__}: {error}")
        if process is not None:
            terminate(summary, process, config["unit"], fallback, group)
    finally:
        summary.update(elapsed_seconds=time.monotonic() - started, disk_peak_bytes=disk_peak,
                       rss_peak_bytes=max(rss_peak, summary.get("rss_peak_bytes", 0)))
        write_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True))
    return summary["returncode"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--worker"]:
        return worker(argv[1], argv[2:] == ["--fallback"])
    def interrupted(_signum, _frame):
        raise KeyboardInterrupt
    for kind in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(kind, interrupted)
    try:
        return run(parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"limited experiment refused: {error}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
