"""Cheap tests of resource gates; no heavy command or systemd service is launched."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("experiment_limits", ROOT / "scripts/run_limited_experiment.py")
LIMITS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIMITS)


class ExperimentLimitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="zxcl-limit-tests-")
        self.path = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def config(self):
        return {"memory_bytes": 32 << 30, "cpu_ids": [2, 3, 4, 5], "seconds": 600,
                "disk_bytes": 32 << 30, "output_dir": str(self.path), "cwd": str(ROOT),
                "command": ["/bin/true"], "unit": "zxcl-experiment-test.service"}

    def group(self, **overrides):
        group = self.path / "group"
        group.mkdir(exist_ok=True)
        values = {"memory.max": str(32 << 30), "memory.swap.max": "0", "memory.oom.group": "1",
                  "cpu.max": "400000 100000", "pids.max": "128"}
        values.update(overrides)
        for key, value in values.items():
            (group / key).write_text(value)
        return group

    def test_finite_positive_bounded_sizes(self):
        self.assertEqual(LIMITS.size_gib("0.25"), 1 << 28)
        for bad in ("nan", "inf", "-1", "0", "1025"):
            with self.subTest(value=bad), self.assertRaises(argparse.ArgumentTypeError):
                LIMITS.size_gib(bad)

    def test_disk_stop_threshold_reserves_headroom_below_budget(self):
        self.assertEqual(LIMITS.disk_stop_threshold(32 << 30), 31 << 30)
        self.assertEqual(LIMITS.disk_stop_threshold(1 << 30), (1 << 30) - (32 << 20))
        self.assertEqual(LIMITS.disk_stop_threshold(16 << 20), 15 << 20)
        self.assertEqual(LIMITS.disk_stop_threshold(1 << 20), 0)

    def test_rss_stop_threshold_reserves_shared_mapping_headroom(self):
        self.assertEqual(LIMITS.rss_stop_threshold(32 << 30), 28 << 30)
        self.assertEqual(LIMITS.rss_stop_threshold(256 << 20), 224 << 20)

    def test_rss_sums_exact_cgroup_processes_and_nested_subgroups(self):
        group = self.path / "rss-group"
        child = group / "child"
        child.mkdir(parents=True)
        (group / "cgroup.procs").write_text("11\n")
        (child / "cgroup.procs").write_text("12\n11\n13\n")
        proc = self.path / "proc"
        for pid, rss in ((11, 2048), (12, 4096), (99, 8192)):
            directory = proc / str(pid)
            directory.mkdir(parents=True)
            (directory / "status").write_text(f"State:\tS (sleeping)\nVmRSS:\t{rss} kB\n")
        self.assertEqual(LIMITS.cgroup_rss_bytes(group, proc), 6 << 20)

    def test_rss_missing_counter_fails_closed_except_for_exited_zombie(self):
        group = self.path / "rss-group"
        group.mkdir()
        (group / "cgroup.procs").write_text("11\n")
        proc = self.path / "proc"
        (proc / "11").mkdir(parents=True)
        status = proc / "11/status"
        status.write_text("State:\tZ (zombie)\n")
        self.assertEqual(LIMITS.cgroup_rss_bytes(group, proc), 0)
        status.write_text("State:\tS (sleeping)\n")
        with self.assertRaisesRegex(RuntimeError, "no RSS counter"):
            LIMITS.cgroup_rss_bytes(group, proc)

    def test_rss_exec_transition_uses_identity_checked_statm_counter(self):
        proc = self.path / "proc"
        directory = proc / "11"
        directory.mkdir(parents=True)
        (directory / "status").write_text("State:\tR (running)\n")
        fields = ["R"] + ["0"] * 21
        fields[19] = "1234"  # /proc/stat field 22: PID start time.
        fields[21] = "5"     # /proc/stat field 24: resident pages.
        (directory / "stat").write_text("11 (exec name) " + " ".join(fields))
        (directory / "statm").write_text("100 7 0 0 0 0 0\n")
        self.assertEqual(LIMITS.process_rss_bytes(11, proc), 7 * LIMITS.PAGE_SIZE)
        fields[21] = "0"
        (directory / "stat").write_text("11 (exec name) " + " ".join(fields))
        (directory / "statm").write_text("0 0 0 0 0 0 0\n")
        self.assertEqual(LIMITS.process_rss_bytes(11, proc), 0)

    def test_rss_permission_failure_is_not_treated_as_an_exited_process(self):
        proc = self.path / "proc"
        directory = proc / "11"
        directory.mkdir(parents=True)
        (directory / "status").write_text("State:\tR (running)\n")
        fields = ["R"] + ["0"] * 21
        fields[19] = "1234"
        (directory / "stat").write_text("11 (process) " + " ".join(fields))
        original = Path.read_text

        def read_text(path, *args, **kwargs):
            if path.name == "statm":
                raise PermissionError("unreadable live counter")
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", read_text):
            with self.assertRaises(PermissionError):
                LIMITS.process_rss_bytes(11, proc)

    def test_readiness_only_accepts_this_service_cgroup(self):
        unit = self.config()["unit"]
        ready = self.path / "worker-ready.json"
        ready.write_text(json.dumps({"unit": unit, "cgroup": "/sys/fs/cgroup/user.slice/" + unit}))
        self.assertEqual(LIMITS.read_worker_group(self.path, unit).name, unit)
        ready.write_text(json.dumps({"unit": unit, "cgroup": "/sys/fs/cgroup/other.service"}))
        with self.assertRaisesRegex(RuntimeError, "unexpected"):
            LIMITS.read_worker_group(self.path, unit)

    def test_cgroup_gate_rejects_missing_aggregate_controls(self):
        with mock.patch.object(os, "sched_getaffinity", return_value={2, 3, 4, 5}):
            LIMITS.verify_cgroup(self.config(), self.group())
            for override in ({"memory.max": "max"}, {"memory.max": str(64 << 30)},
                             {"memory.swap.max": "1"}, {"memory.oom.group": "0"},
                             {"cpu.max": "max 100000"}, {"cpu.max": "500000 100000"},
                             {"pids.max": "max"}, {"pids.max": "129"}):
                with self.subTest(override=override), self.assertRaises(RuntimeError):
                    LIMITS.verify_cgroup(self.config(), self.group(**override))

    def test_wrong_affinity_is_rejected(self):
        with mock.patch.object(os, "sched_getaffinity", return_value={2, 3, 4, 5, 6}):
            with self.assertRaisesRegex(RuntimeError, "affinity"):
                LIMITS.verify_cgroup(self.config(), self.group())

    def test_missing_cpu_controller_requires_service_affinity(self):
        group = self.group()
        (group / "cpu.max").unlink()
        with mock.patch.object(os, "sched_getaffinity", return_value={2, 3, 4, 5}):
            self.assertIn("unavailable", LIMITS.verify_cgroup(self.config(), group))
        with mock.patch.object(os, "sched_getaffinity", return_value={2, 3, 4, 5, 6}):
            with self.assertRaisesRegex(RuntimeError, "affinity"):
                LIMITS.verify_cgroup(self.config(), group)

    def test_worker_never_executes_command_after_failed_cgroup_gate(self):
        config_path = self.path / "config.json"
        config_path.write_text(json.dumps(self.config()))
        with mock.patch.object(LIMITS, "cgroup_path", return_value=self.group(**{"memory.max": "max"})), \
                mock.patch.object(LIMITS.subprocess, "Popen") as command:
            self.assertEqual(LIMITS.worker(config_path, False), 125)
        command.assert_not_called()
        self.assertEqual(json.loads((self.path / "worker-result.json").read_text())["status"], "setup-failed")

    def test_cleanup_uses_supported_kill_flag_and_waits_for_cgroup_to_empty(self):
        process = mock.Mock()
        process.wait.return_value = -9
        response = mock.Mock(returncode=0)
        with mock.patch.object(LIMITS.subprocess, "run", return_value=response) as control, \
                mock.patch.object(LIMITS, "kill_cgroup_processes") as direct_kill, \
                mock.patch.object(LIMITS, "cgroup_process_ids", side_effect=[{101}, set(), set()]), \
                mock.patch.object(LIMITS.time, "sleep"):
            result = LIMITS.stop_experiment(process, self.config()["unit"], False, self.path / "group")
        self.assertTrue(result["confirmed"])
        process.wait.assert_called_once()
        self.assertEqual(direct_kill.call_count, 2)
        commands = [call.args[0] for call in control.call_args_list]
        kill = next(command for command in commands if "kill" in command)
        self.assertIn("--kill-who=all", kill)
        self.assertNotIn("--kill-whom=all", kill)
        self.assertTrue(any("stop" in command for command in commands))

    def test_unconfirmed_cleanup_is_not_reported_as_a_successful_interrupt(self):
        summary = {"status": "interrupted", "returncode": 130}
        with mock.patch.object(LIMITS, "stop_experiment", return_value={"confirmed": False}):
            LIMITS.terminate(summary, mock.Mock(), self.config()["unit"], False, None)
        self.assertEqual(summary["status"], "termination-failed")
        self.assertEqual(summary["returncode"], 125)

    def test_service_sets_all_enforcement_properties_without_shell(self):
        command = LIMITS.service_command(self.config(), self.path / "config.json")
        for prop in ("MemoryMax=34359738368", "MemorySwapMax=0", "OOMPolicy=kill", "CPUQuota=400%",
                     "AllowedCPUs=2,3,4,5", "CPUAffinity=2 3 4 5", "TasksMax=128", "RuntimeMaxSec=600", "KillMode=control-group"):
            self.assertIn("--property=" + prop, command)
        self.assertEqual(command[0], "systemd-run")
        self.assertIn("--wait", command)
        self.assertNotIn("sh", command)
        self.assertFalse(any(arg.startswith("--property=LimitAS=") for arg in command))

    def test_default_wall_time_is_unlimited_and_does_not_limit_virtual_mapping(self):
        args = LIMITS.parser().parse_args(["--output-dir", str(self.path / "out"), "--", "/bin/true"])
        self.assertEqual(args.seconds, 0)
        config = self.config()
        config["seconds"] = 0
        command = LIMITS.service_command(config, self.path / "config.json")
        self.assertFalse(any(arg.startswith("--property=RuntimeMaxSec=") for arg in command))
        self.assertFalse(any(arg.startswith("--property=LimitAS=") for arg in command))

    def test_missing_user_bus_fails_closed_before_output_or_command(self):
        output = self.path / "new-output"
        args = LIMITS.parser().parse_args(["--output-dir", str(output), "--", "/bin/true"])
        with mock.patch.object(LIMITS, "user_systemd_available", return_value=False), \
                mock.patch.object(LIMITS.subprocess, "Popen") as command:
            with self.assertRaisesRegex(RuntimeError, "refusing"):
                LIMITS.run(args)
        self.assertFalse(output.exists())
        command.assert_not_called()

    def test_sparse_usage_counts_logical_size_and_does_not_follow_symlink(self):
        scope = self.path / "scope"
        scope.mkdir()
        sparse = scope / "sparse"
        with sparse.open("wb") as stream:
            stream.truncate(1 << 20)
        outside = self.path / "outside"
        outside.write_bytes(b"x" * 8192)
        (scope / "link").symlink_to(outside)
        measured = LIMITS.disk_bytes([scope])
        self.assertGreaterEqual(measured, 1 << 20)
        self.assertLess(measured, (1 << 20) + 8192)

    def test_nested_write_roots_count_once(self):
        outer = self.path / "outer"
        inner = outer / "inner"
        inner.mkdir(parents=True)
        self.assertEqual(LIMITS.nonoverlapping_roots([inner, outer, inner]), [outer.resolve()])

    def test_owned_output_must_be_new(self):
        args = LIMITS.parser().parse_args(["--output-dir", str(self.path), "--", "/bin/true"])
        with self.assertRaisesRegex(ValueError, "must not already exist"):
            LIMITS.run(args)

    def test_environment_caps_threads_and_keeps_caches_in_owned_scope(self):
        with mock.patch.dict(os.environ, {"PRIVATE_EXPERIMENT_SECRET": "not-transferable"}):
            env = LIMITS.worker_environment(self.config())
        self.assertEqual(env["RAYON_NUM_THREADS"], "4")
        self.assertEqual(env["CARGO_BUILD_JOBS"], "4")
        self.assertNotIn("PRIVATE_EXPERIMENT_SECRET", env)
        for key in ("TMPDIR", "CARGO_TARGET_DIR", "XDG_CACHE_HOME"):
            self.assertIn(self.path, Path(env[key]).parents)


if __name__ == "__main__":
    unittest.main()
