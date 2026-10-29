#!/usr/bin/env python3
"""Exercise the actual wrapper against an inexpensive, deliberately fallible solver."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('solve_manifest', ROOT / 'scripts/solve_manifest.py')
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)
MOCK = r'''#!/usr/bin/env python3
import json, os, signal, struct, sys
from pathlib import Path
cmd = sys.argv[1]
with open(os.environ['MOCK_LOG'], 'a') as log:
    existing_seed = sorted(p.name for p in Path(sys.argv[4]).iterdir()) if cmd == 'solve-full-mdp-l0-f32' and Path(sys.argv[4]).exists() else []
    log.write(json.dumps({'cmd': cmd, 'args': sys.argv[2:], 'policy_only': os.environ.get('FULL_MDP_POLICY_ONLY'), 'existing_seed': existing_seed}) + '\n')
if cmd == 'solve-one-loop-mdp-u16':
    kind = 'one-loop'
elif cmd == 'solve-full-mdp-l0-f32':
    kind = 'seed'
    output = Path(sys.argv[4]); output.mkdir(parents=True, exist_ok=True)
    (output / 'layer0.values.f32.bin').write_bytes(struct.pack('<ff', 1, 1))
elif cmd == 'emit-full-mdp-layers-from-l0':
    output, policy = Path(sys.argv[5]), Path(sys.argv[8])
    output.mkdir(parents=True, exist_ok=True); policy.mkdir(parents=True, exist_ok=True)
    kind = 'emit' if policy.name.startswith('policy_') and policy.name != 'policy_final' else 'final'
    action = 0 if policy.name == 'policy_0' else 1
    for k in range(10):
        (policy / f'layer{k}.policy.u16.bin').write_bytes(struct.pack('<HH', action, action))
        if os.environ.get('FULL_MDP_POLICY_ONLY') != '1' or k == 0:
            (output / f'layer{k}.values.f32.bin').write_bytes(struct.pack('<ff', 2, 2))
elif cmd in ('pi-policy-eval-gmres', 'pi-policy-eval-reduced'):
    kind = 'eval'
    round = int(Path(sys.argv[4]).name.split('_')[-1])
    Path(sys.argv[6]).write_bytes(struct.pack('<ff', round + 2, round + 2))
else:
    raise SystemExit('unexpected mock command: ' + cmd)
if os.environ.get('MOCK_MUTATE') == kind:
    changed = Path(os.environ.get('MOCK_MUTATE_PATH', os.environ['GRAPH_BIN']))
    changed.write_bytes(changed.read_bytes() + b'x')
if os.environ.get('MOCK_NONFINITE') == kind:
    target = Path(sys.argv[6]) if kind == 'eval' else output / 'layer0.values.f32.bin'
    target.write_bytes(struct.pack('<ff', float('nan'), 1))
if os.environ.get('MOCK_FAIL') == kind:
    raise SystemExit(23)
if os.environ.get('MOCK_KILL') == kind:
    os.kill(os.getppid(), signal.SIGKILL)
'''


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="zxcl-wrapper-")
        self.path = Path(self.tmp.name)
        self.pruned = self.path / 'pruned'
        self.pruned.mkdir()
        for k in range(10):
            prefix = self.pruned / f'layer{k}q_pruned'
            prefix.with_suffix('.field_index.bin').write_bytes(b'FCHNIDX1' + struct.pack('<IIQIIQ', 1, 1, 2, k, 2, 0))
            prefix.with_suffix('.chains.u24.bin').write_bytes(b'\x00\x00\x00\x01\x00\x00')
        self.graph = self.path / 'graph.bin'
        self.graph.write_bytes(b'graph-A')
        self.solver = self.path / 'mock_solver'
        self.solver.write_text(MOCK)
        self.solver.chmod(0o755)
        self.log = self.path / 'calls.jsonl'
        self.env = dict(os.environ, GRAPH_BIN=str(self.graph), PRUNED_DIR=str(self.pruned),
                        SOLVER_BIN=str(self.solver), ZXCL_DATA_DIR=str(self.path / 'data'),
                        ZXCL_SCRATCH_DIR=str(self.path / 'work'), MOCK_LOG=str(self.log),
                        RAYON_NUM_THREADS='1', MAX_PI_ROUNDS='4',
                        MDP_GAMMA='1.0', GMRES_MAX_ITERS='300', GMRES_REL_TOL='1e-6', CACHE_GIB='16',
                        PI_POLICY_EVAL_BACKEND='gmres', PI_REDUCED_OPERATOR='sweep',
                        PI_REDUCED_METHOD='gmres')

    def tearDown(self):
        self.tmp.cleanup()

    def run_wrapper(self, command='solve', **extra):
        result = subprocess.run(['bash', str(ROOT / 'scripts/solve_mdp.sh'), command], env=dict(self.env, **extra),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result

    def calls(self, command=None):
        entries = [json.loads(line) for line in self.log.read_text().splitlines()]
        return [entry for entry in entries if command is None or entry['cmd'] == command]

    def assert_ok(self, result):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_rejected_preallocated_value_is_recomputed_and_warmstarts_previous_value(self):
        first = self.run_wrapper(MOCK_FAIL='eval')
        self.assertNotEqual(first.returncode, 0)
        value = self.path / 'data/results/v_pi0.f32.bin'
        self.assertTrue(value.exists())
        self.assertFalse(Path(str(value) + '.complete.json').exists())
        self.assert_ok(self.run_wrapper())
        evaluations = self.calls('pi-policy-eval-gmres')
        self.assertEqual(len(evaluations), 3)  # failed round 0, accepted rounds 0 and 1
        self.assertEqual(evaluations[-1]['args'][3], str(value))
        self.assertEqual(len(self.calls('emit-full-mdp-layers-from-l0')), 3)
        self.assertTrue(all(c['policy_only'] == '1' for c in self.calls('emit-full-mdp-layers-from-l0')))
        before = len(self.calls())
        self.assert_ok(self.run_wrapper())
        self.assertEqual(len(self.calls()), before)

    def test_reduced_dispatch_retries_and_warmstarts_previous_accepted_value(self):
        first = self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced', MOCK_FAIL='eval')
        self.assertNotEqual(first.returncode, 0)
        value = self.path / 'data/results/v_pi0.f32.bin'
        completion = Path(str(value) + '.complete.json')
        self.assertTrue(value.exists())
        self.assertFalse(completion.exists())
        self.assert_ok(self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced'))
        evaluations = self.calls('pi-policy-eval-reduced')
        self.assertEqual(len(evaluations), 3)
        self.assertEqual(self.calls('pi-policy-eval-gmres'), [])
        self.assertEqual(evaluations[1]['args'], [
            str(self.graph), str(self.pruned), str(self.path / 'data/results/policy_0'),
            str(self.path / 'data/seed_l0/layer0.values.f32.bin'), str(value),
            str(self.path / 'work/pi/eval_0'), '1.0', '300', '1e-6', '16',
        ])
        self.assertEqual(evaluations[-1]['args'][3], str(value))
        self.assertTrue(completion.exists())
        self.assertTrue((value.parent / 'stable.complete.json').exists())
        self.assertTrue(all(c['policy_only'] == '1' for c in self.calls('emit-full-mdp-layers-from-l0')))
        before = len(self.calls())
        self.assert_ok(self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced'))
        self.assertEqual(len(self.calls()), before)

    def test_switching_policy_eval_backend_invalidates_stage_certificates(self):
        self.assert_ok(self.run_wrapper())
        self.assert_ok(self.run_wrapper('finalize'))
        certificates = [
            self.path / 'data/seed_l0/seed.complete.json',
            self.path / 'data/results/policy_0/policy.complete.json',
            self.path / 'data/results/v_pi0.f32.bin.complete.json',
            self.path / 'data/results/stable.complete.json',
            self.path / 'data/results/final.complete.json',
        ]
        old_contexts = [json.loads(path.read_text())['context'] for path in certificates]
        before = len(self.calls())
        result = self.run_wrapper('finalize', PI_POLICY_EVAL_BACKEND='reduced')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('no validated stable result', result.stderr)
        self.assertEqual(len(self.calls()), before)
        self.assert_ok(self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced'))
        self.assert_ok(self.run_wrapper('finalize', PI_POLICY_EVAL_BACKEND='reduced'))
        self.assertEqual(len(self.calls('pi-policy-eval-gmres')), 2)
        self.assertEqual(len(self.calls('pi-policy-eval-reduced')), 2)
        self.assertEqual(len(self.calls('solve-full-mdp-l0-f32')), 2)
        new_contexts = [json.loads(path.read_text())['context'] for path in certificates]
        self.assertTrue(all(old != new for old, new in zip(old_contexts, new_contexts)))

    def test_reduced_rejects_invalid_configuration_before_solver_runs(self):
        invalid = [
            ({'PI_POLICY_EVAL_BACKEND': 'invalid'}, 'invalid PI_POLICY_EVAL_BACKEND'),
            ({'PI_POLICY_EVAL_BACKEND': ''}, 'invalid PI_POLICY_EVAL_BACKEND'),
            ({'PI_REDUCED_OPERATOR': 'invalid'}, 'invalid PI_REDUCED_OPERATOR'),
            ({'PI_REDUCED_OPERATOR': ''}, 'invalid PI_REDUCED_OPERATOR'),
            ({'PI_REDUCED_METHOD': 'invalid'}, 'invalid PI_REDUCED_METHOD'),
            ({'PI_REDUCED_METHOD': ''}, 'invalid PI_REDUCED_METHOD'),
            ({'PI_REDUCED_METHOD': 'deflated'}, 'requires PI_REDUCED_OPERATOR=dag'),
        ]
        for gamma in ('0.9', 'nan', 'inf', '-inf', '1e309', 'invalid'):
            invalid.append(({'MDP_GAMMA': gamma}, 'requires finite MDP_GAMMA=1'))
        for settings, error in invalid:
            with self.subTest(settings=settings):
                result = self.run_wrapper(**dict({'PI_POLICY_EVAL_BACKEND': 'reduced'}, **settings))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(self.log.exists())
                self.assertFalse((self.path / 'data').exists())

    def test_reduced_accepts_numeric_gamma_one_spellings(self):
        for gamma in ('1', '1.0', '1.00'):
            with self.subTest(gamma=gamma):
                self.assert_ok(self.run_wrapper('one-loop', PI_POLICY_EVAL_BACKEND='reduced', MDP_GAMMA=gamma))
        self.assertEqual(len(self.calls('solve-full-mdp-l0-f32')), 3)

    def test_reduced_nonfinite_or_changed_policy_does_not_receive_certificate(self):
        value = self.path / 'data/results/v_pi0.f32.bin'
        policy = self.path / 'data/results/policy_0/layer0.policy.u16.bin'
        for settings, error in [
            ({'MOCK_NONFINITE': 'eval'}, 'nonfinite'),
            ({'MOCK_MUTATE': 'eval', 'MOCK_MUTATE_PATH': str(policy)}, 'input changed during solve'),
        ]:
            with self.subTest(settings=settings):
                result = self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced', **settings)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(error, result.stderr)
                self.assertFalse(Path(str(value) + '.complete.json').exists())
        self.assert_ok(self.run_wrapper(PI_POLICY_EVAL_BACKEND='reduced'))

    def test_killed_wrapper_does_not_reuse_preallocated_policy(self):
        killed = self.run_wrapper(MOCK_KILL='emit')
        self.assertNotEqual(killed.returncode, 0)
        marker = self.path / 'data/results/policy_0/policy.complete.json'
        self.assertFalse(marker.exists())
        self.assert_ok(self.run_wrapper())
        self.assertEqual(len(self.calls('emit-full-mdp-layers-from-l0')), 4)

    def test_partial_seed_is_recomputed_and_final_materialization_is_certified(self):
        self.assertNotEqual(self.run_wrapper('one-loop', MOCK_FAIL='seed').returncode, 0)
        self.assertFalse((self.path / 'data/seed_l0/seed.complete.json').exists())
        self.assert_ok(self.run_wrapper())
        self.assertEqual(len(self.calls('solve-full-mdp-l0-f32')), 2)
        self.assertNotEqual(self.run_wrapper('finalize', MOCK_FAIL='final').returncode, 0)
        marker = self.path / 'data/results/final.complete.json'
        self.assertFalse(marker.exists())
        self.assert_ok(self.run_wrapper('finalize'))
        self.assertTrue(marker.exists())
        self.assertEqual(self.calls('emit-full-mdp-layers-from-l0')[-1]['policy_only'], '0')
        before = len(self.calls())
        self.assert_ok(self.run_wrapper('finalize'))
        self.assertEqual(len(self.calls()), before)

    def test_changed_graph_index_chain_and_policy_invalidate_completion(self):
        self.assert_ok(self.run_wrapper())
        for path, replacement in [
            (self.graph, b'graph-B'),
            (self.pruned / 'layer1q_pruned.field_index.bin', b'FCHNIDX1' + struct.pack('<IIQIIQ', 1, 1, 2, 99, 2, 0)),
            (self.pruned / 'layer1q_pruned.chains.u24.bin', b'\x00\x00\x00\x02\x00\x00'),
        ]:
            before = len(self.calls('solve-full-mdp-l0-f32'))
            path.write_bytes(replacement)
            self.assert_ok(self.run_wrapper())
            self.assertEqual(len(self.calls('solve-full-mdp-l0-f32')), before + 1)
        before = len(self.calls('pi-policy-eval-gmres'))
        (self.path / 'data/results/policy_0/layer0.policy.u16.bin').write_bytes(b'\x02\x00\x02\x00')
        self.assert_ok(self.run_wrapper())
        self.assertGreater(len(self.calls('pi-policy-eval-gmres')), before)

    def test_changed_parameters_invalidate_accepted_results(self):
        self.assert_ok(self.run_wrapper())
        before = len(self.calls('pi-policy-eval-gmres'))
        self.assert_ok(self.run_wrapper(MDP_GAMMA='0.9'))
        self.assertEqual(len(self.calls('pi-policy-eval-gmres')), before + 2)
        before = len(self.calls('pi-policy-eval-gmres'))
        self.assert_ok(self.run_wrapper(MDP_GAMMA='0.9', GMRES_REL_TOL='1e-7'))
        self.assertGreater(len(self.calls('pi-policy-eval-gmres')), before)

    def test_seed_rebuild_uses_clean_directory_and_preserves_accepted_seed_on_failure(self):
        self.assert_ok(self.run_wrapper('one-loop'))
        seed = self.path / 'data/seed_l0/layer0.values.f32.bin'
        completion = seed.parent / 'seed.complete.json'
        accepted = seed.read_bytes(), completion.read_bytes()
        # These are deliberately resumable direct-solver files from another run.
        (seed.parent / 'layer0.input.f32.bin').write_bytes(struct.pack('<ff', 100, 100))
        failed = self.run_wrapper('one-loop', SEED_MAX_ITERS='2', MOCK_FAIL='seed')
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual((seed.read_bytes(), completion.read_bytes()), accepted)
        self.assert_ok(self.run_wrapper('one-loop', SEED_MAX_ITERS='2'))
        self.assertTrue(all(not call['existing_seed'] for call in self.calls('solve-full-mdp-l0-f32')))
        self.assertFalse(list(seed.parent.glob('.seed-build.*')))

    def test_copied_completion_cannot_certify_another_destination(self):
        self.assert_ok(self.run_wrapper('one-loop'))
        copied = self.path / 'seed_copy'
        shutil.copytree(self.path / 'data/seed_l0', copied)
        (copied / 'layer0.values.f32.bin').write_bytes(b'bad')
        before = len(self.calls('solve-full-mdp-l0-f32'))
        self.assert_ok(self.run_wrapper('one-loop', SEED_DIR=str(copied)))
        self.assertEqual(len(self.calls('solve-full-mdp-l0-f32')), before + 1)
        self.assertEqual((copied / 'layer0.values.f32.bin').stat().st_size, 8)

    def test_inputs_changed_during_stage_do_not_receive_completion(self):
        result = self.run_wrapper(MOCK_MUTATE='seed')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('input changed during solve', result.stderr)
        self.assertFalse((self.path / 'data/seed_l0/seed.complete.json').exists())
        self.assert_ok(self.run_wrapper())

    def test_policy_changed_during_evaluation_is_not_accepted(self):
        value = self.path / 'data/results/v_pi0.f32.bin'
        policy = self.path / 'data/results/policy_0/layer0.policy.u16.bin'
        result = self.run_wrapper(MOCK_MUTATE='eval', MOCK_MUTATE_PATH=str(policy))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('input changed during solve', result.stderr)
        self.assertFalse(Path(str(value) + '.complete.json').exists())
        self.assert_ok(self.run_wrapper())

    def test_nonfinite_boundary_is_not_certified(self):
        result = self.run_wrapper(MOCK_NONFINITE='eval')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('nonfinite', result.stderr)
        self.assertFalse((self.path / 'data/results/v_pi0.f32.bin.complete.json').exists())
        self.assert_ok(self.run_wrapper())

    def test_from_vstar_snapshots_source_when_it_aliases_layer0_output(self):
        source = self.path / 'data/results/vstar_full/layer0.values.f32.bin'
        source.parent.mkdir(parents=True)
        supplied = struct.pack('<ff', 7, 9)
        source.write_bytes(supplied)
        self.assert_ok(self.run_wrapper('from-vstar', VSTAR_LAYER0=str(source)))
        self.assertEqual(source.read_bytes(), supplied)
        self.assertEqual((self.path / 'data/results/vstar_boundary.f32.bin').read_bytes(), supplied)
        self.assertTrue((self.path / 'data/results/from-vstar.complete.json').exists())
        self.assertFalse(list((self.path / 'work').glob('.from-vstar.*')))


class ManifestTests(unittest.TestCase):
    def test_unchanged_inputs_reuse_cache_and_malformed_cache_is_repaired(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-manifest-') as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'content')
            cache = Path(tmp) / 'cache'
            expected = MANIFEST.content_identity(path, cache)
            entry = next(cache.glob('*.json'))
            signature = MANIFEST.signature(entry)
            self.assertEqual(MANIFEST.content_identity(path, cache), expected)
            self.assertEqual(MANIFEST.signature(entry), signature)
            for invalid in [[], {}, {'signature': MANIFEST.signature(path)},
                            {'signature': MANIFEST.signature(path), 'sha256': 'invalid'}]:
                entry.write_text(json.dumps(invalid))
                self.assertEqual(MANIFEST.content_identity(path, cache), expected)
                self.assertEqual(json.loads(entry.read_text())['sha256'], hashlib.sha256(b'content').hexdigest())

    def test_small_file_snapshot_rejects_changed_content_when_all_metadata_matches(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-snapshot-small-') as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'old')
            before = path.stat()
            stored = MANIFEST.signature(path)
            path.write_bytes(b'new')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            # Neutralize ctime deterministically rather than relying on the
            # filesystem clock to place both writes in the same tick.
            stored['ctime_ns'] = path.stat().st_ctime_ns
            current = MANIFEST.signature(path)
            self.assertEqual({k: v for k, v in stored.items() if k != 'small_sha256'},
                             {k: v for k, v in current.items() if k != 'small_sha256'})
            with self.assertRaisesRegex(ValueError, 'input changed during solve'):
                MANIFEST.guard_snapshot(MANIFEST.canonical([stored]).decode())

    def test_small_file_cache_rechecks_content_when_metadata_matches(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-cache-small-') as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'old')
            before = path.stat()
            cache = Path(tmp) / 'cache'
            old_identity = MANIFEST.content_identity(path, cache)
            entry = next(cache.glob('*.json'))
            saved = json.loads(entry.read_text())
            path.write_bytes(b'new')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            saved['signature']['ctime_ns'] = path.stat().st_ctime_ns
            entry.write_text(json.dumps(saved))
            new_identity = MANIFEST.content_identity(path, cache)
            self.assertNotEqual(new_identity, old_identity)
            self.assertEqual(new_identity['sha256'], hashlib.sha256(b'new').hexdigest())

    def test_small_signature_rejects_a_file_changed_while_fingerprinting(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-signature-race-') as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'old')
            real_hash = hashlib.sha256

            def change_after_read(data):
                path.write_bytes(b'changed-size')
                return real_hash(data)

            with mock.patch.object(MANIFEST.hashlib, 'sha256', side_effect=change_after_read):
                with self.assertRaisesRegex(ValueError, 'changed while fingerprinting'):
                    MANIFEST.signature(path)

    def test_large_signature_does_not_read_input_contents(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-signature-large-') as tmp:
            path = Path(tmp) / 'input'
            with path.open('wb') as stream:
                stream.truncate(MANIFEST.SMALL_SIGNATURE_BYTES + 1)
            with mock.patch.object(Path, 'open', side_effect=AssertionError('large signature must not read')):
                result = MANIFEST.signature(path)
            self.assertNotIn('small_sha256', result)

    def test_snapshot_rejects_changed_input_even_when_size_and_mtime_are_restored(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-snapshot-') as tmp:
            path = Path(tmp) / 'input'
            path.write_bytes(b'old')
            before = path.stat()
            snapshot = MANIFEST.canonical([MANIFEST.signature(path)]).decode()
            MANIFEST.guard_snapshot(snapshot)
            path.write_bytes(b'new')
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            with self.assertRaisesRegex(ValueError, 'input changed during solve'):
                MANIFEST.guard_snapshot(snapshot)

    def test_snapshot_rejects_retargeted_symlink_while_old_target_stays_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='zxcl-symlink-snapshot-') as tmp:
            first, second, alias = (Path(tmp) / name for name in ('first', 'second', 'alias'))
            first.write_bytes(b'first')
            second.write_bytes(b'second')
            alias.symlink_to(first)
            snapshot = MANIFEST.canonical([MANIFEST.signature(alias)]).decode()
            MANIFEST.guard_snapshot(snapshot)
            alias.unlink()
            alias.symlink_to(second)
            with self.assertRaisesRegex(ValueError, 'input changed during solve'):
                MANIFEST.guard_snapshot(snapshot)


if __name__ == '__main__':
    unittest.main()
