#!/usr/bin/env python3
"""Actual opt-in GMRES driver checks using an existing read-only graph.

Set ZXCL_RUNTIME_SOLVER and GRAPH_BIN as in test_gmres_runtime.py. The small
NONE-policy fixture isolates command dispatch, early rejection, publication,
and the independent original final gate. Nonzero signed ten-layer equivalence
is covered by loop1_prepared_tests in the Rust test binary.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ZXCL_RUNTIME_SOLVER') and os.environ.get('GRAPH_BIN'),
                     'set ZXCL_RUNTIME_SOLVER and GRAPH_BIN for actual solver runtime checks')
class CompiledL9RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zxcl-compiled-l9-runtime-')
        self.path = Path(self.tmp.name)
        self.pruned = self.path / 'pruned'
        self.policy = self.path / 'policy'
        self.pruned.mkdir()
        self.policy.mkdir()
        for layer in range(10):
            prefix = self.pruned / f'layer{layer}q_pruned'
            prefix.with_suffix('.field_index.bin').write_bytes(
                b'FCHNIDX1' + struct.pack('<IIQIIQ', 1, 1, 1, 0, 1, 0))
            prefix.with_suffix('.chains.u24.bin').write_bytes(b'\x00\x00\x00')
            (self.policy / f'layer{layer}.policy.u16.bin').write_bytes(b'\xff\xff')
        self.value = self.path / 'accepted.f32.bin'
        self.accepted = struct.pack('<f', 123.0)
        self.value.write_bytes(self.accepted)
        # Avoid inheriting production accelerators, large task caches, or caps.
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('PI_', 'FULL_MDP_'))}
        self.env.update(
            RAYON_NUM_THREADS='1', PI_LOOP1_COMPILED_L9='1', PI_GMRES_AITKEN='0',
            PI_ACCEPT_BELLMAN_INF='0.001', PI_GMRES_BASIS_MAX_GIB='0',
            FULL_MDP_MEMORY_MAX_GIB='1', FULL_MDP_MEMORY_RESERVE_GIB='0',
            FULL_MDP_RAM_NEXT_LAYER='0', FULL_MDP_BLOCK_CHUNK='1',
            FULL_MDP_BATCH_CHUNKS='1')

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, initial='zero', iterations=0, *, gamma='1', graph=None, overrides=None):
        env = dict(self.env)
        if overrides:
            env.update(overrides)
        return subprocess.run(
            [os.environ['ZXCL_RUNTIME_SOLVER'], 'pi-policy-eval-gmres',
             str(graph or os.environ['GRAPH_BIN']), str(self.pruned), str(self.policy),
             initial, str(self.value), str(self.path / 'scratch'), gamma,
             str(iterations), '1e-6', '0'], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)

    def test_compiled_driver_uses_original_final_gate_even_when_direct_l9_is_enabled(self):
        result = self.evaluate(iterations=1000000, overrides={'FULL_MDP_L9_DIRECT': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('compiled-l9', result.stderr)
        self.assertIn('sink=compiled-rank-direct', result.stderr)
        self.assertIn('krylov_iters=0', result.stderr)
        self.assertIn('final verification affine sweep T_pi(v) (original uncompiled)', result.stderr)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0))
        # Compiled applies skip L9's value sink. Only the forced original final
        # verification writes this file, proving that DIRECT=1 did not bypass it.
        self.assertEqual((self.path / 'scratch' / 'layer9.values.f32.bin').read_bytes(),
                         struct.pack('<f', 0.0))

    def test_compiled_driver_uses_ram_gather_when_next_layer_is_pinned(self):
        result = self.evaluate(iterations=1000000, overrides={
            'FULL_MDP_L9_DIRECT': '1',
            'FULL_MDP_RAM_NEXT_LAYER': '1',
            'FULL_MDP_RAM_NEXT_LAYER_MAX_GIB': '1',
            'FULL_MDP_RAM_PAIR_MAX_GIB': '1',
        })
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('sink=compiled-ram-gather', result.stderr)
        self.assertIn('krylov_iters=0', result.stderr)
        self.assertIn('final verification affine sweep T_pi(v) (original uncompiled)', result.stderr)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0))

    def test_rejected_candidate_keeps_accepted_output_after_original_residual_gate(self):
        initial = self.path / 'initial.f32.bin'
        initial.write_bytes(struct.pack('<f', -1.0))
        result = self.evaluate(str(initial), 0)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('compiled-l9', result.stderr)
        self.assertIn('final verification affine sweep T_pi(v) (original uncompiled)', result.stderr)
        self.assertIn('NOT ACCEPTED', result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_discount_rejected_before_graph_loading_or_scratch_creation(self):
        result = self.evaluate(gamma='0.5', graph=self.path / 'missing-graph.bin')
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PI_LOOP1_COMPILED_L9', result.stderr)
        self.assertIn('gamma', result.stderr)
        self.assertNotIn('loading graph', result.stderr)
        self.assertFalse((self.path / 'scratch').exists())
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_unreachable_corrupt_l9_action_rejected_before_any_sweep_or_publication(self):
        (self.policy / 'layer9.policy.u16.bin').write_bytes(struct.pack('<H', 0x0700))
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('invalid policy placed piece 7', result.stderr)
        self.assertNotIn('computing c_pi', result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_scratch_hardlink_to_accepted_value_rejected_with_either_solver_mode(self):
        scratch = self.path / 'scratch'
        scratch.mkdir()
        output = scratch / 'layer9.values.f32.bin'
        for enabled in ['0', '1']:
            with self.subTest(compiled=enabled):
                os.link(self.value, output)
                try:
                    result = self.evaluate(overrides={'PI_LOOP1_COMPILED_L9': enabled})
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn('alias', result.stderr)
                    self.assertNotIn('loading graph', result.stderr)
                    self.assertEqual(self.value.read_bytes(), self.accepted)
                    self.assertEqual(output.read_bytes(), self.accepted)
                finally:
                    output.unlink()

    def test_scratch_symlink_to_policy_rejected_before_source_is_modified(self):
        scratch = self.path / 'scratch'
        scratch.mkdir()
        policy = self.policy / 'layer0.policy.u16.bin'
        original = policy.read_bytes()
        output = scratch / 'layer9.values.f32.bin'
        output.symlink_to(policy)
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('alias', result.stderr)
        self.assertNotIn('loading graph', result.stderr)
        self.assertEqual(policy.read_bytes(), original)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_applied_output_hardlink_to_policy_rejected_before_source_is_modified(self):
        policy = self.policy / 'layer9.policy.u16.bin'
        original = policy.read_bytes()
        applied = self.value.with_suffix('.applied.f32.bin')
        os.link(policy, applied)
        result = self.evaluate()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('alias', result.stderr)
        self.assertNotIn('loading graph', result.stderr)
        self.assertEqual(policy.read_bytes(), original)
        self.assertEqual(applied.read_bytes(), original)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_dangling_scratch_symlink_to_absent_accepted_value_rejected(self):
        self.value.unlink()
        scratch = self.path / 'scratch'
        scratch.mkdir()
        output = scratch / 'layer0.values.f32.bin'
        output.symlink_to(self.value)
        inputs = {
            path: path.read_bytes()
            for directory in [self.pruned, self.policy]
            for path in directory.iterdir()
        }
        for enabled in ['0', '1']:
            with self.subTest(compiled=enabled):
                result = self.evaluate(overrides={'PI_LOOP1_COMPILED_L9': enabled})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('alias', result.stderr)
                self.assertNotIn('loading graph', result.stderr)
                self.assertFalse(self.value.exists())
                self.assertTrue(output.is_symlink())
                for path, original in inputs.items():
                    self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
