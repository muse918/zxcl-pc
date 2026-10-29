#!/usr/bin/env python3
"""Actual GMRES CLI checks with tiny membership and an existing read-only graph.

Run after building the solver:
  ZXCL_RUNTIME_SOLVER=/absolute/path/to/zxcl GRAPH_BIN=/absolute/path/to/graph.bin \
    python3 -m unittest discover -s tests -p 'test_gmres_runtime.py' -v
The NONE policy makes every affine/linear sweep zero, without building an MDP.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ZXCL_RUNTIME_SOLVER') and os.environ.get('GRAPH_BIN'),
                     'set ZXCL_RUNTIME_SOLVER and GRAPH_BIN for actual solver runtime checks')
class GmresRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zxcl-gmres-runtime-')
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
        self.env = dict(os.environ, RAYON_NUM_THREADS='1', PI_GMRES_AITKEN='0',
                        PI_ACCEPT_BELLMAN_INF='0.001', PI_GMRES_BASIS_MAX_GIB='0',
                        FULL_MDP_RAM_NEXT_LAYER='1', FULL_MDP_RAM_NEXT_LAYER_MAX_GIB='1',
                        FULL_MDP_BLOCK_CHUNK='1', FULL_MDP_BATCH_CHUNKS='1')

    def tearDown(self):
        self.tmp.cleanup()

    def evaluate(self, initial, iterations):
        return subprocess.run(
            [os.environ['ZXCL_RUNTIME_SOLVER'], 'pi-policy-eval-gmres', os.environ['GRAPH_BIN'],
             str(self.pruned), str(self.policy), initial, str(self.value), str(self.path / 'scratch'),
             '1', str(iterations), '1e-6', '0'], env=self.env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)

    def test_converged_warmstart_skips_unused_basis_and_still_verifies_true_residual(self):
        result = self.evaluate('zero', 1000000)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('krylov_iters=0', result.stderr)
        self.assertIn('final verification affine sweep T_pi(v)', result.stderr)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0))

    def test_rejected_finite_or_nonfinite_candidate_preserves_accepted_output(self):
        accepted = struct.pack('<f', 123)
        for value in [1.0, float('nan'), float('inf')]:
            with self.subTest(initial=value):
                self.value.write_bytes(accepted)
                initial = self.path / 'initial.f32.bin'
                initial.write_bytes(struct.pack('<f', value))
                result = self.evaluate(str(initial), 0)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.value.read_bytes(), accepted)
                if value == 1.0:
                    self.assertIn('NOT ACCEPTED', result.stderr)
                    self.assertIn('final verification affine sweep T_pi(v)', result.stderr)
                else:
                    self.assertIn('nonfinite GMRES initial value', result.stderr)

    def test_nonconverged_start_still_enforces_basis_budget_without_publication(self):
        accepted = struct.pack('<f', 123)
        self.value.write_bytes(accepted)
        initial = self.path / 'initial.f32.bin'
        initial.write_bytes(struct.pack('<f', 1.0))
        result = self.evaluate(str(initial), 1)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('GMRES basis would need', result.stderr)
        self.assertEqual(self.value.read_bytes(), accepted)


if __name__ == '__main__':
    unittest.main()
