#!/usr/bin/env python3
"""Exercise the reduced evaluator CLI with an existing, read-only graph.

Run after building the solver:
  ZXCL_RUNTIME_SOLVER=/absolute/path/to/zxcl GRAPH_BIN=/absolute/path/to/graph.bin \
    python3 -m unittest discover -s tests -p 'test_reduced_runtime.py' -v

The reset layer contains the complete production quotient boundary. Its NONE
policy makes K=P=0, so u=1 and the reconstructed quotient value is zero. Other
layers are empty; no full MDP or production membership files are needed.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


PHYSICAL_EMPTY_FIELD_ID = 0
PC_TERMINAL_FIELD_ID = 15_185_705
QSTATE_COUNT = 817_740


@unittest.skipUnless(os.environ.get('ZXCL_RUNTIME_SOLVER') and os.environ.get('GRAPH_BIN'),
                     'set ZXCL_RUNTIME_SOLVER and GRAPH_BIN for actual solver runtime checks')
class ReducedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zxcl-reduced-runtime-')
        self.path = Path(self.tmp.name)
        self.pruned = self.path / 'pruned'
        self.policy = self.path / 'policy'
        self.pruned.mkdir()
        self.policy.mkdir()
        self.write_reset_membership(QSTATE_COUNT)
        for layer in range(1, 10):
            prefix = self.pruned / f'layer{layer}q_pruned'
            prefix.with_suffix('.field_index.bin').write_bytes(
                b'FCHNIDX1' + struct.pack('<IIQ', 1, 0, 0))
            prefix.with_suffix('.chains.u24.bin').write_bytes(b'')
            (self.policy / f'layer{layer}.policy.u16.bin').write_bytes(b'')
        self.value = self.path / 'accepted.f32.bin'
        self.accepted = struct.pack('<f', 123.0) * QSTATE_COUNT
        # Inherited production tuning must not make these small fixtures reserve
        # large caches, enable unrelated accelerators, or change the solver mode.
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('PI_', 'FULL_MDP_'))}
        self.env.update(
            RAYON_NUM_THREADS='1', PI_REDUCED_OPERATOR='sweep',
            PI_REDUCED_METHOD='gmres', PI_GMRES_AITKEN='0',
            PI_ACCEPT_BELLMAN_INF='0.001', PI_GMRES_BASIS_MAX_GIB='1',
            PI_DAG_MAX_STATES='1000000', PI_DAG_MAX_NODES='100',
            PI_DAG_MAX_EDGES='100', PI_DAG_FIELD_CACHE_MIB='8',
            FULL_MDP_RAM_NEXT_LAYER='1', FULL_MDP_RAM_NEXT_LAYER_MAX_GIB='1',
            FULL_MDP_RAM_PAIR_MAX_GIB='1', FULL_MDP_BLOCK_CHUNK='1',
            FULL_MDP_BATCH_CHUNKS='1')

    def tearDown(self):
        self.tmp.cleanup()

    def write_reset_membership(self, count, field=PHYSICAL_EMPTY_FIELD_ID):
        prefix = self.pruned / 'layer0q_pruned'
        prefix.with_suffix('.field_index.bin').write_bytes(
            b'FCHNIDX1' + struct.pack('<IIQIIQ', 1, 1, count, field, count, 0))
        chains = bytearray(count * 3)
        for qid in range(count):
            offset = qid * 3
            chains[offset] = qid & 0xff
            chains[offset + 1] = (qid >> 8) & 0xff
            chains[offset + 2] = qid >> 16
        prefix.with_suffix('.chains.u24.bin').write_bytes(chains)
        (self.policy / 'layer0.policy.u16.bin').write_bytes(b'\xff\xff' * count)

    def initial_file(self, value):
        initial = self.path / 'initial.f32.bin'
        initial.write_bytes(struct.pack('<f', value) * QSTATE_COUNT)
        return str(initial)

    def evaluate(self, initial='zero', iterations=1, *, operator='sweep', method='gmres',
                 gamma='1', overrides=None, graph=None):
        env = dict(self.env, PI_REDUCED_OPERATOR=operator, PI_REDUCED_METHOD=method)
        if overrides:
            env.update(overrides)
        return subprocess.run(
            [os.environ['ZXCL_RUNTIME_SOLVER'], 'pi-policy-eval-reduced',
             str(graph or os.environ['GRAPH_BIN']), str(self.pruned), str(self.policy),
             initial, str(self.value), str(self.path / 'scratch'), gamma,
             str(iterations), '1e-6', '0'], env=env,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)

    def assert_rejected_without_publication(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_none_policy_publishes_zero_with_sweep_dag_and_deflation(self):
        for operator, method in [('sweep', 'gmres'), ('dag', 'gmres'), ('dag', 'deflated')]:
            with self.subTest(operator=operator, method=method):
                self.value.write_bytes(self.accepted)
                result = self.evaluate(operator=operator, method=method,
                                       overrides={'PI_GMRES_BASIS_MAX_GIB':
                                                  '0' if method == 'gmres' else '1'})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.value.read_bytes(), b'\x00' * (4 * QSTATE_COUNT))
                self.assertIn('reset_membership=complete', result.stderr)
                self.assertIn('true_reduced_residual_l2=', result.stderr)
                self.assertIn('final verification affine sweep T_pi(v)', result.stderr)
                if method == 'gmres':
                    self.assertIn('krylov_iters=0', result.stderr)
                else:
                    self.assertIn('fixed_policy_certificate=verified', result.stderr)

    def test_nonzero_warmstart_converges_in_one_iteration(self):
        result = self.evaluate(self.initial_file(1.0), 1)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.value.read_bytes(), b'\x00' * (4 * QSTATE_COUNT))
        self.assertIn('krylov_iters=1', result.stderr)
        self.assertIn('final verification affine sweep T_pi(v)', result.stderr)

    def test_iteration_exhaustion_and_nonfinite_initial_preserve_accepted_output(self):
        for initial in [1.0, float('nan'), float('inf')]:
            with self.subTest(initial=initial):
                self.value.write_bytes(self.accepted)
                result = self.evaluate(self.initial_file(initial), 0, operator='dag')
                self.assert_rejected_without_publication(result)
                if initial == 1.0:
                    self.assertIn('NOT ACCEPTED', result.stderr)
                    self.assertIn('true |(I-P)u-1|', result.stderr)
                else:
                    self.assertIn('nonfinite reduced policy initial value', result.stderr)

    def test_incomplete_reset_boundary_preserves_accepted_output(self):
        self.write_reset_membership(1)
        self.value.write_bytes(self.accepted)
        result = self.evaluate(operator='dag')
        self.assert_rejected_without_publication(result)
        self.assertIn('incomplete layer0 reset membership', result.stderr)

    def test_terminal_sentinel_is_not_a_physical_reset_field(self):
        self.write_reset_membership(QSTATE_COUNT, field=PC_TERMINAL_FIELD_ID)
        self.value.write_bytes(self.accepted)
        result = self.evaluate(operator='dag')
        self.assert_rejected_without_publication(result)
        self.assertIn('requires a physically empty layer0 reset field (hash=0)', result.stderr)

    def test_invalid_gamma_and_configuration_reject_before_graph_load(self):
        missing_graph = self.path / 'must-not-load-graph.bin'
        cases = [
            ({'gamma': '0.9'}, 'gamma'),
            ({'operator': 'invalid'}, 'PI_REDUCED_OPERATOR'),
            ({'method': 'invalid'}, 'PI_REDUCED_METHOD'),
            ({'operator': 'sweep', 'method': 'deflated'}, 'dag'),
        ]
        for options, diagnostic in cases:
            with self.subTest(options=options):
                self.value.write_bytes(self.accepted)
                result = self.evaluate(graph=missing_graph, **options)
                self.assert_rejected_without_publication(result)
                self.assertIn(diagnostic, result.stderr)
                self.assertNotIn(str(missing_graph), result.stderr)

    def test_dag_state_cap_failure_preserves_accepted_output(self):
        self.value.write_bytes(self.accepted)
        result = self.evaluate(operator='dag', overrides={'PI_DAG_MAX_STATES': '1'})
        self.assert_rejected_without_publication(result)
        self.assertIn('state cap', result.stderr.lower())

    def test_hardlinked_scratch_file_preserves_accepted_output(self):
        self.value.write_bytes(self.accepted)
        scratch = self.path / 'scratch'
        scratch.mkdir()
        scratch_value = scratch / 'layer9.values.f32.bin'
        os.link(self.value, scratch_value)
        result = self.evaluate(
            self.initial_file(1.0), 0,
            overrides={'FULL_MDP_RAM_NEXT_LAYER': '0', 'FULL_MDP_MMAP_SPILL': '0'})
        self.assert_rejected_without_publication(result)
        self.assertEqual(scratch_value.read_bytes(), self.accepted)
        self.assertIn('accepted output must differ from fixed-policy scratch value files',
                      result.stderr)
        self.assertIn('path or inode alias at layer 9', result.stderr)


if __name__ == '__main__':
    unittest.main()
