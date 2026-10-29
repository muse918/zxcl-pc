#!/usr/bin/env python3
"""Whole-DAG production CLI routing and publication checks.

Set ZXCL_RUNTIME_SOLVER and GRAPH_BIN to a built solver and the read-only
production graph. Two stored L0 roots and NONE policies keep this fixture small.
Nonzero ordered arithmetic and source mutation after compilation are checked in
Rust; these tests exercise the real CLI, full validation, and independent gate.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


QSTATE_COUNT = 817_740
PC_TERMINAL_FIELD_ID = 15_185_705


@unittest.skipUnless(os.environ.get('ZXCL_RUNTIME_SOLVER') and os.environ.get('GRAPH_BIN'),
                     'set ZXCL_RUNTIME_SOLVER and GRAPH_BIN for actual solver runtime checks')
class WholeDagRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zxcl-whole-dag-runtime-')
        self.path = Path(self.tmp.name)
        self.pruned = self.path / 'pruned'
        self.policy = self.path / 'policy'
        self.pruned.mkdir()
        self.policy.mkdir()
        for layer in range(10):
            self.write_membership(layer, [0, 1] if layer == 0 else [0])
        self.value = self.path / 'accepted.f32.bin'
        self.accepted = struct.pack('<f', 123.0) * 2
        self.value.write_bytes(self.accepted)
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('PI_', 'FULL_MDP_'))}
        self.env.update(
            RAYON_NUM_THREADS='1', PI_LOOP1_WHOLE_DAG='1',
            PI_LOOP1_WHOLE_MEMO_MIB='1', PI_LOOP1_WHOLE_REFERENCE_MEMO_MIB='1',
            PI_LOOP1_WHOLE_MAX_NODES='100', PI_LOOP1_WHOLE_MAX_EDGES='1000',
            PI_LOOP1_WHOLE_L9_TAIL='1', PI_GMRES_AITKEN='0',
            PI_ACCEPT_BELLMAN_INF='0.001', PI_GMRES_BASIS_MAX_GIB='0',
            FULL_MDP_MEMORY_MAX_GIB='1', FULL_MDP_MEMORY_RESERVE_GIB='0',
            FULL_MDP_RAM_NEXT_LAYER='0', FULL_MDP_BLOCK_CHUNK='1',
            FULL_MDP_BATCH_CHUNKS='1', FULL_MDP_MMAP_SPILL='0')

    def tearDown(self):
        self.tmp.cleanup()

    def write_membership(self, layer, qids, field=0):
        prefix = self.pruned / f'layer{layer}q_pruned'
        prefix.with_suffix('.field_index.bin').write_bytes(
            b'FCHNIDX1' + struct.pack('<IIQIIQ', 1, 1, len(qids), field, len(qids), 0))
        prefix.with_suffix('.chains.u24.bin').write_bytes(
            b''.join(qid.to_bytes(3, 'little') for qid in qids))
        (self.policy / f'layer{layer}.policy.u16.bin').write_bytes(b'\xff\xff' * len(qids))

    def run_solver(self, command='apply', *, initial='zero', iterations=0, gamma='1',
                   graph=None, output=None, scratch=None, compare=None, overrides=None):
        env = dict(self.env)
        if overrides:
            env.update(overrides)
        args = [os.environ['ZXCL_RUNTIME_SOLVER'],
                'apply-fixed-policy-boundary' if command == 'apply' else 'pi-policy-eval-gmres',
                str(graph or os.environ['GRAPH_BIN']), str(self.pruned), str(self.policy),
                str(initial), str(output or self.value), str(scratch or self.path / 'scratch'), gamma]
        if command == 'apply':
            args.append('0')
            if compare is not None:
                args.append(str(compare))
        else:
            args.extend([str(iterations), '1e-6', '0'])
        return subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=180)

    def assert_not_published(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def assert_no_numeric_layer_files(self, scratch):
        self.assertEqual(list(scratch.glob('layer*.values.f32.bin')), [])

    def test_unsupported_gamma_conflicting_modes_and_invalid_caps_reject_before_io(self):
        missing_graph = self.path / 'must-not-load-graph.bin'
        cases = [
            ({'gamma': '0.5'}, 'gamma'),
            ({'overrides': {'PI_LOOP1_COMPILED_L8': '1'}}, 'PI_LOOP1_COMPILED_L8'),
            ({'overrides': {'PI_LOOP1_ACTION_CACHE_MIB': '1'}}, 'PI_LOOP1_ACTION_CACHE_MIB'),
            ({'overrides': {'PI_LOOP1_WHOLE_MEMO_MIB': 'not-a-number'}},
             'PI_LOOP1_WHOLE_MEMO_MIB'),
            ({'overrides': {'PI_LOOP1_WHOLE_MAX_NODES': 'not-a-number'}},
             'PI_LOOP1_WHOLE_MAX_NODES'),
        ]
        for command in ['apply', 'gmres']:
            for options, diagnostic in cases:
                with self.subTest(command=command, options=options):
                    result = self.run_solver(command, graph=missing_graph, **options)
                    self.assert_not_published(result)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertNotIn('loading graph', result.stderr)
                    self.assertFalse((self.path / 'scratch').exists())

    def test_single_apply_covers_both_stored_roots_and_matches_original_with_either_tail(self):
        reference = self.path / 'original.f32.bin'
        original = self.run_solver(output=reference, scratch=self.path / 'original-scratch',
                                   overrides={'PI_LOOP1_WHOLE_DAG': '0'})
        self.assertEqual(original.returncode, 0, original.stdout + original.stderr)
        self.assertEqual(reference.read_bytes(), struct.pack('<f', 0.0) * 2)
        for tail in ['0', '1']:
            with self.subTest(compiled_tail=tail):
                self.value.write_bytes(self.accepted)
                scratch = self.path / f'dag-tail-{tail}'
                result = self.run_solver(compare=reference, scratch=scratch,
                                         overrides={'PI_LOOP1_WHOLE_L9_TAIL': tail})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('whole-dag', result.stderr)
                self.assertIn('whole-dag verified roots=2 finite=1 bitwise_match=1', result.stderr)
                self.assertEqual(self.value.read_bytes(), reference.read_bytes())
                self.assert_no_numeric_layer_files(scratch)

    def test_additional_l0_field_is_evaluated_after_the_reset_prefix(self):
        # L0 may contain other fields after the first physical-empty reset
        # block. The input/output boundary contains all three stored rows.
        prefix = self.pruned / 'layer0q_pruned'
        prefix.with_suffix('.field_index.bin').write_bytes(
            b'FCHNIDX1' + struct.pack('<IIQ', 1, 2, 3)
            + struct.pack('<IIQ', 0, 2, 0) + struct.pack('<IIQ', 1, 1, 2))
        prefix.with_suffix('.chains.u24.bin').write_bytes(
            (0).to_bytes(3, 'little') + (1).to_bytes(3, 'little') + (0).to_bytes(3, 'little'))
        (self.policy / 'layer0.policy.u16.bin').write_bytes(b'\xff\xff' * 3)
        self.accepted = struct.pack('<f', 123.0) * 3
        self.value.write_bytes(self.accepted)
        initial = self.path / 'three-root-input.f32.bin'
        initial.write_bytes(struct.pack('<fff', -2.25, 5.0, 91.0))
        reference = self.path / 'three-root-original.f32.bin'
        original = self.run_solver(initial=initial, output=reference,
                                   scratch=self.path / 'three-root-original',
                                   overrides={'PI_LOOP1_WHOLE_DAG': '0'})
        self.assertEqual(original.returncode, 0, original.stdout + original.stderr)
        self.assertEqual(reference.read_bytes(), struct.pack('<f', 0.0) * 3)
        scratch = self.path / 'three-root-dag'
        result = self.run_solver(initial=initial, compare=reference, scratch=scratch)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('whole-dag verified roots=3 finite=1 bitwise_match=1', result.stderr)
        self.assertEqual(self.value.read_bytes(), reference.read_bytes())
        self.assert_no_numeric_layer_files(scratch)

    def test_gmres_uses_raw_numeric_full_root_reference_before_publication(self):
        scratch = self.path / 'gmres-whole'
        result = self.run_solver('gmres', iterations=1_000_000, scratch=scratch, overrides={
            'PI_LOOP1_COMPILED_L9': '1', 'FULL_MDP_L9_DIRECT': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('krylov_iters=0', result.stderr)
        self.assertIn('whole-dag numeric-reference roots=2 gamma=1 compiled_l9=0 dag=0', result.stderr)
        self.assertIn('whole-dag verified roots=2 finite=1 bitwise_match=1', result.stderr)
        self.assertIn('final verification affine T_pi(v) (raw numeric all-boundary, DAG bit comparison)',
                      result.stderr)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0) * 2)
        self.assert_no_numeric_layer_files(scratch)

    def test_gmres_nonzero_warmstart_completes_one_krylov_step_and_verifies_publication(self):
        # NONE policies give b=M=0, hence A=I. A single coordinate produces an
        # exact unit Krylov vector and must converge in one step, rather than
        # taking the already-converged warmstart branch.
        initial = self.path / 'one-coordinate-warmstart.f32.bin'
        initial.write_bytes(struct.pack('<ff', -1.0, 0.0))
        scratch = self.path / 'gmres-one-krylov-step'
        result = self.run_solver('gmres', initial=initial, iterations=1, scratch=scratch,
                                 overrides={'PI_GMRES_AITKEN': '0',
                                            'PI_GMRES_BASIS_MAX_GIB': '1'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('pi-policy-eval-gmres iter=1 ', result.stderr)
        self.assertIn('whole-dag numeric-reference roots=2 gamma=1 compiled_l9=0 dag=0', result.stderr)
        self.assertIn('whole-dag verified roots=2 finite=1 bitwise_match=1', result.stderr)
        self.assertIn('final verification affine T_pi(v) (raw numeric all-boundary, DAG bit comparison)',
                      result.stderr)
        done = next(line for line in result.stderr.splitlines()
                    if line.startswith('pi-policy-eval-gmres done '))
        self.assertIn('status=CONVERGED', done)
        self.assertIn('krylov_iters=1', done)
        self.assertEqual(float(done.split('|T(v)-v|_inf=', 1)[1].split()[0]), 0.0)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0) * 2)
        self.assert_no_numeric_layer_files(scratch)

    def test_final_true_residual_rejection_preserves_accepted_output(self):
        initial = self.path / 'nonconverged.f32.bin'
        initial.write_bytes(struct.pack('<f', -1.0) * 2)
        scratch = self.path / 'gmres-rejected'
        result = self.run_solver('gmres', initial=initial, iterations=0, scratch=scratch)
        self.assert_not_published(result)
        self.assertIn('whole-dag numeric-reference roots=2 gamma=1 compiled_l9=0 dag=0', result.stderr)
        self.assertIn('whole-dag verified roots=2 finite=1 bitwise_match=1', result.stderr)
        self.assertIn('NOT ACCEPTED', result.stderr)
        self.assert_no_numeric_layer_files(scratch)

    def test_full_policy_validation_rejects_unreachable_corrupt_policy_and_membership(self):
        # L0 NONE actions reach neither source. Production must validate all
        # ten stored layers before compiling the all-root operator.
        cases = [
            (self.policy / 'layer9.policy.u16.bin', struct.pack('<H', 0x0700),
             'invalid policy'),
            (self.policy / 'layer6.policy.u16.bin', struct.pack('<H', 0x0700),
             'invalid policy piece layer=6'),
            ((self.pruned / 'layer5q_pruned').with_suffix('.chains.u24.bin'),
             QSTATE_COUNT.to_bytes(3, 'little'), 'invalid membership at layer 5'),
        ]
        for path, malformed, diagnostic in cases:
            with self.subTest(source=path.name):
                original = path.read_bytes()
                path.write_bytes(malformed)
                try:
                    result = self.run_solver('gmres')
                    self.assert_not_published(result)
                    self.assertIn(diagnostic, result.stderr)
                    self.assertNotIn('computing c_pi', result.stderr)
                    self.assert_no_numeric_layer_files(self.path / 'scratch')
                finally:
                    path.write_bytes(original)

    def test_strict_compare_rejects_signed_zero_difference_without_publishing(self):
        reference = self.path / 'negative-zero.f32.bin'
        reference.write_bytes(struct.pack('<f', -0.0) * 2)
        result = self.run_solver(compare=reference)
        self.assert_not_published(result)
        self.assertIn('bitwise', result.stderr)
        self.assertIn('output was not published', result.stderr)
        self.assert_no_numeric_layer_files(self.path / 'scratch')

    def test_default_legacy_mode_still_supports_discounting(self):
        # Invalid inactive DAG tuning must not change the established solver.
        result = self.run_solver(gamma='0.5', overrides={
            'PI_LOOP1_WHOLE_DAG': '0', 'PI_LOOP1_WHOLE_MAX_NODES': 'inactive-setting'})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('whole-dag ready roots=', result.stderr)
        self.assertNotIn('whole-dag apply roots=', result.stderr)
        self.assertNotIn('whole-dag numeric-reference roots=', result.stderr)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0) * 2)
        self.assertTrue((self.path / 'scratch' / 'layer9.values.f32.bin').exists())

    def test_physical_empty_l0_guard_preserves_accepted_output(self):
        self.write_membership(0, [0, 1], field=PC_TERMINAL_FIELD_ID)
        result = self.run_solver()
        self.assert_not_published(result)
        self.assertIn('empty', result.stderr.lower())
        self.assert_no_numeric_layer_files(self.path / 'scratch')

    def test_initial_value_may_share_the_accepted_output_path(self):
        for command in ['apply', 'gmres']:
            with self.subTest(command=command):
                # The initial file is immutable until the authorized atomic
                # publication replaces it. GMRES starts converged to skip basis.
                self.value.write_bytes(struct.pack('<f', 123.0 if command == 'apply' else 0.0) * 2)
                scratch = self.path / f'same-input-output-{command}'
                result = self.run_solver(command, initial=self.value, iterations=1_000_000,
                                         scratch=scratch)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('whole-dag verified roots=2 finite=1 bitwise_match=1', result.stderr)
                self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0) * 2)
                self.assert_no_numeric_layer_files(scratch)

    def test_scratch_aliases_reject_before_graph_loading_and_preserve_sources(self):
        scratch = self.path / 'scratch'
        scratch.mkdir()
        destination = scratch / 'layer9.values.f32.bin'
        policy = self.policy / 'layer9.policy.u16.bin'
        for source in [self.value, policy]:
            with self.subTest(source=source.name):
                original = source.read_bytes()
                os.link(source, destination)
                try:
                    result = self.run_solver(scratch=scratch)
                    self.assert_not_published(result)
                    self.assertIn('alias', result.stderr)
                    self.assertNotIn('loading graph', result.stderr)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertEqual(destination.read_bytes(), original)
                finally:
                    destination.unlink()


if __name__ == '__main__':
    unittest.main()
