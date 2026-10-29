#!/usr/bin/env python3
"""Actual full-L8 solver dispatch and publication checks with tiny membership.

Set ZXCL_RUNTIME_SOLVER and GRAPH_BIN to use a built binary and the read-only
production graph. The NONE fixture isolates CLI selection, complete-layer
compilation, sink choice, and failure/publication ordering. Signed nonzero
arithmetic through all ten layers is checked by full_l8_tests in Rust.
"""
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('ZXCL_RUNTIME_SOLVER') and os.environ.get('GRAPH_BIN'),
                     'set ZXCL_RUNTIME_SOLVER and GRAPH_BIN for actual solver runtime checks')
class FullL8RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zxcl-full-l8-runtime-')
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
        self.env = {key: value for key, value in os.environ.items()
                    if not key.startswith(('PI_', 'FULL_MDP_'))}
        self.env.update(
            RAYON_NUM_THREADS='1', PI_LOOP1_COMPILED_L9='1', PI_LOOP1_COMPILED_L8='1',
            PI_GMRES_AITKEN='0', PI_ACCEPT_BELLMAN_INF='0.001', PI_GMRES_BASIS_MAX_GIB='0',
            FULL_MDP_MEMORY_MAX_GIB='1', FULL_MDP_MEMORY_RESERVE_GIB='0',
            FULL_MDP_RAM_NEXT_LAYER='0', FULL_MDP_BLOCK_CHUNK='1', FULL_MDP_BATCH_CHUNKS='1',
            PI_LOOP1_L8_MAX_MAPPING_MIB='16', PI_LOOP1_L8_BUILD_CACHE_MIB='1',
            PI_LOOP1_L8_BUILD_MAX_MIB='256', PI_LOOP1_L8_MAX_UNIQUE_AVERAGES='100')

    def tearDown(self):
        self.tmp.cleanup()

    def apply(self, *, output=None, scratch=None, compare=None, gamma='1', graph=None,
              cache_gib='0', overrides=None):
        env = dict(self.env)
        if overrides:
            env.update(overrides)
        args = [os.environ['ZXCL_RUNTIME_SOLVER'], 'apply-fixed-policy-boundary',
                str(graph or os.environ['GRAPH_BIN']), str(self.pruned), str(self.policy),
                'zero', str(output or self.value), str(scratch or self.path / 'scratch'), gamma, cache_gib]
        if compare is not None:
            args.append(str(compare))
        return subprocess.run(args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, timeout=180)

    def test_single_apply_both_compilers_match_original_in_file_ram_and_mmap(self):
        reference = self.path / 'original.f32.bin'
        original = self.apply(output=reference, scratch=self.path / 'original-scratch', overrides={
            'PI_LOOP1_COMPILED_L9': '0', 'PI_LOOP1_COMPILED_L8': '0'})
        self.assertEqual(original.returncode, 0, original.stdout + original.stderr)
        self.assertEqual(reference.read_bytes(), struct.pack('<f', 0.0))
        for mode in ['file', 'ram', 'mmap']:
            with self.subTest(mode=mode):
                self.value.write_bytes(self.accepted)
                scratch = self.path / f'scratch-{mode}'
                result = self.apply(scratch=scratch, compare=reference, overrides={
                    'FULL_MDP_L9_DIRECT': '1',
                    'FULL_MDP_RAM_NEXT_LAYER': '1' if mode == 'ram' else '0',
                    'FULL_MDP_RAM_NEXT_LAYER_MAX_GIB': '1',
                    'FULL_MDP_RAM_PAIR_MAX_GIB': '1',
                    'FULL_MDP_MMAP_SPILL': '1' if mode == 'mmap' else '0',
                })
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('compiled-l9 states=1', result.stderr)
                self.assertIn('compiled-l8 states=1', result.stderr)
                self.assertIn('sink=compiled-l8-leaves', result.stderr)
                self.assertIn(f'sink=compiled-{mode}-gather', result.stderr)
                self.assertIn('differing=0', result.stderr)
                self.assertEqual(self.value.read_bytes(), reference.read_bytes())
                self.assertFalse((scratch / 'layer9.values.f32.bin').exists())
                if mode == 'ram':
                    self.assertFalse((scratch / 'layer8.values.f32.bin').exists())
                else:
                    self.assertEqual((scratch / 'layer8.values.f32.bin').read_bytes(),
                                     struct.pack('<f', 0.0))

    def test_gmres_l8_implicitly_enables_l9_and_uses_original_final_gate(self):
        env = dict(self.env, PI_LOOP1_COMPILED_L9='0', FULL_MDP_L9_DIRECT='1')
        result = subprocess.run(
            [os.environ['ZXCL_RUNTIME_SOLVER'], 'pi-policy-eval-gmres', os.environ['GRAPH_BIN'],
             str(self.pruned), str(self.policy), 'zero', str(self.value), str(self.path / 'scratch'),
             '1', '1000000', '1e-6', '0'], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=180)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('compiled-l9 states=1', result.stderr)
        self.assertIn('compiled-l8 states=1', result.stderr)
        self.assertIn('krylov_iters=0', result.stderr)
        gate = 'final verification affine sweep T_pi(v) (original uncompiled)'
        self.assertIn(gate, result.stderr)
        before_gate, final_gate = result.stderr.split(gate, 1)
        self.assertIn('sink=compiled-file-gather', before_gate)
        self.assertNotIn('sink=compiled-', final_gate)
        self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0))
        # Only the original final gate can create the L9 value sink. The L8
        # compiled path also has to yield to that raw ten-layer verification.
        self.assertEqual((self.path / 'scratch' / 'layer9.values.f32.bin').read_bytes(),
                         struct.pack('<f', 0.0))
        self.assertEqual((self.path / 'scratch' / 'layer8.values.f32.bin').read_bytes(),
                         struct.pack('<f', 0.0))

    def test_action_cache_only_matches_discounted_original_and_original_gmres_gate(self):
        # The action cache supports discounting independently of the gamma=1
        # compiled terminal paths. Use enough task cache to allocate real slots,
        # rather than only exercising a disabled-cache fallback.
        settings = {'PI_LOOP1_COMPILED_L9': '0', 'PI_LOOP1_COMPILED_L8': '0',
                    'PI_LOOP1_ACTION_CACHE_MIB': '1', 'FULL_MDP_CACHE_STATS': '1'}
        reference = self.path / 'discounted-original.f32.bin'
        original = self.apply(output=reference, scratch=self.path / 'discounted-original',
                              gamma='0.5', cache_gib='1', overrides=dict(
                                  settings, PI_LOOP1_ACTION_CACHE_MIB='0'))
        self.assertEqual(original.returncode, 0, original.stdout + original.stderr)
        self.assertEqual(reference.read_bytes(), struct.pack('<f', 0.0))
        result = self.apply(scratch=self.path / 'discounted-cached', compare=reference,
                            gamma='0.5', cache_gib='1', overrides=settings)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn('compiled-l9', result.stderr)
        self.assertNotIn('compiled-l8', result.stderr)
        self.assertRegex(result.stderr, r'action_avg_cache_bytes=[1-9][0-9]*')
        self.assertIn('differing=0', result.stderr)
        self.assertEqual(self.value.read_bytes(), reference.read_bytes())

        env = dict(self.env, **settings)
        scratch = self.path / 'discounted-gmres'
        gmres = subprocess.run(
            [os.environ['ZXCL_RUNTIME_SOLVER'], 'pi-policy-eval-gmres', os.environ['GRAPH_BIN'],
             str(self.pruned), str(self.policy), 'zero', str(self.value), str(scratch),
             '0.5', '1000000', '1e-6', '1'], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=180)
        self.assertEqual(gmres.returncode, 0, gmres.stdout + gmres.stderr)
        self.assertNotIn('compiled-l9', gmres.stderr)
        self.assertNotIn('compiled-l8', gmres.stderr)
        gate = 'final verification affine sweep T_pi(v) (original uncompiled)'
        self.assertIn(gate, gmres.stderr)
        before_gate, final_gate = gmres.stderr.split(gate, 1)
        self.assertRegex(before_gate, r'action_avg_cache_bytes=[1-9][0-9]*')
        final_layers = [line for line in final_gate.splitlines() if 'fixed-policy layer=' in line]
        self.assertEqual(len(final_layers), 10, gmres.stderr)
        for line in final_layers:
            self.assertIn('action_avg_cache_bytes=0', line)
        self.assertIn('krylov_iters=0', gmres.stderr)
        self.assertEqual(self.value.read_bytes(), reference.read_bytes())

    def test_action_cache_only_rejects_corrupt_policy_without_publishing_output(self):
        # This policy is unreachable from the NONE-policy boundary, but every
        # stored state still belongs to the fixed operator and must be checked.
        (self.policy / 'layer8.policy.u16.bin').write_bytes(struct.pack('<H', 0x0700))
        result = self.apply(gamma='0.5', cache_gib='1', overrides={
            'PI_LOOP1_COMPILED_L9': '0', 'PI_LOOP1_COMPILED_L8': '0',
            'PI_LOOP1_ACTION_CACHE_MIB': '1'})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('corrupt policy code 0x0700 layer=8', result.stderr)
        self.assertNotIn('compiled-l9', result.stderr)
        self.assertNotIn('compiled-l8', result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_mapping_and_compiler_heap_caps_fail_before_sweep_and_preserve_output(self):
        scratch = self.path / 'scratch'
        scratch.mkdir()
        markers = {}
        for layer in [0, 8, 9]:
            path = scratch / f'layer{layer}.values.f32.bin'
            markers[path] = b'accepted scratch bytes'
            path.write_bytes(markers[path])
        for cap, expected in [('PI_LOOP1_L8_MAX_MAPPING_MIB', 'mapping byte cap'),
                              ('PI_LOOP1_L8_BUILD_MAX_MIB', 'capacity')]:
            with self.subTest(cap=cap):
                result = self.apply(overrides={cap: '0'})
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(expected, result.stderr)
                self.assertNotIn('fixed-policy layer=', result.stderr)
                self.assertEqual(self.value.read_bytes(), self.accepted)
                for path, marker in markers.items():
                    self.assertEqual(path.read_bytes(), marker)

    def test_unreachable_malformed_l8_policy_is_rejected_during_full_compilation(self):
        (self.policy / 'layer8.policy.u16.bin').write_bytes(struct.pack('<H', 0x0700))
        result = self.apply()
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('invalid policy placed piece 7', result.stderr)
        self.assertNotIn('fixed-policy layer=', result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_bitwise_compare_failure_preserves_accepted_output_even_for_signed_zero(self):
        reference = self.path / 'negative-zero.f32.bin'
        reference.write_bytes(struct.pack('<f', -0.0))
        result = self.apply(compare=reference)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('1 bitwise differences', result.stderr)
        self.assertIn('output was not published', result.stderr)
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_l8_gamma_is_rejected_before_graph_loading_or_scratch_creation(self):
        result = self.apply(gamma='0.5', graph=self.path / 'absent-graph.bin', overrides={
            'PI_LOOP1_COMPILED_L9': '0'})
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('PI_LOOP1_COMPILED_L8', result.stderr)
        self.assertIn('gamma', result.stderr)
        self.assertNotIn('loading graph', result.stderr)
        self.assertFalse((self.path / 'scratch').exists())
        self.assertEqual(self.value.read_bytes(), self.accepted)

    def test_complete_empty_l8_does_not_mmap_an_empty_file_or_leave_a_stale_pin(self):
        prefix = self.pruned / 'layer8q_pruned'
        prefix.with_suffix('.field_index.bin').write_bytes(b'FCHNIDX1' + struct.pack('<IIQ', 1, 0, 0))
        prefix.with_suffix('.chains.u24.bin').write_bytes(b'')
        (self.policy / 'layer8.policy.u16.bin').write_bytes(b'')
        for mode in ['file', 'ram', 'mmap']:
            with self.subTest(mode=mode):
                result = self.apply(scratch=self.path / f'empty-{mode}', overrides={
                    'FULL_MDP_RAM_NEXT_LAYER': '1' if mode == 'ram' else '0',
                    'FULL_MDP_MMAP_SPILL': '1' if mode == 'mmap' else '0'})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('compiled-l8 states=0', result.stderr)
                self.assertIn('sink=compiled-empty', result.stderr)
                self.assertEqual((self.path / f'empty-{mode}' / 'layer8.values.f32.bin').read_bytes(), b'')
                self.assertEqual(self.value.read_bytes(), struct.pack('<f', 0.0))


if __name__ == '__main__':
    unittest.main()
