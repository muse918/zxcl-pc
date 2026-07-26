# Data and generated artifacts

No external-memory MDP layers are stored in Git. By default scripts use `data/` for
persistent output and `work/` for disposable scratch. Override these locations with
`ZXCL_DATA_DIR` and `ZXCL_SCRATCH_DIR`.

## External input

`graph.bin` is 510,917,451 bytes and contains 15,185,706 canonical field records.
It is published by [randomidiot13/hydra](https://github.com/randomidiot13/hydra) and
downloaded from its
[Google Drive file](https://drive.google.com/file/d/1XEYrDFhatN-McOcpTpWuAYq8ubyanJtq/view?usp=sharing).
Known hashes:

```text
MD5    1490c9bd252d6bfa5c7d00518737be6b
SHA-1  3add9a11996e386db2482763344eb5062a745afa
SHA-256 7dc434e3611684756911480d6d338d37bf8ca5ade263611d651fa8709e0294f4
```

## Layer index

Each pruned layer has two structural files:

- `layerKq_pruned.field_index.bin`: field blocks and cumulative chain offsets
- `layerKq_pruned.chains.u24.bin`: sorted quotient-chain IDs, three bytes each

The position in these files is the state ID. Value and policy arrays are raw,
position-aligned little-endian arrays:

- `layerK.values.f32.bin`: four bytes per state
- `layerK.policy.u16.bin`: two bytes per state

Do not reorder, filter, or combine layer files after values have been generated.

Each policy entry encodes the selected action as follows:

```text
high byte = placed piece ID (0..6)
low byte  = index into graph.edges(field, placed_piece)
0xFFFF    = no available action
```

To use a value or policy entry, first locate `(field_id, quotient_chain_id)` in the
matching pruned layer index. Its position is the offset into both raw arrays. Policy
files are therefore only meaningful together with the `graph.bin` and
`forward_pruned/` build that produced them.

## Bundled layer-0 V*

`precomputed/vstar_layer0.f32.bin` contains 817,740 position-aligned little-endian
f32 values for the layer-0 boundary index:

```text
Size    3,270,960 bytes
SHA-256 40fece5d8b8f5ea4c57e69624f91ee9fb20a2be7a61093d5dbbd9889d4c5fff1
```

It can be expanded into all ten value layers with `scripts/solve_mdp.sh from-vstar`.
The bundled vector is index-dependent: it must be used with the documented
`graph.bin` and a `forward_pruned/` build produced from that graph by this repository.
