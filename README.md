# zxcl

`zxcl` is a Rust project I made to build and solve a Tetris Perfect Clear MDP.
It starts from `graph.bin`, builds the reachable state layers, finds a policy with
Policy Iteration + GMRES, and can write the value table for every layer.

This repository only contains that pipeline. It does not include a bot, game client,
viewer, or gameplay frontend.

## What's here

| Path | Contents |
|---|---|
| `src/` | MDP builder and solver |
| `scripts/` | Restartable build and solve commands |
| `config/` | Example settings for 32 GiB and 128 GiB machines |
| `precomputed/` | The layer-0 `V*` file from my run |
| `docs/` | Build notes and file formats |

## Before running it

The full job is large. My run used 128 GiB RAM, fast NVMe scratch space, several
hundred GiB of persistent storage, and took several days. A 32 GiB profile is included,
but it relies much more heavily on disk spill and will be slower.

See [docs/BUILD_MDP.md](docs/BUILD_MDP.md) for storage estimates and the available
environment variables.

## Quick start

```bash
git clone https://github.com/muse918/zxcl-pc.git
cd zxcl-pc
cargo test --workspace
./scripts/download_graph.sh
```

`graph.bin` is downloaded separately. The script gets the file published with
[randomidiot13/hydra](https://github.com/randomidiot13/hydra) and checks its size and
hashes. You can also download it from the original
[Google Drive link](https://drive.google.com/file/d/1XEYrDFhatN-McOcpTpWuAYq8ubyanJtq/view?usp=sharing)
and put it at `data/graph.bin`.

Choose storage paths and load a hardware profile:

```bash
export ZXCL_DATA_DIR=/mnt/large/zxcl-pc
export ZXCL_SCRATCH_DIR=/mnt/nvme/zxcl-pc
source config/reference-128gb.env
```

Build the MDP:

```bash
./scripts/build_mdp.sh prepare
./scripts/build_mdp.sh backward
./scripts/build_mdp.sh prune
```

Then solve it:

```bash
./scripts/solve_mdp.sh one-loop
./scripts/solve_mdp.sh solve
./scripts/solve_mdp.sh finalize
```

Both scripts also accept `all`. Completed stages are kept so a long run can be
restarted. Use `./scripts/build_mdp.sh env` and `./scripts/solve_mdp.sh env` to see the
current paths and tuning values.

## Using the included layer-0 V*

If you only want to rebuild the full value layers and policy, first build the MDP and
then use the included layer-0 file:

```bash
./scripts/build_mdp.sh all
./scripts/solve_mdp.sh from-vstar
```

This creates the files under `data/results/` by default:

- `vstar_boundary.f32.bin`: the included layer-0 `V*`
- `vstar_full/layerK.values.f32.bin`: values for layers 0-9
- `policy_final/layerK.policy.u16.bin`: the best placement recorded for each state

The files are raw arrays aligned with the matching `forward_pruned/` index; they are
not a standalone bot or lookup API. See [docs/DATA.md](docs/DATA.md) before reading
them from another program. Set `RESULTS_DIR` to write them somewhere else.

The pipeline is roughly:

```text
graph.bin
  -> rgraph / qgraph
  -> backward layers
  -> forward prune
  -> layer 0-3 u16 starting value
  -> Policy Iteration + GMRES
  -> stop when the policy no longer changes
  -> V* for layers 0-9
```

Some useful numbers from my run:

- 15,185,706 fields in `graph.bin`
- 817,740 layer-0 states
- about 86.3 billion states across layers 0-9
- mean layer-0 `V*` of about 4,338.53 PCs

These are reference results for spotting an incomplete run. The current graph reader
still expects the reference field count and three-byte target IDs; loading a different
field catalog needs a separate compatibility change.

## License

GPL-3.0-only. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
