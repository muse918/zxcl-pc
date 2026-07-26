# How zxcl works

This is a development note about the MDP structure and solve order. It is mainly here
so that the code is easier to return to later. For commands and hardware settings, see
[BUILD_MDP.md](BUILD_MDP.md).

## Goal

`zxcl` works out which placements are best when trying to continue Perfect Clears
under the 7-bag rule.

The value `V(s)` of a state `s` is the expected number of future PCs from that state.
The policy chooses the placement that maximizes the expected immediate reward plus the
value of the next state.

```text
V(s) = max_a E[r(s, a, s') + γ V(s')]
π(s) = argmax_a E[r(s, a, s') + γ V(s')]
```

Here, `a` is a placement and `s'` is the state after the placement and next-piece
reveal. Completing a PC adds `r=1` and starts the next section from an empty field.
The completed result included with this project uses `γ=1`.

## State

A state combines a field with the piece information needed to make a decision.

```text
state = (field, piece state)
```

- field: a board state stored in `graph.bin`
- piece state: available pieces, hold, queue, and the pieces left in the current bag

Piece states with the same future choices are merged into quotient chains. Different
movement or rotation paths also count as the same MDP state when they end at the same
field with the same future choices.

## Why about 86.3 billion states remain

`graph.bin` contains 15,185,706 fields, and the code enumerates 1,120,140 valid
hold/queue/bag states before applying the quotient. Their direct Cartesian product is:

```text
15,185,706 fields × 1,120,140 piece states
  = 17,010,116,718,840 states
  ≈ 17.0 trillion states
```

Merging equivalent piece states reduces their count to 817,740, but even that product
is still about 12.4 trillion states:

```text
15,185,706 × 817,740 = 12,417,959,224,440
```

Most field and piece-state pairs can never occur together. The quotient alone is
therefore not enough; the build also needs reachability pruning. It keeps only states
that satisfy both of these conditions:

1. The state is reachable from an empty field under the 7-bag rule.
2. A PC can still be reached from the state with the remaining placements.

The `backward` step first builds the states that can reach a PC. `forward prune` then
walks forward from the empty field and intersects its results with that list. About
86.3 billion states remain across layers 0 through 9.

```text
pruned states = forward reachable ∩ terminal reachable
```

## Why layer 0 is small

States are split into layers according to the number of pieces already placed in the
current section.

- layer 0: an empty field before placing a piece
- layers 1-9: intermediate fields
- after a PC: add reward 1 and return to an empty field

The intermediate layers are very large, but the field is always empty in layer 0.
Only the piece state varies there, leaving 817,740 layer-0 states.

Once a layer-0 value is given, the other layers can be calculated in this order:

```text
layer 9 -> 8 -> ... -> 1 -> 0
```

This means only the 817,740 layer-0 values need to be updated repeatedly, rather than
all 86.3 billion values. Each sweep still reads the intermediate states to calculate
the expected value of their placements.

## Solving

One backward sweep chooses the best placement at each state and produces a new layer-0
value. The selected placements make up the policy.

With the policy fixed, the sweep becomes a linear operation on the layer-0 vector:

```text
T_π(v) = c_π + γ M_π v
v_π = c_π + γ M_π v_π
(I - γ M_π) v_π = c_π
```

GMRES solves the last system. `M_π` is not stored as a giant matrix; a fixed-policy
backward sweep calculates `M_π v` when GMRES asks for it. This keeps the solve
matrix-free.

Policy Iteration repeats the following steps:

1. Build a policy for every layer from the current layer-0 value.
2. Use GMRES to calculate the layer-0 value of that policy.
3. Build a new policy from the new value.
4. Stop when the policy no longer changes.

`MAX_PI_ROUNDS=8` is a safety limit, not a fixed round count. The solver stops earlier
when two consecutive policies are identical.

## Starting value

Before PI begins, a smaller u16 calculation creates a starting layer-0 value. Every
surviving layer-4 state is assigned `5040/5040`, then layers 3, 2, 1, and 0 are
calculated backward.

```text
ONE_LOOP_START_LAYER=3
ONE_LOOP_BASE_FULL_NEXT_LAYER=4
```

This is only a starting point, not the final `V*`. One full f32 sweep turns it into the
seed passed to PI/GMRES.

## Build and solve

The project keeps MDP construction and value solving in separate scripts.

```text
build_mdp.sh
  graph.bin
  -> rgraph / qgraph
  -> backward
  -> forward prune
  -> forward_pruned/

solve_mdp.sh
  forward_pruned/
  -> u16 starting value
  -> Policy Iteration + GMRES
  -> stable layer-0 V*
  -> values and final policy for layers 0-9
```

The sorted order in `forward_pruned/` is the index used by the value and policy files.
Do not mix an index from one build with values produced by another build.

## File sizes

Approximate sizes from the completed build:

| Item | Size |
|---|---:|
| layer-0 states | 817,740 |
| all pruned states | about 86.3 billion |
| largest layer | about 21.7 billion states |
| f32 values for layers 0-9 | about 321 GiB |
| u16 policy for layers 0-9 | about 161 GiB |

The intermediate files are too large to keep in RAM. Large layers are streamed through
NVMe or another scratch device, while the requested final outputs are kept in the
results directory.
