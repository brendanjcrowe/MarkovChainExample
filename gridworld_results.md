# 5×5 Gridworld — Markov Transition Visualisations

Two transition matrices applied to an initial state distribution over 50 time-steps.

---

## 1. Sink-Uniform (neighbour transitions + absorbing sink)

- **Sink** at cell (0, 0) — once probability enters, it never leaves.
- Every other state transitions **uniformly to its 4-neighbours** (+ 20 % self-loop).
- Initial distribution: uniform over a 2×2 block near the centre.

### Start (step 0)

![sink_uniform start](sink_uniform_start.png)

### End (step 50)

![sink_uniform end](sink_uniform_end.png)

### Animation

![sink_uniform animation](sink_uniform.gif)

---

## 2. Regular-Dense (ergodic, non-uniform)

- All entries strictly > 0, so every state is reachable from every other.
- Heavily **biased toward neighbours and self-loops** — not uniform.
- Guaranteed to converge to a **unique stationary distribution**.
- Initial distribution: cross shape centred at (2, 2).

### Start (step 0)

![regular_dense start](regular_dense_start.png)

### End (step 50)

![regular_dense end](regular_dense_end.png)

### Animation

![regular_dense animation](regular_dense.gif)
