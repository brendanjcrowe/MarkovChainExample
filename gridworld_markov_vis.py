"""
gridworld_markov_vis.py

Visualize repeated application of a Markov transition matrix on a 5x5 gridworld
(no actions; just state distribution dynamics).

Requirements:
  - numpy
  - matplotlib

Examples:
  python gridworld_markov_vis.py
  python gridworld_markov_vis.py --steps 300 --interval 50 --mode local --start 0,0
  python gridworld_markov_vis.py --mode random_dense --seed 7
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt


GridPos = Tuple[int, int]


@dataclass(frozen=True)
class Gridworld:
    height: int = 5
    width: int = 5

    @property
    def n_states(self) -> int:
        return self.height * self.width

    def to_index(self, r: int, c: int) -> int:
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError(f"Position out of bounds: {(r, c)} for {self.height}x{self.width}")
        return r * self.width + c

    def to_rc(self, idx: int) -> GridPos:
        if not (0 <= idx < self.n_states):
            raise ValueError(f"Index out of bounds: {idx} for n_states={self.n_states}")
        return divmod(idx, self.width)

    def neighbors4(self, r: int, c: int) -> Iterable[GridPos]:
        # Up, Down, Left, Right within bounds.
        if r - 1 >= 0:
            yield (r - 1, c)
        if r + 1 < self.height:
            yield (r + 1, c)
        if c - 1 >= 0:
            yield (r, c - 1)
        if c + 1 < self.width:
            yield (r, c + 1)


def make_transition_matrix_local(
    gw: Gridworld,
    p_stay: float = 0.10,
    p_neighbor: float = 0.90,
) -> np.ndarray:
    """
    Local transition matrix: from each cell, distribute mass across 4-neighbors
    (uniformly) plus an optional stay probability.

    Returns P with shape (n_states, n_states) where:
      next_dist = dist @ P
    """
    if not (0.0 <= p_stay <= 1.0):
        raise ValueError("p_stay must be in [0,1]")
    if not (0.0 <= p_neighbor <= 1.0):
        raise ValueError("p_neighbor must be in [0,1]")
    if not np.isclose(p_stay + p_neighbor, 1.0):
        raise ValueError("p_stay + p_neighbor must sum to 1")

    n = gw.n_states
    P = np.zeros((n, n), dtype=float)

    for r in range(gw.height):
        for c in range(gw.width):
            s = gw.to_index(r, c)
            P[s, s] += p_stay
            nbrs = list(gw.neighbors4(r, c))
            if not nbrs:
                P[s, s] = 1.0
                continue
            share = p_neighbor / len(nbrs)
            for rr, cc in nbrs:
                P[s, gw.to_index(rr, cc)] += share

    # Numerical sanity.
    P /= P.sum(axis=1, keepdims=True)
    print(P)
    return P


def make_transition_matrix_random_dense(gw: Gridworld, rng: np.random.Generator) -> np.ndarray:
    """
    Dense random transition matrix: each row is a random simplex draw (Dirichlet).
    """
    n = gw.n_states
    alpha = np.ones(n, dtype=float)
    P = rng.dirichlet(alpha=alpha, size=n)
    return P


def make_transition_matrix_sink_uniform(
    gw: Gridworld, sink_idx: int, p_stay: float = 0.20
) -> np.ndarray:
    """
    Neighbor + self transitions with an absorbing sink:
      - The sink state is absorbing: P[sink, sink] = 1
      - Every other state has probability p_stay of staying put
        and (1 - p_stay) spread uniformly across its 4-neighbors.

    Returns P with shape (n_states, n_states) where:
      next_dist = dist @ P
    """
    n = gw.n_states
    if not (0 <= sink_idx < n):
        raise ValueError(f"sink_idx must be in [0, {n-1}]")
    P = np.zeros((n, n), dtype=float)
    for r in range(gw.height):
        for c in range(gw.width):
            s = gw.to_index(r, c)
            if s == sink_idx:
                P[s, s] = 1.0
                continue
            P[s, s] += p_stay
            nbrs = list(gw.neighbors4(r, c))
            share = (1.0 - p_stay) / len(nbrs)
            for rr, cc in nbrs:
                P[s, gw.to_index(rr, cc)] += share
    print(P)
    return P


def make_transition_matrix_regular_dense(
    gw: Gridworld,
    rng: np.random.Generator,
    epsilon: float = 0.005,
    p_stay: float = 4.0,
    neighbor_weight: float = 20.0,
) -> np.ndarray:
    """
    Regular (primitive) transition matrix biased toward neighbors:
      - Start with a base weight for self-loops and heavy weights on 4-neighbors.
      - Add small random perturbation to every entry so no two rows look the same.
      - Mix with a tiny uniform floor (epsilon) to guarantee all entries > 0.

    Result: clearly non-uniform rows that favour staying or moving to neighbors,
    yet every state is reachable from every other => unique steady state.
    """
    if not (0.0 < epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")
    n = gw.n_states

    # Build a neighbor-biased base matrix.
    Q = np.zeros((n, n), dtype=float)
    for r in range(gw.height):
        for c in range(gw.width):
            s = gw.to_index(r, c)
            Q[s, s] += p_stay
            for rr, cc in gw.neighbors4(r, c):
                Q[s, gw.to_index(rr, cc)] += neighbor_weight
    # Add random noise so the matrix isn't symmetric / boring.
    Q += rng.exponential(scale=0.05, size=(n, n))
    # Row-normalise to get a stochastic matrix.
    Q /= Q.sum(axis=1, keepdims=True)

    # Mix with a uniform floor to ensure strict positivity.
    U = np.full((n, n), 1.0 / n, dtype=float)
    P = (1.0 - epsilon) * Q + epsilon * U
    P /= P.sum(axis=1, keepdims=True)
    print(P)
    return P


def parse_start(start: str, gw: Gridworld) -> np.ndarray:
    """
    start can be:
      - "uniform" for uniform distribution over all states
      - "r,c" (0-indexed) for a one-hot distribution
      - "r1,c1;r2,c2;..." for a small subset of cells (uniform over that subset)
      - "i" or "i1;i2;..." for flat indices in [0, n_states-1]
    """
    start = start.strip().lower()
    n = gw.n_states
    if start == "uniform":
        return np.ones(n, dtype=float) / n

    # Allow semicolon-separated list of coordinates or indices.
    tokens = [tok.strip() for tok in start.split(";") if tok.strip()]
    if not tokens:
        raise ValueError(
            "start must be 'uniform', 'r,c', 'r1,c1;r2,c2;...', 'i', or 'i1;i2;...'"
        )

    # indices: list[int] = []
    # for tok in tokens:
    #     if "," in tok:
    #         parts = tok.split(",")
    #         if len(parts) != 2:
    #             raise ValueError(
    #                 "Each coordinate token must be 'r,c' with exactly one comma."
    #             )
    #         r, c = int(parts[0]), int(parts[1])
    #         indices.append(gw.to_index(r, c))
    #     else:
    #         idx = int(tok)
    #         if not (0 <= idx < n):
    #             raise ValueError(f"Index {idx} out of bounds for n_states={n}")
    #         indices.append(idx)

    indices = [ 7, 10, 21]
    # Uniform over the (unique) selected states.
    unique = sorted(set(indices))
    dist = np.zeros(n, dtype=float)
    mass = 1.0 / len(unique)
    for idx in unique:
        dist[idx] = mass
    return dist


def parse_sink(sink: str, gw: Gridworld) -> int:
    """
    sink can be:
      - "r,c" (0-indexed)
      - "idx" (integer index in [0, n_states-1])
    """
    sink = sink.strip().lower()
    if "," in sink:
        parts = sink.split(",")
        if len(parts) != 2:
            raise ValueError("sink must be 'r,c' or an integer index")
        r, c = int(parts[0]), int(parts[1])
        return gw.to_index(r, c)
    return int(sink)


def format_prob(p: float) -> str:
    # Compact display that stays readable in small cells.
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.01:
        return f"{p:.3f}"
    return f"{p:.1e}"


def animate_distribution(
    gw: Gridworld,
    P: np.ndarray,
    dist0: np.ndarray,
    steps: int,
    interval_ms: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    power: float,
) -> None:
    dist = dist0.astype(float, copy=True)
    dist /= dist.sum()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("5x5 Gridworld — state distribution under Markov transitions")
    ax.set_xticks(np.arange(-0.5, gw.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, gw.height, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    def dist_to_grid(d: np.ndarray) -> np.ndarray:
        g = d.reshape(gw.height, gw.width)
        if power != 1.0:
            # Purely for visualization (makes small probabilities more visible).
            g = np.power(g, power)
        return g

    grid = dist_to_grid(dist)
    im = ax.imshow(grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Probability (optionally power-transformed for visibility)")

    texts = []
    for r in range(gw.height):
        for c in range(gw.width):
            t = ax.text(
                c,
                r,
                format_prob(dist[gw.to_index(r, c)]),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )
            texts.append(t)

    # Helpful contrast: flip text color depending on cell brightness.
    def update_text_colors(data2d: np.ndarray) -> None:
        norm = im.norm
        for r in range(gw.height):
            for c in range(gw.width):
                val = data2d[r, c]
                x = float(norm(val)) if norm is not None else float(val)
                idx = r * gw.width + c
                texts[idx].set_color("white" if x > 0.55 else "black")

    update_text_colors(grid)

    for t in range(steps + 1):
        grid = dist_to_grid(dist)
        im.set_data(grid)
        ax.set_xlabel(f"step={t}    sum(dist)={dist.sum():.6f}")
        for r in range(gw.height):
            for c in range(gw.width):
                idx = gw.to_index(r, c)
                texts[r * gw.width + c].set_text(format_prob(dist[idx]))
        update_text_colors(grid)

        fig.canvas.draw_idle()
        plt.pause(max(interval_ms, 1) / 1000.0)

        # Next step.
        dist = dist @ P
        dist /= dist.sum()

    plt.show()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Visualize repeated application of a Markov transition matrix on a 5x5 gridworld."
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of transition steps to visualize.")
    parser.add_argument(
        "--interval",
        type=int,
        default=400,
        help="Milliseconds between frames (uses plt.pause; 200-600 feels good).",
    )
    parser.add_argument(
        "--mode",
        choices=["local", "random_dense", "sink_uniform", "regular_dense"],
        default="local",
        help="How to construct the transition matrix.",
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2,2",
        help=(
            "Initial distribution: 'uniform', 'r,c' (0-indexed), "
            "'r1,c1;r2,c2;...' for a small subset, or 'i1;i2;...' for flat indices. "
            "Default is center (2,2)."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (used for random_dense mode).")
    parser.add_argument(
        "--sink",
        type=str,
        default="0,0",
        help="Sink state for sink_uniform mode: 'r,c' (0-indexed) or integer index. Default 0,0.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="Teleport/uniform mixing for regular_dense mode (must be in (0,1)).",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="coolwarm",
        help="Matplotlib colormap name (default: 'coolwarm' hot–cold palette).",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color scale min (default: auto). Consider 0.0.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color scale max (default: auto). Consider 1.0 for one-hot start.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.0,
        help="Optional power transform for visualization only (e.g. 0.5 to boost small probs).",
    )
    args = parser.parse_args()

    gw = Gridworld(height=5, width=5)

    rng = np.random.default_rng(args.seed)
    if args.mode == "local":
        P = make_transition_matrix_local(gw, p_stay=0.10, p_neighbor=0.90)
        # Hard-coded start distribution (change later if you want).
        dist0 = parse_start("2,2", gw)  # center cell
    elif args.mode == "random_dense":
        P = make_transition_matrix_random_dense(gw, rng=rng)
        # Hard-coded start distribution (change later if you want).
        dist0 = parse_start("uniform", gw)
    elif args.mode == "sink_uniform":
        sink_idx = parse_sink(args.sink, gw)
        P = make_transition_matrix_sink_uniform(gw, sink_idx=sink_idx)
        # Hard-coded start distribution (change later if you want).
        # A small 2x2 block near the middle:
        dist0 = parse_start("1,1;1,2;2,1;2,2", gw)
    else:  # regular_dense
        P = make_transition_matrix_regular_dense(gw, rng=rng, epsilon=args.epsilon)
        # Hard-coded start distribution (change later if you want).
        # A cross shape centered at (2,2):
        dist0 = parse_start("2,2;1,2;3,2;2,1;2,3", gw)

    # Basic sanity checks.
    if P.shape != (gw.n_states, gw.n_states):
        raise RuntimeError("Transition matrix has wrong shape.")
    row_sums = P.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-9):
        raise RuntimeError("Transition matrix rows must sum to 1.")

    animate_distribution(
        gw=gw,
        P=P,
        dist0=dist0,
        steps=max(args.steps, 0),
        interval_ms=max(args.interval, 0),
        cmap=args.cmap,
        vmin=args.vmin,
        vmax=args.vmax,
        power=args.power,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

