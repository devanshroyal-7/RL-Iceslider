"""
Visualize (s_t, s_t+1) pairs from the IceSlider experience dataset.
Shows a temporary plot with a delay between each pair, then closes.
"""

import pickle
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


def visualize_dataset_pairs(
    experience_path: Union[str, Path],
    num_pairs: int = 15,
    delay_seconds: float = 1.0,
) -> None:
    """
    Load the experience pickle and show the first num_pairs (s_t, s_t1) as a
    temporary plot. Displays one pair at a time with delay_seconds between
    each frame, then closes the plot and returns (no file output).
    """
    experience_path = Path(experience_path)
    if not experience_path.exists():
        raise FileNotFoundError(f"Experience file not found: {experience_path}")

    with open(experience_path, "rb") as f:
        experience = pickle.load(f)

    n = min(num_pairs, len(experience))
    if n == 0:
        print("No samples to visualize.")
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(8, 4))
    ax_left.set_title("s_t")
    ax_right.set_title("s_{t+1}")

    for i in range(n):
        state_t, state_t1, _ = experience[i]
        s_t = np.asarray(state_t).squeeze()
        s_t1 = np.asarray(state_t1).squeeze()
        vmin = min(s_t.min(), s_t1.min())
        vmax = max(s_t.max(), s_t1.max())
        if vmax <= vmin:
            vmax = vmin + 1

        ax_left.clear()
        ax_left.imshow(s_t, cmap="gray", vmin=vmin, vmax=vmax)
        ax_left.set_axis_off()
        ax_left.set_title("s_t")

        ax_right.clear()
        ax_right.imshow(s_t1, cmap="gray", vmin=vmin, vmax=vmax)
        ax_right.set_axis_off()
        ax_right.set_title("s_{t+1}")

        fig.suptitle(f"Pair {i + 1} / {n}")
        plt.draw()
        plt.pause(delay_seconds)

    plt.close(fig)
