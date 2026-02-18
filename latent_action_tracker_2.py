from typing import Set, Tuple
import numpy as np
import torch
import pdb

class LatentStateTracker:
    def __init__(self):
        self.visited_states= np.empty((0, 16))
        self.action_table: dict[int, set[int]] = {}
        self.action_count: dict[int, list[int]] = {}
        self.distance_threshold = 0.1
        self.state_count: dict[int, int] = dict()
    
    def get_visited_actions(self, latent_vector: torch.Tensor) -> Set[int]:
        visited_actions = set()
        action_count = [0 for i in range(5)]
        neighbor_idx = self.get_neighbors(latent_vector.cpu().numpy().flatten())
        if neighbor_idx.size > 0:
            return self.action_table[neighbor_idx[0]], self.action_count[neighbor_idx[0]]
        return visited_actions, action_count
    
    def record_action(self, latent_vector: torch.Tensor, action: int):
        neighbor_idx = self.get_neighbors(latent_vector.cpu().numpy().flatten())
        if neighbor_idx.size == 0:
            self.visited_states = np.vstack([self.visited_states, latent_vector.cpu().numpy().flatten()])
            idx = len(self.visited_states) - 1
            self.action_table[idx] = {action}
            self.action_count[idx] = [0 for i in range(5)]
            self.update_count(idx, action)
            # breakpoint()
        else:
            self.action_table[neighbor_idx[0]].add(action)
            self.update_count(neighbor_idx[0], action)   

    def update_count(self, idx, action):
        self.action_count[idx][action] += 1

    def get_neighbors(self, latent_np: np.array):
        distances = np.sum((self.visited_states - latent_np)**2, axis=1)   # (?)
        # breakpoint()
        neighbor_idx = np.array([])
        if distances.size:
            nearest = np.argmin(distances)
            if distances[nearest] < (self.distance_threshold/2)**2:
                neighbor_idx = np.append(neighbor_idx, nearest)
        return neighbor_idx     # (1) np array
    
    def has_visited(self, latent_vector: torch.Tensor, action: int) -> bool:
        neighbor_idx = self.get_neighbors(latent_vector.cpu().numpy().flatten())
        return np.any(neighbor_idx)

    def reset(self):
        self.visited_states = np.empty((0, 16))
        self.action_table.clear()
