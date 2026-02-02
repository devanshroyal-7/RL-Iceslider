from typing import Set, Tuple
import numpy as np
import torch

class LatentStateTracker:
    def __init__(self):
        self.visited_states= np.empty((0, 16))
        self.action_table: dict[int, set[int]] = {}
        self.distance_threshold = 0.1
    
    def get_visited_actions(self, latent_vector: torch.Tensor) -> Set[int]:
        visited_actions = set()
        neighbor_idx = self.get_neighbors(latent_vector.cpu().numpy().flatten())
        # print(type(neighbor_idx))
        if neighbor_idx.size > 0:
            for idx in list(neighbor_idx):
                visited_actions.update(self.action_table[idx])
        return visited_actions
    
    def record_action(self, latent_vector: torch.Tensor, action: int):
        state_match = np.all(self.visited_states == latent_vector.cpu().numpy().flatten(), axis = 1)
        if not np.any(state_match):
            self.visited_states = np.vstack([self.visited_states, latent_vector.cpu().numpy().flatten()])
            idx = len(self.visited_states) - 1
            self.action_table[idx] = {action}
        else:
            idx = np.where(state_match)[0][0]
            # print(self.action_table)
            self.action_table[idx].add(action)
            
    def get_neighbors(self, latent_np: np.array):
        neighbor_idx = np.where(np.sum((self.visited_states - latent_np)**2, axis=1) < (self.distance_threshold/2)**2)[0]
        return neighbor_idx
    
    def has_visited(self, latent_vector: torch.Tensor, action: int) -> bool:
        neighbor_idx = self.get_neighbors(latent_vector.cpu().numpy().flatten())
        return np.any(neighbor_idx)

    def reset(self):
        self.visited_states = np.empty((0, 16))
        self.action_table.clear()
