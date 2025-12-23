from typing import Set, Tuple
import numpy as np
import torch

class LatentStateTracker:
    def __init__(self):
        self.visited_states: dict[Tuple[float, ...], Set[int]] = {}
    
    def _hash_latent_state(self, latent_vector: torch.Tensor) -> Tuple[float, ...]:
        if isinstance(latent_vector, torch.Tensor):
            latent_array = latent_vector.cpu().numpy().flatten()
        else:
            latent_array = np.array(latent_vector).flatten()
        return tuple(np.round(latent_array, decimals=2))
    
    def has_visited(self, latent_vector: torch.Tensor, action: int) -> bool:
        latent_key = self._hash_latent_state(latent_vector)
        if latent_key not in self.visited_states:
            return False
        return action in self.visited_states[latent_key]
    
    def record_action(self, latent_vector: torch.Tensor, action: int):
        latent_key = self._hash_latent_state(latent_vector)
        if latent_key not in self.visited_states:
            self.visited_states[latent_key] = set()
        self.visited_states[latent_key].add(action)
    
    def get_visited_actions(self, latent_vector: torch.Tensor) -> Set[int]:
        latent_key = self._hash_latent_state(latent_vector)
        return self.visited_states.get(latent_key, set())
    
    def get_latent_key(self, latent_vector: torch.Tensor) -> Tuple[float, ...]:
        return self._hash_latent_state(latent_vector)
    
    def reset(self):
        self.visited_states.clear()

