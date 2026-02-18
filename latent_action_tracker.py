from typing import Set, Tuple
import numpy as np
import torch

class LatentStateTracker:
    def __init__(self):
        self.visited_states: dict[Tuple[float, ...], Set[int]] = {}
        self.distance_threshold = 0.1
    
    def _hash_latent_state(self, latent_vector: torch.Tensor) -> Tuple[float, ...]:
        if isinstance(latent_vector, torch.Tensor):
            latent_array = latent_vector.cpu().numpy().flatten()
        else:
            latent_array = np.array(latent_vector).flatten()
        return tuple(np.round(latent_array, decimals=2))
    
    def has_visited(self, latent_vector: torch.Tensor, action: int) -> bool:
        """
        Docstring for has_visited
        Doesn't just check the current state, but neighboring states too
        """
        latent_key = self._hash_latent_state(latent_vector)

        neighboring_states = self.get_neighboring_states(latent_key)
        
        return len(neighboring_states) > 0

    def record_action(self, latent_vector: torch.Tensor, action: int):
        latent_key = self._hash_latent_state(latent_vector)
        if latent_key not in self.visited_states:
            self.visited_states[latent_key] = set()
        self.visited_states[latent_key].add(action)
    
    def get_visited_actions(self, latent_vector: torch.Tensor) -> Set[int]:
        latent_key = self._hash_latent_state(latent_vector)
        neighboring_states = self.get_neighboring_states(latent_key)
        visited_actions = set()
        for state in neighboring_states:
            visited_actions.update(self.visited_states[state])
        return visited_actions

    def get_latent_key(self, latent_vector: torch.Tensor) -> Tuple[float, ...]:
        return self._hash_latent_state(latent_vector)

    def get_neighboring_states(self, latent_key: np.ndarray) -> list[Tuple[float, ...]]:
        neighboring_states = []
        latent_array = np.array(latent_key)

        for key in self.visited_states:
            key_array = np.array(key)
            distance = np.linalg.norm(latent_array - key_array)
            if distance < self.distance_threshold/2:
                neighboring_states.append(key)
        
        return neighboring_states

    def reset(self):
        self.visited_states.clear()

