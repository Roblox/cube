from dataclasses import dataclass, field
import torch

@dataclass
class Cache:
    key_states: torch.Tensor
    value_states: torch.Tensor
    _supports_index_copy: bool = field(init=False) # For CUDA graph support

    def __post_init__(self):
        self._supports_index_copy = self._check_index_copy_support()

    def _check_index_copy_support(self) -> bool:
        """Verifies support for `index_copy_` on device."""
        try:
            device = self.key_states.device
            dummy = torch.tensor([0, 0], device=device)
            dummy.index_copy_(0, torch.tensor([0], device=device), torch.tensor([1], device=device))
            return True
        except NotImplementedError:
            return False

    def update(self, curr_pos_id: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Updates the cache based on device operator support.
        Args:
            curr_pos_id (torch.Tensor): Current position indices for decoding.
            k (torch.Tensor): The keys to update
            v (torch.Tensor): The values to update
        """
        if self._supports_index_copy: # CUDA/CPU
            self.key_states.index_copy_(2, curr_pos_id, k)
            self.value_states.index_copy_(2, curr_pos_id, v)
        else: # MPS
            self.key_states[:, :, curr_pos_id:curr_pos_id +1, ...].copy_(k)
            self.value_states[:, :, curr_pos_id:curr_pos_id +1, ...].copy_(v)