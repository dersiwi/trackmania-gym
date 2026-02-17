import torch
import jax
import jax.numpy as jnp

from torch.utils import dlpack as torch_dlpack
from jax import dlpack as jax_dlpack


class TorchJaxAdapter:
    """
    Bidirectional adapter between PyTorch tensors and JAX arrays
    using DLPack for zero-copy conversion when possible.
    """

    @staticmethod
    def torch_to_jax(tensor: torch.Tensor) -> jax.Array:
        """
        Convert a PyTorch tensor to a JAX array.

        Notes:
        - Zero-copy when tensor is on CPU or CUDA
        - Tensor must not require gradients
        """
        if tensor.requires_grad:
            raise ValueError("Tensor must not require gradients for DLPack conversion")

        dlpack_capsule = torch_dlpack.to_dlpack(tensor)
        return jax_dlpack.from_dlpack(dlpack_capsule)

    @staticmethod
    def jax_to_torch(array: jax.Array) -> torch.Tensor:
        """
        Convert a JAX array to a PyTorch tensor.

        Notes:
        - Zero-copy when array is on CPU or CUDA
        - Resulting tensor will be detached
        """
        dlpack_capsule = jax_dlpack.to_dlpack(array)
        return torch_dlpack.from_dlpack(dlpack_capsule)


from trackmania_env.envs.vectorized import VectorizedTMEnvironment

class JaxVecEnv:
    """Implements an adapter for a given environment; to translate from torch -> jax (essentially exposes jax environemnt)"""
    def __init__(self, env : VectorizedTMEnvironment):
        self.env = env

    def __getattr__(self, name):
        """
        Delegate attribute access to the wrapped environment
        if not found on this wrapper.
        """
        return getattr(self.env, name)

    
    def step(self, action : jax.Array):
        obs, rewards, terminated, truncated, info = self.env.step(TorchJaxAdapter.jax_to_torch(action))
        return TorchJaxAdapter.torch_to_jax(obs), \
                TorchJaxAdapter.torch_to_jax(rewards), \
                TorchJaxAdapter.torch_to_jax(terminated), \
                TorchJaxAdapter.torch_to_jax(truncated), info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return TorchJaxAdapter.torch_to_jax(obs), info