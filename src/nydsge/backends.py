from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol

import numpy as np

from nydsge.runtime import ResolvedRuntime, RuntimeConfig


class ArrayBackend(Protocol):
    name: str
    device: str
    dtype: str

    def array(self, value: Any) -> Any: ...

    def zeros(self, shape: tuple[int, ...]) -> Any: ...

    def eye(self, n: int) -> Any: ...

    def matmul(self, left: Any, right: Any) -> Any: ...

    def transpose(self, value: Any) -> Any: ...

    def solve(self, left: Any, right: Any) -> Any: ...

    def slogdet(self, value: Any) -> tuple[Any, Any]: ...

    def scalar(self, value: Any) -> float: ...

    def as_numpy(self, value: Any) -> np.ndarray: ...


@dataclass(frozen=True)
class NumpyBackend:
    dtype: str = "float64"
    name: str = "numpy"
    device: str = "cpu"

    @property
    def _dtype(self) -> np.dtype[Any]:
        return np.dtype(self.dtype)

    def array(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=self._dtype)

    def zeros(self, shape: tuple[int, ...]) -> np.ndarray:
        return np.zeros(shape, dtype=self._dtype)

    def eye(self, n: int) -> np.ndarray:
        return np.eye(n, dtype=self._dtype)

    def matmul(self, left: Any, right: Any) -> np.ndarray:
        return np.matmul(self.array(left), self.array(right))

    def transpose(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=self._dtype).T

    def solve(self, left: Any, right: Any) -> np.ndarray:
        return np.linalg.solve(self.array(left), self.array(right))

    def slogdet(self, value: Any) -> tuple[Any, Any]:
        return np.linalg.slogdet(self.array(value))

    def scalar(self, value: Any) -> float:
        return float(np.asarray(value, dtype=self._dtype))

    def as_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value, dtype=self._dtype)


@dataclass(frozen=True)
class TorchBackend:
    device: str
    dtype: str = "float64"
    name: str = "torch"

    @property
    def _torch(self) -> Any:
        return import_module("torch")

    @property
    def _dtype(self) -> Any:
        torch = self._torch
        return torch.float64 if self.dtype == "float64" else torch.float32

    def array(self, value: Any) -> Any:
        torch = self._torch
        return torch.as_tensor(value, dtype=self._dtype, device=torch.device(self.device))

    def zeros(self, shape: tuple[int, ...]) -> Any:
        torch = self._torch
        return torch.zeros(shape, dtype=self._dtype, device=torch.device(self.device))

    def eye(self, n: int) -> Any:
        torch = self._torch
        return torch.eye(n, dtype=self._dtype, device=torch.device(self.device))

    def matmul(self, left: Any, right: Any) -> Any:
        torch = self._torch
        return torch.matmul(self.array(left), self.array(right))

    def transpose(self, value: Any) -> Any:
        return self.array(value).T

    def solve(self, left: Any, right: Any) -> Any:
        torch = self._torch
        return torch.linalg.solve(self.array(left), self.array(right))

    def slogdet(self, value: Any) -> tuple[Any, Any]:
        torch = self._torch
        return torch.linalg.slogdet(self.array(value))

    def scalar(self, value: Any) -> float:
        return float(self.as_numpy(value))

    def as_numpy(self, value: Any) -> np.ndarray:
        return value.detach().cpu().numpy()


@dataclass(frozen=True)
class JaxBackend:
    device: str
    dtype: str = "float64"
    name: str = "jax"

    @property
    def _modules(self) -> tuple[Any, Any]:
        jax = import_module("jax")
        jax.config.update("jax_enable_x64", self.dtype == "float64")
        jnp = import_module("jax.numpy")

        return jax, jnp

    @property
    def _device(self) -> Any:
        jax, _ = self._modules
        if self.device == "cpu":
            return jax.devices("cpu")[0]
        for device in jax.devices():
            if device.platform in {"gpu", "cuda"}:
                return device
        msg = "No JAX CUDA device is available."
        raise RuntimeError(msg)

    @property
    def _dtype(self) -> Any:
        _, jnp = self._modules
        return jnp.float64 if self.dtype == "float64" else jnp.float32

    def array(self, value: Any) -> Any:
        jax, jnp = self._modules
        return jax.device_put(jnp.asarray(value, dtype=self._dtype), self._device)

    def zeros(self, shape: tuple[int, ...]) -> Any:
        _, jnp = self._modules
        return jnp.zeros(shape, dtype=self._dtype)

    def eye(self, n: int) -> Any:
        _, jnp = self._modules
        return jnp.eye(n, dtype=self._dtype)

    def matmul(self, left: Any, right: Any) -> Any:
        _, jnp = self._modules
        return jnp.matmul(self.array(left), self.array(right))

    def transpose(self, value: Any) -> Any:
        return self.array(value).T

    def solve(self, left: Any, right: Any) -> Any:
        _, jnp = self._modules
        return jnp.linalg.solve(self.array(left), self.array(right))

    def slogdet(self, value: Any) -> tuple[Any, Any]:
        _, jnp = self._modules
        return jnp.linalg.slogdet(self.array(value))

    def scalar(self, value: Any) -> float:
        return float(self.as_numpy(value))

    def as_numpy(self, value: Any) -> np.ndarray:
        return np.asarray(value)


def get_backend(config: RuntimeConfig | ResolvedRuntime | None = None) -> ArrayBackend:
    if config is None:
        resolved = RuntimeConfig().resolve()
    elif isinstance(config, ResolvedRuntime):
        resolved = config
    else:
        resolved = config.resolve()
    if resolved.backend == "numpy":
        return NumpyBackend(dtype=resolved.dtype)
    if resolved.backend == "torch":
        return TorchBackend(device=resolved.device, dtype=resolved.dtype)
    if resolved.backend == "jax":
        return JaxBackend(device=resolved.device, dtype=resolved.dtype)
    msg = f"Unsupported resolved backend: {resolved.backend}"
    raise ValueError(msg)
