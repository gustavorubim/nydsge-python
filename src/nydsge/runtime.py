from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Literal, cast

BackendName = Literal["auto", "numpy", "torch", "jax"]
ResolvedBackendName = Literal["numpy", "torch", "jax"]
DeviceName = Literal["auto", "cpu", "cuda", "mps"]
DTypeName = Literal["float64", "float32"]


class UnsupportedRuntimeError(RuntimeError):
    """Raised when a requested backend/device combination is not available."""


@dataclass(frozen=True)
class RuntimeConfig:
    backend: BackendName = "auto"
    device: DeviceName = "auto"
    dtype: DTypeName = "float64"

    def resolve(self) -> ResolvedRuntime:
        return resolve_runtime(self)


@dataclass(frozen=True)
class ResolvedRuntime:
    backend: ResolvedBackendName
    device: Literal["cpu", "cuda", "mps"]
    dtype: DTypeName
    platform: str
    reason: str


@dataclass(frozen=True)
class RuntimeStatus:
    backend: str
    device: str
    available: bool
    reason: str


def platform_name() -> str:
    return platform.system() or "Unknown"


def is_wsl() -> bool:
    if "WSL_INTEROP" in os.environ or "WSL_DISTRO_NAME" in os.environ:
        return True
    if platform.system() != "Linux":
        return False
    release = platform.release().casefold()
    version = platform.version().casefold()
    return "microsoft" in release or "microsoft" in version or "wsl" in release


def native_environment_status() -> RuntimeStatus:
    if is_wsl():
        return RuntimeStatus(
            "platform",
            "native",
            False,
            "WSL is not supported; use native Windows, macOS, or Linux",
        )
    return RuntimeStatus(
        "platform",
        "native",
        True,
        f"native {platform_name()} environment",
    )


def _torch_status(device: str) -> RuntimeStatus:
    try:
        torch: Any = import_module("torch")
    except ImportError:
        return RuntimeStatus("torch", device, False, "torch is not installed")
    if device == "cpu":
        return RuntimeStatus("torch", device, True, "torch CPU is available")
    if device == "cuda":
        ok = bool(torch.cuda.is_available())
        reason = "torch CUDA is available" if ok else "no CUDA device"
        return RuntimeStatus("torch", device, ok, reason)
    if device == "mps":
        mps = getattr(torch.backends, "mps", None)
        ok = bool(mps is not None and mps.is_available())
        reason = "torch MPS is available" if ok else "no MPS device"
        return RuntimeStatus("torch", device, ok, reason)
    return RuntimeStatus("torch", device, False, "unsupported torch device")


def _jax_status(device: str) -> RuntimeStatus:
    try:
        jax: Any = import_module("jax")
    except ImportError:
        return RuntimeStatus("jax", device, False, "jax is not installed")
    if device == "cpu":
        return RuntimeStatus("jax", device, True, "jax CPU is available")
    if device == "cuda":
        if platform_name() != "Linux":
            return RuntimeStatus("jax", device, False, "JAX CUDA wheels are Linux-only")
        has_gpu = any(dev.platform in {"gpu", "cuda"} for dev in jax.devices())
        reason = "JAX CUDA is available" if has_gpu else "no JAX GPU"
        return RuntimeStatus("jax", device, has_gpu, reason)
    if device == "mps":
        return RuntimeStatus("jax", device, False, "JAX MPS is not a supported target")
    return RuntimeStatus("jax", device, False, "unsupported jax device")


def runtime_report() -> list[RuntimeStatus]:
    return [
        native_environment_status(),
        RuntimeStatus("numpy", "cpu", True, "NumPy CPU reference is available"),
        RuntimeStatus("numpy", "cuda", False, "NumPy backend is CPU-only"),
        RuntimeStatus("numpy", "mps", False, "NumPy backend is CPU-only"),
        _torch_status("cpu"),
        _torch_status("cuda"),
        _torch_status("mps"),
        _jax_status("cpu"),
        _jax_status("cuda"),
        _jax_status("mps"),
    ]


def resolve_runtime(config: RuntimeConfig) -> ResolvedRuntime:
    if is_wsl():
        msg = "WSL is not supported; use native Windows, macOS, or Linux."
        raise UnsupportedRuntimeError(msg)

    if config.dtype not in {"float64", "float32"}:
        msg = f"Unsupported dtype: {config.dtype}"
        raise UnsupportedRuntimeError(msg)

    if config.backend == "numpy":
        if config.device not in {"auto", "cpu"}:
            msg = "NumPy backend only supports device='cpu'."
            raise UnsupportedRuntimeError(msg)
        return ResolvedRuntime("numpy", "cpu", config.dtype, platform_name(), "explicit numpy CPU")

    if config.backend == "torch":
        return _resolve_explicit("torch", config.device, config.dtype)

    if config.backend == "jax":
        return _resolve_explicit("jax", config.device, config.dtype)

    if config.backend != "auto":
        msg = f"Unsupported backend: {config.backend}"
        raise UnsupportedRuntimeError(msg)

    if config.device in {"cuda", "mps"}:
        requested_device = cast(Literal["cuda", "mps"], config.device)
        for backend in ("torch", "jax"):
            status = _status_for_runtime(backend, requested_device, config.dtype)
            if status.available:
                return ResolvedRuntime(
                    cast(ResolvedBackendName, backend),
                    requested_device,
                    config.dtype,
                    platform_name(),
                    status.reason,
                )
        msg = (
            "No native backend is available for "
            f"device='{requested_device}' and dtype='{config.dtype}'."
        )
        raise UnsupportedRuntimeError(msg)

    if config.device == "cpu":
        return ResolvedRuntime("numpy", "cpu", config.dtype, platform_name(), "auto CPU reference")

    for backend, device in (("torch", "cuda"), ("torch", "mps"), ("jax", "cuda")):
        status = _status_for_runtime(backend, device, config.dtype)
        if status.available:
            return ResolvedRuntime(
                cast(ResolvedBackendName, backend),
                device,
                config.dtype,
                platform_name(),
                status.reason,
            )

    return ResolvedRuntime("numpy", "cpu", config.dtype, platform_name(), "auto CPU fallback")


def _status_for(backend: str, device: str) -> RuntimeStatus:
    if backend == "torch":
        return _torch_status(device)
    if backend == "jax":
        return _jax_status(device)
    if backend == "numpy":
        return RuntimeStatus("numpy", device, device == "cpu", "NumPy CPU reference is available")
    return RuntimeStatus(backend, device, False, "unknown backend")


def _status_for_runtime(backend: str, device: str, dtype: DTypeName) -> RuntimeStatus:
    status = _status_for(backend, device)
    if not status.available:
        return status
    if backend == "torch" and device == "mps" and dtype == "float64":
        return RuntimeStatus(
            backend,
            device,
            False,
            "torch MPS does not support float64; use dtype='float32' or CPU",
        )
    return status


def _resolve_explicit(
    backend: Literal["torch", "jax"],
    device: DeviceName,
    dtype: DTypeName,
) -> ResolvedRuntime:
    if device == "auto":
        candidate_devices = ("cuda", "mps", "cpu") if backend == "torch" else ("cuda", "cpu")
    else:
        candidate_devices = (device,)
    last_reason = ""
    for candidate in candidate_devices:
        status = _status_for_runtime(backend, candidate, dtype)
        last_reason = status.reason
        if status.available:
            return ResolvedRuntime(
                backend,
                candidate,
                dtype,
                platform_name(),
                status.reason,
            )
    detail = "" if not last_reason else f": {last_reason}"
    msg = f"{backend} is not available for device='{device}'{detail}."
    raise UnsupportedRuntimeError(msg)
