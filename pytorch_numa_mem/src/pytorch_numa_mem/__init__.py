"""CPU tensors on chosen NUMA nodes, through an external libnuma_allocator.so.

Two ways to place a tensor:

- ``alloc.empty(...)``, ``alloc.empty_like(t)`` and ``alloc.clone_from(t)`` for
  tensors the application allocates itself.
- ``with use_allocator(alloc):`` for tensors that eager ops allocate on the
  calling thread. This needs a PyTorch build that has
  ``c10::impl::CPUAllocatorOverrideGuard``.
"""

import os
from collections.abc import Iterable

import torch  # loads the libraries that _C links against

from . import _C

__all__ = ["NumaAllocator", "use_allocator"]

_MODES = {
    "bind": _C.BIND,
    "interleave": _C.INTERLEAVE,
    "preferred_many": _C.PREFERRED_MANY,
}


class NumaAllocator:
    """One libnuma_allocator.so instance with its own node set and policy.

    The library is unloaded when this object, every allocation it made and
    every ``use_allocator`` scope using it are gone. A storage emptied with
    ``untyped_storage().resize_(0)`` holds no allocation, so it does not keep
    the library loaded; resizing its tensor after the unload raises
    ``RuntimeError``.

    Args:
        lib_path: path to ``libnuma_allocator.so``.
        nodes: NUMA node IDs to allocate from.
        mode: ``"bind"``, ``"interleave"`` or ``"preferred_many"``.
        prefault: touch every page at allocation time, so placement happens
            then and not at first use.
        *_fn_name: symbol names to look up in the library.
    """

    def __init__(
        self,
        lib_path,
        nodes,
        mode="bind",
        prefault=False,
        *,
        abi_version_fn_name="numa_allocator_abi_version",
        create_fn_name="numa_allocator_create",
        alloc_fn_name="numa_allocator_alloc",
        free_fn_name="numa_allocator_free",
        destroy_fn_name="numa_allocator_destroy",
    ):
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {sorted(_MODES)}, got {mode!r}")
        self.nodes = tuple(int(node) for node in nodes)
        self.mode = mode
        self.prefault = bool(prefault)
        self._impl = _C.NumaAllocatorImpl(
            os.fspath(lib_path),
            list(self.nodes),
            _MODES[mode],
            self.prefault,
            abi_version_fn_name,
            create_fn_name,
            alloc_fn_name,
            free_fn_name,
            destroy_fn_name,
        )

    @property
    def counters(self):
        """Live counters. The returned object stays valid after the allocator is gone.

        Attributes: ``create_calls``, ``destroy_calls``, ``alloc_calls``,
        ``free_calls``, ``live_allocations``, ``live_bytes``. ``alloc_calls`` and
        ``free_calls`` count calls into the library; zero-byte allocations do
        not reach it and only show up in ``live_allocations``.
        """
        return self._impl.counters

    def empty(self, *size, dtype=None):
        """Uninitialized CPU tensor with resizable storage from this allocator."""
        if len(size) == 1 and isinstance(size[0], Iterable):
            size = tuple(size[0])
        if dtype is None:
            dtype = torch.get_default_dtype()
        return self._impl.empty(list(size), dtype)

    def empty_like(self, tensor):
        """Like ``empty``, with the shape and dtype of ``tensor``."""
        return self.empty(tensor.shape, dtype=tensor.dtype)

    def clone_from(self, tensor):
        """Copy of ``tensor`` in storage from this allocator."""
        return self.empty_like(tensor).copy_(tensor)

    def owns(self, tensor):
        """Whether the storage of ``tensor`` was created by this allocator."""
        return self._impl.owns(tensor)


class use_allocator:
    """Route CPU tensors allocated by the calling thread to ``allocator``.

    Covers what ``torch.empty`` and eager ops allocate while the block runs.
    Other threads are not affected, pinned-memory requests are not affected,
    and a tensor keeps the allocator that created it after the block exits.
    ``use_allocator(None)`` selects the default allocator inside an outer block.

    Blocks must be exited in reverse order on the thread that entered them.
    """

    def __init__(self, allocator):
        self._allocator = allocator
        self._scopes = []

    def __enter__(self):
        impl = None if self._allocator is None else self._allocator._impl
        self._scopes.append(_C.OverrideScope(impl))
        return self._allocator

    def __exit__(self, exc_type, exc_value, traceback):
        self._scopes[-1].exit()
        self._scopes.pop()
