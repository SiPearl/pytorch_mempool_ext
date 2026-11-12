import contextlib
import gc
import itertools
import os
import re
import subprocess
import sys
import threading

import pytest
import torch
import torch.nn.functional as F

from pytorch_numa_mem import NumaAllocator, _C, use_allocator

PAGE_SIZE = os.sysconf("SC_PAGESIZE")


def vma_policy(address):
    """Policy string and per-node page counts of the mapping that holds address."""
    best = None
    with open("/proc/self/numa_maps") as numa_maps:
        for line in numa_maps:
            fields = line.split()
            start = int(fields[0], 16)
            if start <= address and (best is None or start > best[0]):
                best = (start, fields)
    pages = {}
    for field in best[1][2:]:
        if match := re.fullmatch(r"N(\d+)=(\d+)", field):
            pages[int(match[1])] = int(match[2])
    return best[1][1], pages


def scope_if(enabled, alloc):
    return use_allocator(alloc) if enabled else contextlib.nullcontext()


# Factories and counters


def test_no_override(alloc):
    assert not alloc.owns(torch.empty(4))
    assert alloc.counters.create_calls == 1
    assert alloc.counters.live_allocations == 0


def test_factories(alloc):
    src = torch.arange(6.0).reshape(2, 3)
    a = alloc.empty(2, 3)
    b = alloc.empty((2, 3), dtype=torch.int32)
    c = alloc.empty_like(src)
    d = alloc.clone_from(src)
    assert all(alloc.owns(t) and t.shape == (2, 3) and t.device.type == "cpu" for t in (a, b, c, d))
    assert (a.dtype, b.dtype, c.dtype) == (torch.float32, torch.int32, src.dtype)
    assert torch.equal(d, src)
    assert alloc.counters.live_allocations == 4
    assert alloc.counters.live_bytes == 4 * 24
    del a, b, c, d
    assert alloc.counters.live_allocations == 0
    assert alloc.counters.live_bytes == 0
    assert alloc.counters.alloc_calls == alloc.counters.free_calls == 4


def test_instances_are_independent(alloc, alloc_b):
    a, b = alloc.empty(16), alloc_b.empty(32)
    assert alloc.owns(a) and not alloc.owns(b)
    assert alloc_b.owns(b) and not alloc_b.owns(a)
    assert alloc.counters.live_bytes == 64
    assert alloc_b.counters.live_bytes == 128


@pytest.mark.parametrize("prefault", [False, True])
@pytest.mark.parametrize("mode", ["bind", "interleave", "preferred_many"])
def test_modes_allocate_usable_memory(lib_path, nodes_a, mode, prefault):
    try:
        alloc = NumaAllocator(lib_path, nodes_a, mode=mode, prefault=prefault)
    except RuntimeError as error:
        if mode != "preferred_many":
            raise
        pytest.skip(f"the kernel rejects MPOL_PREFERRED_MANY: {error}")
    t = alloc.empty(1 << 16)
    t.fill_(3.0)
    assert t.sum().item() == 3.0 * (1 << 16)


def test_rejects_bad_configuration(lib_path, nodes_a):
    with pytest.raises(ValueError, match="mode"):
        NumaAllocator(lib_path, nodes_a, mode="local")
    with pytest.raises(RuntimeError, match="not online"):
        NumaAllocator(lib_path, [1023])
    with pytest.raises(RuntimeError, match="at least one node"):
        NumaAllocator(lib_path, [])
    with pytest.raises(RuntimeError, match="dlopen"):
        NumaAllocator("/nonexistent/libnuma_allocator.so", nodes_a)
    with pytest.raises(RuntimeError, match="symbol not found"):
        NumaAllocator(lib_path, nodes_a, alloc_fn_name="numa_alloc")


def test_allocation_failure_raises(alloc):
    with pytest.raises(RuntimeError, match="failed to allocate"):
        alloc.empty(1 << 60)
    with pytest.raises(RuntimeError, match="negative dimension"):
        alloc.empty(-1)
    assert alloc.counters.live_allocations == 0
    assert alloc.counters.alloc_calls == 0


# The override scope: the 1B cases seen from Python


def test_single_scope_routes_op_outputs(alloc):
    x = torch.randn(8, 16)
    weight = torch.randn(32, 16)
    bias = torch.randn(32)

    def run_ops():
        return {
            "linear": F.linear(x, weight, bias),
            "gelu": F.gelu(x),
            "add": x + x,
            "cat": torch.cat([x, x]),
            "empty_strided": torch.empty_strided((2, 3), (1, 2)),
        }

    expected = run_ops()
    with use_allocator(alloc) as active:
        assert active is alloc
        routed = run_ops()
    assert all(alloc.owns(t) for t in routed.values())
    assert not any(alloc.owns(t) for t in expected.values())
    for name in ("linear", "gelu", "add", "cat"):
        torch.testing.assert_close(routed[name], expected[name])
    assert not alloc.owns(x)
    assert not alloc.owns(torch.empty(4))

    del routed
    assert alloc.counters.live_allocations == 0
    assert alloc.counters.alloc_calls == alloc.counters.free_calls


def test_nested_scopes_restore_outer(alloc, alloc_b):
    with use_allocator(alloc):
        assert alloc.owns(torch.empty(1))
        with use_allocator(alloc_b):
            inner = torch.empty(2)
            assert alloc_b.owns(inner) and not alloc.owns(inner)
        assert alloc.owns(torch.empty(3))
    assert not alloc.owns(torch.empty(4))


def test_none_inner_scope_selects_default(alloc):
    with use_allocator(alloc):
        with use_allocator(None):
            assert not alloc.owns(torch.empty(4))
        assert alloc.counters.alloc_calls == 0
        assert alloc.owns(torch.empty(4))


def test_restored_after_exception(alloc):
    a, b = torch.ones(2), torch.ones(3)
    with pytest.raises(RuntimeError):
        with use_allocator(alloc):
            kept = torch.empty(8)
            a + b
    assert alloc.owns(kept)
    assert not alloc.owns(torch.empty(4))
    del kept
    assert alloc.counters.live_allocations == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no pinned memory allocator in this build")
def test_pin_memory_ignores_override(alloc):
    with use_allocator(alloc):
        pinned = torch.empty(4, pin_memory=True)
    assert not alloc.owns(pinned)
    assert alloc.counters.alloc_calls == 0


def test_view_outliving_scope_frees_once(alloc):
    with use_allocator(alloc):
        view = torch.empty(4, 8)[1:3]
    assert alloc.counters.live_bytes == 128
    assert alloc.counters.free_calls == 0
    del view
    assert alloc.counters.live_bytes == 0
    assert alloc.counters.alloc_calls == alloc.counters.free_calls == 1


def test_resize_after_scope_uses_creating_allocator(alloc):
    with use_allocator(alloc):
        t = torch.empty(4)
    t.resize_(64)
    assert alloc.owns(t)
    assert alloc.counters.live_bytes == 256
    assert (alloc.counters.alloc_calls, alloc.counters.free_calls) == (2, 1)


def test_default_tensor_resized_in_scope_keeps_default(alloc):
    t = torch.empty(4)
    with use_allocator(alloc):
        t.resize_(64)
    assert not alloc.owns(t)
    assert alloc.counters.alloc_calls == 0


def test_storageless_resize_in_scope_is_routed(alloc):
    t = _C._storageless_tensor(torch.float32)
    assert not alloc.owns(t)
    with use_allocator(alloc):
        t.resize_(8)
    assert alloc.owns(t)
    assert alloc.counters.live_bytes == 32


def test_other_threads_are_unaffected(alloc, alloc_b):
    seen = {}

    def without_scope():
        seen["plain"] = torch.empty(4)

    def with_own_scope():
        with use_allocator(alloc_b):
            seen["scoped"] = torch.empty(4)

    with use_allocator(alloc):
        for target in (without_scope, with_own_scope):
            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
        assert alloc.owns(torch.empty(4))
    assert not alloc.owns(seen["plain"]) and not alloc_b.owns(seen["plain"])
    assert alloc_b.owns(seen["scoped"])


def test_out_of_order_exit_raises(alloc, alloc_b):
    outer, inner = use_allocator(alloc), use_allocator(alloc_b)
    outer.__enter__()
    inner.__enter__()
    with pytest.raises(RuntimeError, match="reverse order"):
        outer.__exit__(None, None, None)
    assert alloc_b.owns(torch.empty(4))
    inner.__exit__(None, None, None)
    assert alloc.owns(torch.empty(4))
    outer.__exit__(None, None, None)
    assert not alloc.owns(torch.empty(4))


def test_exit_on_another_thread_raises(alloc):
    scope = use_allocator(alloc)
    scope.__enter__()
    errors = []

    def exit_scope():
        try:
            scope.__exit__(None, None, None)
        except RuntimeError as error:
            errors.append(str(error))

    thread = threading.Thread(target=exit_scope)
    thread.start()
    thread.join()
    assert len(errors) == 1 and "thread that entered" in errors[0]
    assert alloc.owns(torch.empty(4))
    scope.__exit__(None, None, None)
    assert not alloc.owns(torch.empty(4))


def test_scope_abandoned_on_another_thread_leaves_this_thread_alone(alloc, alloc_b):
    abandoned = []

    def enter_and_abandon():
        scope = use_allocator(alloc)
        scope.__enter__()
        abandoned.append(scope)

    thread = threading.Thread(target=enter_and_abandon)
    thread.start()
    thread.join()
    with use_allocator(alloc_b):
        abandoned.clear()
        gc.collect()
        assert alloc_b.owns(torch.empty(4))
    assert not alloc.owns(torch.empty(4)) and not alloc_b.owns(torch.empty(4))


def test_concurrent_scopes(alloc, alloc_b):
    failures = []

    def worker(mine, other):
        for _ in range(200):
            with use_allocator(mine):
                t = torch.ones(64) + 1
            if not mine.owns(t) or other.owns(t):
                failures.append(t)

    threads = [threading.Thread(target=worker, args=pair) for pair in [(alloc, alloc_b), (alloc_b, alloc)] * 4]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not failures
    for counters in (alloc.counters, alloc_b.counters):
        assert counters.live_allocations == 0
        assert counters.alloc_calls == counters.free_calls > 0


# Lifetime


def test_dropping_allocator_while_tensors_live(lib_path, nodes_a):
    alloc = NumaAllocator(lib_path, nodes_a)
    counters = alloc.counters
    t = alloc.empty(1024)
    with use_allocator(alloc):
        u = torch.ones(1024)
    del alloc
    gc.collect()
    assert counters.destroy_calls == 0

    t.fill_(2.0)
    t.resize_(4096)
    assert counters.live_bytes == (4096 + 1024) * 4
    assert t[:1024].eq(2.0).all()
    del t
    assert counters.destroy_calls == 0
    del u
    assert counters.destroy_calls == 1
    assert counters.live_allocations == 0
    assert counters.alloc_calls == counters.free_calls == 3


def test_zero_byte_tensor_keeps_allocator_alive(lib_path, nodes_a):
    alloc = NumaAllocator(lib_path, nodes_a)
    counters = alloc.counters
    z = alloc.empty(0)
    assert alloc.owns(z)
    assert (counters.live_allocations, counters.alloc_calls) == (1, 0)
    del alloc
    gc.collect()
    assert counters.destroy_calls == 0

    z.resize_(100).fill_(1.0)
    assert counters.live_bytes == 400
    del z
    assert (counters.destroy_calls, counters.live_allocations, counters.live_bytes) == (1, 0, 0)
    assert counters.alloc_calls == counters.free_calls == 1


def test_resize_after_allocator_destroyed_raises(lib_path, nodes_a):
    alloc = NumaAllocator(lib_path, nodes_a)
    counters = alloc.counters
    t = alloc.empty(1024)
    t.untyped_storage().resize_(0)
    assert alloc.owns(t)
    assert (counters.live_allocations, counters.destroy_calls) == (0, 0)
    del alloc
    gc.collect()
    assert counters.destroy_calls == 1

    # resize_ sets the shape before it allocates, so a repeat with the same size returns early.
    for numel in (2048, 4096):
        with pytest.raises(RuntimeError, match="allocator was destroyed"):
            t.resize_(numel)
    assert t.untyped_storage().nbytes() == 0
    assert (counters.destroy_calls, counters.live_allocations, counters.live_bytes) == (1, 0, 0)
    assert counters.alloc_calls == counters.free_calls == 1


def test_library_unloaded_after_last_reference(lib_path, nodes_a):
    script = """
import sys
from pytorch_numa_mem import NumaAllocator

lib, nodes = sys.argv[1], [int(node) for node in sys.argv[2].split(",")]

def mapped():
    with open("/proc/self/maps") as maps:
        return any(line.rstrip().endswith(lib) for line in maps)

alloc = NumaAllocator(lib, nodes)
t = alloc.empty(1024)
assert mapped(), "library not mapped after creation"
del alloc
assert mapped(), "library unloaded while a tensor is alive"
del t
assert not mapped(), "library still mapped after the last reference"
"""
    subprocess.run([sys.executable, "-c", script, lib_path, ",".join(map(str, nodes_a))], check=True)


# Placement


def test_tensor_memory_carries_the_policy(lib_path, nodes_a):
    alloc = NumaAllocator(lib_path, nodes_a, prefault=True)
    t = alloc.empty(1 << 20)
    policy, pages = vma_policy(t.data_ptr())
    assert policy.startswith("bind:"), policy
    assert set(pages) <= set(nodes_a), pages
    assert sum(pages.values()) * PAGE_SIZE >= t.nbytes


def test_residency_on_two_node_sets(lib_path, nodes_a, nodes_b):
    if set(nodes_a) == set(nodes_b):
        pytest.skip("needs different NUMA_MEM_NODES_A and NUMA_MEM_NODES_B")
    for nodes in (nodes_a, nodes_b):
        alloc = NumaAllocator(lib_path, nodes)
        with use_allocator(alloc):
            t = torch.ones(1 << 20)
        policy, pages = vma_policy(t.data_ptr())
        assert alloc.owns(t)
        assert set(pages) <= set(nodes), (policy, pages)
        assert sum(pages.values()) * PAGE_SIZE >= t.nbytes


# Behaviour that is recorded, not asserted as policy


def test_record_allocations_per_op(alloc, observe):
    x = torch.randn(8, 16)
    weight = torch.randn(32, 16)
    bias = torch.randn(32)
    ops = {
        "F.linear(x, weight, bias)": lambda: F.linear(x, weight, bias),
        "F.gelu(x)": lambda: F.gelu(x),
        "x + x": lambda: x + x,
        "x * 2.0": lambda: x * 2.0,
        "torch.cat([x, x])": lambda: torch.cat([x, x]),
        "x.sum()": lambda: x.sum(),
        "torch.tensor([1.0, 2.0])": lambda: torch.tensor([1.0, 2.0]),
    }
    counters = alloc.counters
    with use_allocator(alloc):
        for name, op in ops.items():
            calls, live = counters.alloc_calls, counters.live_bytes
            out = op()
            observe(name, alloc_calls=counters.alloc_calls - calls, output_bytes=counters.live_bytes - live)
            assert alloc.owns(out)
            del out
    assert counters.live_allocations == 0


def test_record_autograd_behaviour(alloc, observe):
    torch.manual_seed(0)
    weight_init = torch.randn(64, 64)
    x = torch.randn(8, 64)
    grads = []
    counters = alloc.counters
    for forward_in_scope, backward_in_scope in itertools.product([False, True], repeat=2):
        weight = weight_init.clone().requires_grad_()
        start = counters.alloc_calls
        with scope_if(forward_in_scope, alloc):
            loss = F.gelu(F.linear(x, weight)).sum()
        after_forward = counters.alloc_calls
        with scope_if(backward_in_scope, alloc):
            loss.backward()
        observe(
            f"forward {'inside' if forward_in_scope else 'outside'}, backward {'inside' if backward_in_scope else 'outside'}",
            loss_owned=alloc.owns(loss),
            grad_owned=alloc.owns(weight.grad),
            forward_alloc_calls=after_forward - start,
            backward_alloc_calls=counters.alloc_calls - after_forward,
            live_allocations_after_backward=counters.live_allocations,
        )
        grads.append(weight.grad.clone())
        del weight, loss
    for grad in grads[1:]:
        torch.testing.assert_close(grad, grads[0])
    assert counters.live_allocations == 0


def test_record_torch_compile_behaviour(alloc, observe):
    def f(x, weight):
        return F.gelu(F.linear(x, weight))

    x = torch.randn(8, 64)
    weight = torch.randn(64, 64)
    expected = f(x, weight)
    try:
        torch.testing.assert_close(torch.compile(f)(x, weight), expected)
    except Exception as error:
        pytest.skip(f"torch.compile does not work outside a scope here: {type(error).__name__}: {error}")
    torch._dynamo.reset()

    compiled = torch.compile(f)
    counters = alloc.counters
    with use_allocator(alloc):
        first = compiled(x, weight)
        compile_call_allocs = counters.alloc_calls
        second = compiled(x, weight)
        second_call_allocs = counters.alloc_calls - compile_call_allocs
    outside = compiled(x, weight)
    outside_call_allocs = counters.alloc_calls - compile_call_allocs - second_call_allocs
    for out in (first, second, outside):
        torch.testing.assert_close(out, expected)
    observe(
        "compiled inside the scope",
        first_call_output_owned=alloc.owns(first),
        second_call_output_owned=alloc.owns(second),
        call_outside_scope_output_owned=alloc.owns(outside),
        alloc_calls_during_compiling_call=compile_call_allocs,
        alloc_calls_during_second_call=second_call_allocs,
        alloc_calls_during_call_outside_scope=outside_call_allocs,
    )
    del first, second, outside
    observe("after dropping the outputs", live_allocations=counters.live_allocations, live_bytes=counters.live_bytes)
    del compiled
    torch._dynamo.reset()
    gc.collect()
    observe("after torch._dynamo.reset()", live_allocations=counters.live_allocations, live_bytes=counters.live_bytes)
