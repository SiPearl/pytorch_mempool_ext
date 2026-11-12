"""Fixtures for the pytorch_numa_mem tests.

Environment variables:
    NUMA_ALLOCATOR_LIB   path to libnuma_allocator.so (default: the sibling build)
    NUMA_MEM_NODES_A     node list of the first instance, e.g. "0-3" (default "0")
    NUMA_MEM_NODES_B     node list of the second instance, e.g. "4-7" (default "0")
"""

import json
import os
from pathlib import Path

import pytest

from pytorch_numa_mem import NumaAllocator

DEFAULT_LIB = Path(__file__).resolve().parents[2] / "libnuma_allocator" / "build" / "libnuma_allocator.so"

OBSERVATIONS = []


def parse_nodes(text):
    nodes = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        nodes.extend(range(int(first), int(last or first) + 1))
    return nodes


@pytest.fixture(scope="session")
def lib_path():
    path = Path(os.environ.get("NUMA_ALLOCATOR_LIB", DEFAULT_LIB)).resolve()
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run `make` in libnuma_allocator or set NUMA_ALLOCATOR_LIB")
    return str(path)


@pytest.fixture(scope="session")
def nodes_a():
    return parse_nodes(os.environ.get("NUMA_MEM_NODES_A", "0"))


@pytest.fixture(scope="session")
def nodes_b():
    return parse_nodes(os.environ.get("NUMA_MEM_NODES_B", "0"))


@pytest.fixture
def alloc(lib_path, nodes_a):
    return NumaAllocator(lib_path, nodes_a)


@pytest.fixture
def alloc_b(lib_path, nodes_b):
    return NumaAllocator(lib_path, nodes_b)


@pytest.fixture
def observe(request):
    """Record behaviour that the tests report but do not assert a policy on."""

    def record(label, **values):
        OBSERVATIONS.append((request.node.name, label, values))

    return record


def pytest_terminal_summary(terminalreporter):
    if not OBSERVATIONS:
        return
    terminalreporter.section("recorded behaviour")
    for test, label, values in OBSERVATIONS:
        terminalreporter.write_line(f"{test} [{label}] {json.dumps(values)}")
