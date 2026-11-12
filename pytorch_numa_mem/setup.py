from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

ROOT = Path(__file__).resolve().parent

setup(
    ext_modules=[
        CppExtension(
            name="pytorch_numa_mem._C",
            sources=["src/pytorch_numa_mem/_C.cpp"],
            # numa_allocator.h: the config struct and the ABI version.
            include_dirs=[str(ROOT.parent / "libnuma_allocator")],
            libraries=["dl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
