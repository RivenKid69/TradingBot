# setup.py
from __future__ import annotations

import sys
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize
except Exception as e:
    raise SystemExit(
        "Cython is required to build extensions.\n"
        "Install it first, e.g.: pip install cython"
    )

# --- numpy include dir helper ---
class BuildExtWithNumpy(build_ext):
    def finalize_options(self):
        build_ext.finalize_options(self)
        # defer numpy import until build time (so it's available in venv)
        import numpy as _np
        self.include_dirs = (self.include_dirs or []) + [_np.get_include()]

# --- common flags ---
if sys.platform.startswith("win"):
    c_args = ["/O2"]
    cxx_args = ["/O2", "/std:c++17"]
    link_args = []
else:
    # gnu++17 for <algorithm>, <random>, etc.
    c_args = ["-O3", "-fvisibility=hidden"]
    cxx_args = ["-O3", "-std=gnu++17", "-fvisibility=hidden"]
    link_args = ["-std=gnu++17"]

include_dirs = ["."]
library_dirs = []
libraries = []

# --- sources ---
# C++ sources expected to exist alongside pyx
orderbook_cpp = "OrderBook.cpp"  # your LOB implementation
micro_cpp = "cpp_microstructure_generator.cpp"

ext_modules = [
    Extension(
        name="obs_builder",
        sources=["obs_builder.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execlob_book",
        sources=["execlob_book.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="coreworkspace",
        sources=["coreworkspace.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="execevents",
        sources=["execevents.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c",
        extra_compile_args=c_args,
    ),
    Extension(
        name="fast_lob",
        sources=["fast_lob.pyx", orderbook_cpp],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="fast_market",
        sources=["fast_market.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="reward",
        sources=["reward.pyx"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="micro_sim",
        sources=["micro_sim.pyx", orderbook_cpp, micro_cpp],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="marketmarket_simulator_wrapper",
        sources=["marketmarket_simulator_wrapper.pyx", "MarketSimulator.cpp"],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
    Extension(
        name="lob_state_cython",
        sources=[
            "lob_state_cython.pyx",
            "OrderBook.cpp",
            "MarketSimulator.cpp",
            "cpp_microstructure_generator.cpp",
        ],
        include_dirs=include_dirs,
        libraries=libraries,
        library_dirs=library_dirs,
        language="c++",
        extra_compile_args=cxx_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="tradingbot-extensions",
    version="0.1.0",
    description="Cython/C++ extensions: LOB and microstructure generator",
    py_modules=["apply_no_trade_mask", "no_trade"],
    entry_points={
        "console_scripts": ["no-trade-mask=apply_no_trade_mask:main"],
    },
    ext_modules=cythonize(
        ext_modules,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
        },
    ),
    cmdclass={"build_ext": BuildExtWithNumpy},
)
