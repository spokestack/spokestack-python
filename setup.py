import os
import subprocess
import sys

from setuptools import Extension, find_packages, setup
from setuptools.command.build_py import build_py


try:
    from numpy import get_include
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy==1.19.2"])
    from numpy import get_include

try:
    from Cython.Build import cythonize
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython==0.29.22"])
    from Cython.Build import cythonize


class CustomBuild(build_py):  # type: ignore
    """Custom build command to build PortAudio."""

    def run(self) -> None:
        """Custom run function that builds and installs PortAudio/PyAudio."""

        if sys.platform == "mingw":
            # build with MinGW for windows
            command = ["./configure && make && make install"]
        elif sys.platform in ["win32", "win64"]:
            # win32/64 users should install the PyAudio wheel or Conda package
            command = None
        else:
            # macos or linux
            command = ["./configure && make"]

        if command:
            # build PortAudio with system specific command
            subprocess.run(
                command,
                shell=True,
                check=True,
                cwd="spokestack/extensions/portaudio",
            )
            # install PyAudio after PortAudio has been built
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "pyaudio"],
                shell=True,
                check=True,
            )
        # run the normal build process
        build_py.run(self)


SOURCES = [
    os.path.join("spokestack/extensions/webrtc", source)
    for source in [
        "filter_audio/other/complex_bit_reverse.c",
        "filter_audio/other/complex_fft.c",
        "filter_audio/other/copy_set_operations.c",
        "filter_audio/other/cross_correlation.c",
        "filter_audio/other/division_operations.c",
        "filter_audio/other/dot_product_with_scale.c",
        "filter_audio/other/downsample_fast.c",
        "filter_audio/other/energy.c",
        "filter_audio/other/get_scaling_square.c",
        "filter_audio/other/min_max_operations.c",
        "filter_audio/other/real_fft.c",
        "filter_audio/other/resample_by_2.c",
        "filter_audio/other/resample_by_2_internal.c",
        "filter_audio/other/resample_fractional.c",
        "filter_audio/other/resample_48khz.c",
        "filter_audio/other/spl_init.c",
        "filter_audio/other/spl_sqrt.c",
        "filter_audio/other/spl_sqrt_floor.c",
        "filter_audio/other/vector_scaling_operations.c",
        "filter_audio/vad/vad_core.c",
        "filter_audio/vad/vad_filterbank.c",
        "filter_audio/vad/vad_gmm.c",
        "filter_audio/vad/vad_sp.c",
        "filter_audio/vad/webrtc_vad.c",
        "filter_audio/agc/analog_agc.c",
        "filter_audio/agc/digital_agc.c",
        "filter_audio/ns/nsx_core.c",
        "filter_audio/ns/nsx_core_c.c",
        "filter_audio/ns/noise_suppression_x.c",
    ]
]

EXTENSIONS = [
    Extension(
        "spokestack.extensions.webrtc.agc",
        ["spokestack/extensions/webrtc/agc.pyx"] + SOURCES,
        include_dirs=["filter_audio/agc/include/"],
    ),
    Extension(
        "spokestack.extensions.webrtc.nsx",
        ["spokestack/extensions/webrtc/nsx.pyx"] + SOURCES,
        include_dirs=["filter_audio/ns/include/"],
    ),
    Extension(
        "spokestack.extensions.webrtc.vad",
        ["spokestack/extensions/webrtc/vad.pyx"] + SOURCES,
        include_dirs=["filter_audio/agc/include/"],
    ),
]
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="spokestack",
    version="0.0.20",
    author="Spokestack",
    author_email="support@spokestack.io",
    description="Spokestack Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spokestack/spokestack-python",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    setup_requires=["setuptools", "wheel", "numpy==1.19.2", "Cython>=0.29.22"],
    install_requires=[
        "numpy==1.19.2",
        "Cython>=0.29.22",
        "websocket_client",
        "tokenizers",
        "requests",
    ],
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[get_include()],
    cmdclass={"build_py": CustomBuild},
    zip_safe=False,
)
