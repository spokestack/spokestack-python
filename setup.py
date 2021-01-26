import os

import numpy as np
import setuptools
from Cython.Build import cythonize

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
    setuptools.Extension(
        "spokestack.extensions.webrtc.agc",
        ["spokestack/extensions/webrtc/agc.pyx"] + SOURCES,
        include_dirs=["filter_audio/agc/include/"],
    ),
    setuptools.Extension(
        "spokestack.extensions.webrtc.nsx",
        ["spokestack/extensions/webrtc/nsx.pyx"] + SOURCES,
        include_dirs=["filter_audio/ns/include/"],
    ),
    setuptools.Extension(
        "spokestack.extensions.webrtc.vad",
        ["spokestack/extensions/webrtc/vad.pyx"] + SOURCES,
        include_dirs=["filter_audio/agc/include/"],
    ),
]
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spokestack",
    version="0.0.15",
    author="Spokestack",
    author_email="support@spokestack.io",
    description="Spokestack Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spokestack/spokestack-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    setup_requires=["setuptools", "cython", "numpy"],
    install_requires=[
        "numpy",
        "pyaudio",
        "webrtcvad",
        "websocket_client",
        "tokenizers",
        "requests",
        "streamp3",
        "cython",
    ],
    ext_modules=cythonize(EXTENSIONS),
    include_dirs=[np.get_include()],
    zip_safe=False,
)
