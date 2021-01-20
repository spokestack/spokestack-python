"""
This module contains the AutomaticNoiseSuppression class implemented in Cython
"""
# distutils: sources = filter_audio/ns/noise_suppression_x.c
# distutils: include_dirs = filter_audio/ns/include/
cimport cnsx
cimport numpy as np

from spokestack.extensions.webrtc import ProcessError


cdef class WebRtcNsx:
    cdef cnsx.NsxHandle* _ans

    def __dealloc__(self):
        cnsx.WebRtcNsx_Free(self._ans)
        self._ans = NULL

    def __init__(self, sample_rate, policy):
        self._ans = NULL

        result = cnsx.WebRtcNsx_Create(&self._ans)
        if result != 0:
            raise MemoryError("out_of_memory")

        result = cnsx.WebRtcNsx_Init(self._ans, sample_rate)
        if result != 0:
            raise ValueError("invalid_config")

        result = cnsx.WebRtcNsx_set_policy(self._ans, policy)
        if result != 0:
            raise ValueError("invalid_config")

    def __call__(self, frame):
        self._process(frame)

    cdef _process(self, frame):
        result = cnsx.WebRtcNsx_Process(self._ans,
                                        <short*> np.PyArray_DATA(frame),
                                        NULL,
                                        <short*> np.PyArray_DATA(frame),
                                        NULL)
        if result != 0:
            raise ProcessError("invalid_input")
