"""
This module contains the AutomaticNoiseSuppression class implemented in Cython
"""
# distutils: sources = filter_audio/ns/noise_suppression_x.c
# distutils: include_dirs = filter_audio/ns/include/
cimport cans
cimport numpy as np


np.import_array()

cdef class AutomaticNoiseSuppression:
    cdef cans.NsxHandle *_ans
    cdef int _sample_rate
    cdef int _policy
    cdef int _frame_width

    def __dealloc__(self):
        cans.WebRtcNsx_Free(self._ans)
        self._ans = NULL

    def __init__(self, sample_rate=16000, policy=0, **kwargs):
        self._ans = NULL
        self._sample_rate = sample_rate
        self._policy = policy
        self._frame_width = self._sample_rate * 10 / 1000

        if self._frame_width % 10 != 0:
            raise ValueError

        result = cans.WebRtcNsx_Create(&self._ans)
        if result == 0:
            result = cans.WebRtcNsx_Init(self._ans, self._sample_rate)

            if result == 0:
                result = cans.WebRtcNsx_set_policy(self._ans, self._policy)

    def __call__(self, context, frame):
        frame_size = self._frame_width
        if len(frame) % frame_size != 0:
            raise ValueError

        for i in range(len(frame) // frame_size):
            self._process(frame[i:frame_size + i])

    cdef _process(self, frame):
        result = cans.WebRtcNsx_Process(self._ans,
                                        <short*> np.PyArray_DATA(frame),
                                        NULL,
                                        <short*> np.PyArray_DATA(frame),
                                        NULL)
        if result != 0:
            raise ValueError

    def close(self):
        pass

    def reset(self):
        pass
