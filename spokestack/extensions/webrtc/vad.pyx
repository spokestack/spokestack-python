cimport cvad
cimport numpy as np

from spokestack.extensions.webrtc import ProcessError


cdef class WebRtcVad:
    cdef cvad.VadInst* _vad
    cdef int _sample_rate

    def __dealloc__(self):
        cvad.WebRtcVad_Free(self._vad)

    def __init__(self, sample_rate, mode):
        self._vad = NULL
        self._sample_rate = sample_rate

        result = cvad.WebRtcVad_Create(&self._vad)
        if result != 0:
            raise MemoryError("out_of_memory")

        result = cvad.WebRtcVad_Init(self._vad)
        if result != 0:
            raise ValueError("invalid_config")

        result = cvad.WebRtcVad_set_mode(self._vad, mode)
        if result != 0:
            raise ValueError("invalid_config")

    def is_speech(self, frame):
        result = cvad.WebRtcVad_Process(
            self._vad,
            self._sample_rate,
            <short*> np.PyArray_DATA(frame),
            len(frame)
        )
        return result
