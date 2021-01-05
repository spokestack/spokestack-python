cimport cvad
cimport numpy as np


np.import_array()

cdef class WebRtcVad:
    cdef cvad.VadInst*_vad
    cdef int _sample_rate
    cdef int _mode

    def __dealloc__(self):
        cvad.WebRtcVad_Free(self._vad)

    def __init__(self, sample_rate=16000, mode=0):
        self._sample_rate = sample_rate
        self._mode = mode

        result = cvad.WebRtcVad_Create(&self._vad)
        if result == 0:
            result = cvad.WebRtcVad_Init(self._vad)

            if result == 0:
                result = cvad.WebRtcVad_set_mode(self._vad, mode)

            if result != 0:
                cvad.WebRtcVad_Free(self._vad)
                self._vad = NULL

    def is_speech(self, frame):
        result = cvad.WebRtcVad_Process(
            self._vad,
            self._sample_rate,
            <short*> np.PyArray_DATA(frame),
            len(frame)
        )

        if result < 0:
            raise ValueError

        return result
