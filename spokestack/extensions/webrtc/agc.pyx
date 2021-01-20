"""
This module contains the AutomaticGainControl class implemented in Cython
"""
# distutils: sources = filter_audio/agc/analog_agc.c, filter_audio/agc/digital_agc.c
# distutils: include_dirs = filter_audio/agc/include/
cimport numpy as np
cimport cagc

from spokestack.extensions.webrtc import ProcessError

MIC_MAX = 255
MIC_TARGET = 180


cdef class WebRtcAgc:
    cdef void* _agc

    def __dealloc__(self):
        cagc.WebRtcAgc_Free(self._agc)

    def __init__(self,
                 sample_rate,
                 frame_width,
                 target_level_dbfs,
                 compression_gain_db,
                 limit_enable
    ):
        self._agc = NULL
        result = cagc.WebRtcAgc_Create(&self._agc)
        if result != 0:
            raise MemoryError("out_of_memory")

        result = cagc.WebRtcAgc_Init(self._agc,
                                     minLevel=0,
                                     maxLevel=MIC_MAX,
                                     agcMode=2,
                                     fs=sample_rate
        )
        if result != 0:
            raise ValueError("invalid_config")

        cdef cagc.WebRtcAgc_config_t config
        config.targetLevelDbfs = target_level_dbfs
        config.limiterEnable = limit_enable
        config.compressionGaindB = compression_gain_db

        result = cagc.WebRtcAgc_set_config(self._agc, config)
        if result != 0:
            raise ValueError("invalid_config")

    def __call__(self, frame):
        self._process(frame)

    cdef _process(self, frame):
        cdef char saturated = 0
        cdef int mic_level = 0
        result = cagc.WebRtcAgc_VirtualMic(
            agcInst=self._agc,
            inMic=<short*> np.PyArray_DATA(frame),
            inMic_H=NULL,
            samples=<short> len(frame),
            micLevelIn=MIC_TARGET,
            micLevelOut=&mic_level
        )
        if result != 0:
            raise ProcessError("mic_failed")

        result = cagc.WebRtcAgc_Process(
                agcInst=self._agc,
                inNear=<short*> np.PyArray_DATA(frame),
                inNearH=NULL,
                samples=<short> len(frame),
                out=<short*> np.PyArray_DATA(frame),
                out_H=NULL,
                inMicLevel=MIC_TARGET,
                outMicLevel=&mic_level,
                echo=0,
                saturationWarning=&saturated
        )
        if result != 0:
            raise ProcessError("invalid_input")
