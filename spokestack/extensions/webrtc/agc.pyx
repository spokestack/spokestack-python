"""
This module contains the AutomaticGainControl class implemented in Cython
"""
# distutils: sources = filter_audio/agc/analog_agc.c, filter_audio/agc/digital_agc.c
# distutils: include_dirs = filter_audio/agc/include/
cimport numpy as np
cimport cagc


MIC_MAX = 255
MIC_TARGET = 180

np.import_array()

cdef class AutomaticGainControl:
    cdef void*_agc
    cdef int _sample_rate
    cdef int _frame_width
    cdef int _target_level_dbfs
    cdef int _compression_gain_db
    cdef int _limit_enable
    cdef cagc.WebRtcAgc_config_t _config

    def __dealloc__(self):
        cagc.WebRtcAgc_Free(self._agc)

    def __init__(self,
                 sample_rate=16000,
                 frame_width=20,
                 target_level_dbfs=0,
                 compression_gain_db=0,
                 limit_enable=True,
                 **kwargs):

        self._sample_rate = sample_rate
        self._frame_width = frame_width
        self._target_level_dbfs = target_level_dbfs
        self._compression_gain_db = compression_gain_db
        self._limit_enable = limit_enable

        self._agc = NULL
        result = cagc.WebRtcAgc_Create(&self._agc)
        if result < 0:
            raise ValueError
        if result == 0:
            result = cagc.WebRtcAgc_Init(self._agc,
                                         minLevel=0,
                                         maxLevel=MIC_MAX,
                                         agcMode=2,
                                         fs=self._sample_rate)

            if result == 0:
                self._config.targetLevelDbfs = self._target_level_dbfs
                self._config.limiterEnable = self._limit_enable
                self._config.compressionGaindB = self._compression_gain_db
                result = cagc.WebRtcAgc_set_config(self._agc, self._config)
        if result < 0:
            raise ValueError

    def __call__(self, context, frame):
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
        if result == 0:
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
        if result < 0:
            raise ValueError

    def close(self):
        pass

    def reset(self):
        pass
