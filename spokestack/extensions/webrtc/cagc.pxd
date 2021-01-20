"""
This module contains the Automatic Gain Control's C API redefined in Cython.
"""



cdef extern from "filter_audio/agc/include/gain_control.h":
    ctypedef struct WebRtcAgc_config_t:
        int targetLevelDbfs
        int compressionGaindB
        int limiterEnable

    int WebRtcAgc_Create(void** agcInst)
    int WebRtcAgc_Init(void*agcInst,
                       int minLevel,
                       int maxLevel,
                       short agcMode,
                       int fs)

    int WebRtcAgc_set_config(void*agcInst, WebRtcAgc_config_t config)

    int WebRtcAgc_Free(void*agcInst)

    int WebRtcAgc_VirtualMic(void*agcInst,
                             short*inMic,
                             short*inMic_H,
                             short samples,
                             int micLevelIn,
                             int*micLevelOut)

    int WebRtcAgc_Process(void*agcInst,
                          short*inNear,
                          short*inNearH,
                          short samples,
                          short*out,
                          short*out_H,
                          int inMicLevel,
                          int*outMicLevel,
                          short echo,
                          char*saturationWarning)
