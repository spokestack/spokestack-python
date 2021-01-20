cdef extern from "filter_audio/ns/include/noise_suppression_x.h":
    ctypedef struct NsxHandle:
        pass

    int WebRtcNsx_Create(NsxHandle** nsxInst)
    int WebRtcNsx_Free(NsxHandle*nsxInst)
    int WebRtcNsx_Init(NsxHandle*nsxInst, int fs)
    int WebRtcNsx_set_policy(NsxHandle*nsxInst, int mode)
    int WebRtcNsx_Process(NsxHandle*nsxInst,
                          short*speechFrame,
                          short*speechFrameHB,
                          short*outFrame,
                          short*outFrameHB)
