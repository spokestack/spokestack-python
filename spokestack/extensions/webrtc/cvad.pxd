cdef extern from "filter_audio/vad/include/webrtc_vad.h":
    ctypedef struct VadInst:
        pass

    int WebRtcVad_Create(VadInst ** handle)
    void WebRtcVad_Free(VadInst*handle)
    int WebRtcVad_Init(VadInst*handle)
    int WebRtcVad_set_mode(VadInst*handle, int mode)
    bint WebRtcVad_Process(VadInst*handle, int fs, short*audio_frame, int frame_length)
