"""
This module contains the SpeechConfig object which
sets the parameters for the SpeechPipeline
"""


class SpeechConfig:
    """ Speech Configuration """

    def __init__(self, config: dict) -> None:
        self.sample_rate = config["sample_rate"]
        self.frame_width = config["frame_width"]
