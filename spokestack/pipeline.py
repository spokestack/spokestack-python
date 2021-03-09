"""
This module contains the speech pipeline which manages the components
for processing speech.
"""
from typing import Any, List, Union

from spokestack.context import SpeechContext


class SpeechPipeline:
    """Pipeline for managing speech components.

    Args:
        input_source: source of audio input
        stages: components desired in the pipeline
        **kwargs: additional keyword arguments
    """

    def __init__(self, input_source: Any, stages: List[Any]) -> None:
        self._context = SpeechContext()
        self._input_source = input_source
        self._stages: list = stages
        self._is_running = False
        self._is_paused = False

    def _dispatch(self) -> None:
        frame = self._input_source.read()
        for stage in self._stages:
            stage(self._context, frame)

    def close(self) -> None:
        """ Closes the running pipeline """
        self.stop()

        for stage in self._stages:
            stage.close()

        self._stages.clear()
        self._input_source.close()

    def activate(self) -> None:
        """ Activates the pipeline """
        self._context.is_active = True

    def deactivate(self) -> None:
        """ Deactivates the pipeline """
        self._context.is_active = False

    def start(self) -> None:
        """ Starts input source of the pipeline """
        if not self._is_running:
            self._input_source.start()
            self._is_running = True

    def stop(self) -> None:
        """ Halts the pipeline """
        self._is_running = False
        self._input_source.stop()
        self._context.reset()

    def pause(self) -> None:
        """ Stops audio input until resume is called """
        self._is_paused = True
        self._input_source.stop()

    def resume(self) -> None:
        """ Resumes audio input after a pause """
        self._is_paused = False
        self._input_source.start()

    def run(self) -> None:
        """ Runs the pipeline to process speech and cleans up after stop is called """
        if not self._is_running:
            self.start()

        while self._is_running:
            self.step()

    def step(self) -> None:
        """ Process a single frame with the pipeline """
        self._context.event("step")
        if not self._is_paused:
            self._dispatch()

    def event(self, function: Any = None, name: Union[str, None] = None) -> Any:
        """Registers an event handler

        Args:
            function: event handler
            name: name of event handler

        Returns:
            Default event handler if a function not specified

        """
        if function:
            self._context.add_handler(
                name or function.__name__.replace("on_", ""), function
            )
        else:
            return lambda function: self.event(function, name)

    @property
    def is_running(self) -> bool:
        """ State of the pipeline """
        return self._is_running

    @property
    def context(self) -> SpeechContext:
        """ Current context """
        return self._context
