from unittest import mock

from spokestack.pipeline import SpeechPipeline


def test_start_stop():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()
    assert pipeline.is_running

    # test second call to start, which ignores if the pipeline is running
    pipeline.start()

    pipeline.step()

    pipeline.stop()
    assert not pipeline.is_running

    pipeline.close()


def test_dispatch():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()

    pipeline._dispatch()


def test_activate_deactivate():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()
    pipeline.activate()
    assert pipeline.context.is_active

    pipeline.deactivate()
    assert not pipeline.context.is_active


def test_events():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    @pipeline.event
    def on_speech(context):
        context.transcript = "event triggered"

    # test empty event
    pipeline.event(name="dummy_event")

    pipeline.context.event("speech")
    assert pipeline.context.transcript == "event triggered"


def test_run():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    @pipeline.event
    def on_step(context):
        pipeline.stop()

    pipeline.run()


def test_pause_resume():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()
    assert pipeline.is_running

    pipeline.step()
    pipeline.pause()
    pipeline._input_source.stop.assert_called()

    # verify it does nothing
    pipeline.step()

    pipeline.resume()
    pipeline._input_source.start.assert_called()
    pipeline.close()
