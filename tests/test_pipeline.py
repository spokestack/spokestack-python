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

    pipeline.close()
    assert not pipeline.is_running


def test_dispatch():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()

    pipeline._dispatch()

    pipeline.is_managed = True

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


def test_cleanup():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.start()
    assert pipeline.is_running

    pipeline.stop()
    assert not pipeline.is_running

    # run after stopped will trigger clean up
    pipeline.run()
    assert not pipeline._stages
    assert not pipeline._input_source


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


def test_managed_getter_setter():
    stages = [
        mock.MagicMock(),
        mock.MagicMock(),
        mock.MagicMock(),
    ]
    pipeline = SpeechPipeline(mock.MagicMock(), stages=stages)

    pipeline.is_managed = True
    assert pipeline.is_managed


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

    pipeline.start()
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
    assert not pipeline.is_running

    # verify it does nothing
    pipeline.step()
    assert not pipeline.is_running

    pipeline.resume()
    assert pipeline.is_running
    pipeline.close()
