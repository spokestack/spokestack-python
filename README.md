# Spokestack Python

<p align="center">
  <img width="100" height="100" src="images/spokestack.png">
</p>

[![GitHub license](https://img.shields.io/github/license/spokestack/spokestack-python)](https://github.com/spokestack/spokestack-python/blob/master/LICENSE.txt)
[![CircleCI](https://circleci.com/gh/pylon/streamp3.svg?style=shield)](https://circleci.com/gh/spokestack/spokestack-python)
[![PyPI version](https://badge.fury.io/py/spokestack.svg)](https://badge.fury.io/py/spokestack)
[![Coverage Status](https://coveralls.io/repos/github/spokestack/spokestack-python/badge.svg?branch=master)](https://coveralls.io/github/spokestack/spokestack-python?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![GitHub stars](https://img.shields.io/github/stars/spokestack/spokestack-python?style=social)](https://github.com/spokestack/spokestack-python/stargazers)
![GitHub watchers](https://img.shields.io/github/watchers/spokestack/spokestack-python?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/spokestack?style=social)

Welcome to Spokestack Python! This library is intended for developing general purpose [voice interfaces](https://en.wikipedia.org/wiki/Voice_user_interface). What is a general purpose voice interface you ask? In our minds, a general purpose voice interface is a voice interface that can be easily integrated with virtually any application that uses [Python](https://www.python.org/). This includes [raspberrypi](https://www.raspberrypi.org/) applications such as a traditional smart speaker to a [Django](https://www.djangoproject.com/) web application. Literally, anything built in [Python](https://www.python.org/) can be given a voice interface.

## Get Started

### System Dependencies

There are some system dependencies that need to be downloaded in order to install spokestack via pip.

#### macOS

```shell
brew install lame portaudio
```

#### Debian/Ubuntu

```shell
sudo apt-get install portaudio19-dev libmp3lame-dev
```

#### Windows

We currently do not support Windows 10 natively, and recommend you install [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with the Debian dependencies. However, if you would like to work on native Windows support we will glady accept pull requests.

Another potential avenue for using `spokestack` on Windows 10 is from [anaconda](https://www.anaconda.com/). This is without support for Text To Speech (TTS) though due to the Lame dependency. PortAudio, on the other hand, can be installed via `conda`.

```shell
conda install portaudio
```

### Installation with pip

Once system dependencies have been satisfied, you can install the library with the following.

```shell
pip install spokestack
```

## Development

### Setup

We use `pyenv` for virtual environments. Below you will find the step-by-step commands to install a virtual environment.

```shell
pyenv install 3.8.6
pyenv virtualenv 3.8.6 spokestack
pyenv local spokestack
pip install -r requirements.txt
```

### Install Tensorflow

This library requires a way to run [TFLite](https://www.tensorflow.org/lite) models. There are two primary ways to add this ability. The first, is installing the full [Tensorflow](https://www.tensorflow.org/) library. In most instances you should be able to install this with `pip`, but in the case of any installation issues follow the installation [guide](https://www.tensorflow.org/install).

#### Tensorflow

The full Tensorflow package is installed with the following:

```shell
pip install tensorflow
```

#### TFLite Interpreter (Embedded Devices)

In use cases where you require a small footprint, such as on a raspberry pi or similar embedded devices, you will want to install the TFLite Interpreter. You can install it for your platform by following the instructions at [TFLite Interpreter](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter).

### Audio Input

We offer a [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) interface for audio input. You can initialize it like this:

```python
from spokestack.io.pyaudio import PyAudioInput

mic = PyAudioInput()
mic.start()
frame = mic.read()
```

### Voice Activity Detection (VAD)

Voice activity detection listens to a frame of audio and determines if a voice is present. Our VAD uses [WebRTC](https://github.com/wiseman/py-webrtcvad). To initialize the VAD, add the following:

```python
from spokestack.vad.webrtc import VoiceActivityTrigger

vad = VoiceActivityTrigger()
```

### Wakeword Detection

Also known as keyword spotting (kws), wakeword is the way to get your applications attention with voice. This is normally through a specific keyword or phrase. The default wakeword for this library is "Spokestack". Wakeword can be added like this:

```python
from spokestack.wakeword.tflite import WakewordDetector

wake = WakewordDetector("path_to_tflite_model")
```

### Automatic Speech Recognition (ASR)

Automatic Speech Recognition is a technique used to turn speech into a transcript. You can initialize our cloud ASR with:

```python
from spokestack.asr.spokestack.speech_recognizer import SpeechRecognizer

asr = SpeechRecognizer("spokestack_id", "spokestack_secret")
```

### Speech Pipeline

The Speech Pipeline is the component that ties together VAD, Wakeword, and ASR. Essentially, this component listens to a frame of audio to determine if speech is present. If speech is detected, the Wakeword model processes the subsequent frames of audio looking for the specific keyword. If the keyword is found, the pipeline is activated and converts the following audio into a transcript. The Speech Pipeline is established like this from the previous initialized components:

```python
from spokestack.pipeline import SpeechPipeline

pipeline = SpeechPipeline(mic, [vad, wake, asr])
pipeline.start()
pipeline.run()
```

#### Speech Context

Speech context manages the state of the pipeline. For example, when the an activation event is triggered, the `is_active` property in the Speech Context is set to `True`. The initialization of `SpeechContext` is handled by the pipeline, but if required elsewhere, can be initialized like this:

```python
from spokestack.context import SpeechContext

context = SpeechContext()
```

#### Pipeline Callbacks

Pipeline callbacks allow additional code to be executed when a speech event is detected. For example, we can print when the pipeline is activated by registering a function with the `pipeline.event` decorator.

```python
@pipeline.event
def on_activate(context):
    print(context.is_active)
```

### Natural Language Understanding (NLU)

Natural Language Understanding turns an utterance into a machine readable format. For our purposes, this is joint intent detection and slot filling. We like to think of intents as the action a user desires from an application, and slots as the optional arguments to fulfill the requested action. Our NLU model is initialized like this:

```python
from spokestack.nlu.tflite import TFLiteNLU

nlu = TFLiteNLU("path_to_tflite_model")
```

### Text To Speech (TTS)

Text To Speech, as the name implies, converts a text into spoken audio. This the method for giving your application a voice. We have a default voice included when you sign up for a Spokestack account. However, you can contact us to set up a truly custom voice. The TTS API keys are the same as `SpeechRecognizer`. The basic TTS initialization is the following:

```python
from spokestack.tts.manager import TextToSpeechManager
from spokestack.tts.clients.spokestack import TextToSpeechClient
from spokestack.io.pyaudio import PyAudioOutput

client = TextToSpeechClient("spokestack_id", "spokestack_secret")
output = PyAudioOutput()
manager = TextToSpeechManager(client, output)
manager.synthesize("welcome to spokestack")
```

## Documentation

### Build the docs

From the root project directory:

```shell
cd docs
make clean && make html
```

## Deployment

This project is distributed using [PyPI](https://pypi.org/). The following is the command to build for installation.

```shell
python setup.py clean --all; rm -r ./dist
python setup.py sdist bdist_wheel
```

[Twine](https://twine.readthedocs.io/en/latest/) is used to upload the wheel and source distribution.

```shell
twine upload dist/*
```

## License

Copyright 2020 Spokestack, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License [here](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
