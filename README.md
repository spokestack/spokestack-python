<a href="https://www.spokestack.io/docs/python/getting-started" title="Getting Started with Spokestack + Python"><img src="images/spokestack-python.png" alt="Spokestack Python"></a>

[![GitHub license](https://img.shields.io/github/license/spokestack/spokestack-python?style=for-the-badge&color=2F5BEA&label-color=2D2D2D)](https://github.com/spokestack/spokestack-python/blob/master/LICENSE.txt)
[![CircleCI](https://img.shields.io/badge/circleci-passing-blue?style=for-the-badge&color=2F5BEA&logo=circleci&label-color=2D2D2D&logoColor=white)](https://circleci.com/gh/spokestack/spokestack-python)
[![PyPI version](https://img.shields.io/pypi/v/spokestack?style=for-the-badge&color=2F5BEA&logo=pypi&label-color=2D2D2D&logoColor=white)](https://badge.fury.io/py/spokestack)
[![Coverage Status](https://img.shields.io/coveralls/github/spokestack/spokestack-python/master?style=for-the-badge&color=2F5BEA&logo=coveralls&label-color=2D2D2D&logoColor=white)](https://coveralls.io/github/spokestack/spokestack-python?branch=master)

Welcome to Spokestack Python! This library is intended for developing [voice interfaces](https://www.spokestack.io/docs/concepts) in Python. This can include anything from [Raspberry Pi](https://www.raspberrypi.org/) applications like traditional smart speakers to [Django](https://www.djangoproject.com/) web applications. _Anything_ built in [Python](https://www.python.org/) can be given a voice interface.

## Get Started

### Installation with pip

Once system dependencies have been satisfied, you can install the library with the following.

```shell
pip install spokestack
```

### Install Tensorflow

This library requires a way to run [TFLite](https://www.tensorflow.org/lite) models. There are two ways to add this ability. The first is installing the full [Tensorflow](https://www.tensorflow.org/) library.

The full Tensorflow package is installed with the following:

```shell
pip install tensorflow
```

#### TFLite Interpreter (Embedded Devices)

In use cases where you require a small footprint, such as on a Raspberry Pi or similar embedded devices, you will want to install the TFLite Interpreter.

```shell
pip install --extra-index-url https://google-coral.github.io/py-repo/ tflite_runtime
```

### System Dependencies (Optional)

If you are unable to install the wheel, you may have to install some system dependencies for audio input and output.

#### macOS

```shell
brew install lame portaudio
```

#### Debian/Ubuntu

```shell
sudo apt-get install portaudio19-dev libmp3lame-dev
```

#### Windows

We currently do not support Windows 10 natively, and recommend you install [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10) with the Debian dependencies. However, if you would like to work on native Windows support, we will gladly accept pull requests.

Another potential avenue for using `spokestack` on Windows 10 is from [anaconda](https://www.anaconda.com/). This is without support for Text To Speech (TTS) though due to the Lame dependency. PortAudio, on the other hand, can be installed via `conda`.

```shell
conda install portaudio
```

## Usage

### Profiles

The quickest way to start using `spokestack` is by using one of the pre-configured pipeline instances. We offer several of these Profiles, which fit many general use cases.

```python
from spokestack.profile.wakeword_asr import WakewordSpokestackASR


pipeline = WakewordSpokestackASR.create(
    "spokestack_id", "spokestack_secret", model_dir="path_to_wakeword_model"
)
```

### Speech Pipeline

If you would like fine-grained control over what is included in the pipeline, you can use `SpeechPipeline`. This is the module that ties together VAD (voice activity detection), wakeword, and ASR (automated speech detection). The VAD listens to a frame of audio captured by the input device to determine if speech is present. If it is, the wakeword model processes subsequent frames of audio looking for the keyword it has been trained to recognize. If the keyword is found, the pipeline is activated and performs speech recognition, converting the subsequent audio into a transcript. The `SpeechPipeline` is initialized like this:

```python
from spokestack.activation_timeout import ActivationTimeout
from spokestack.io.pyaudio import PyAudioInput
from spokestack.pipeline import SpeechPipeline
from spokestack.vad.webrtc import VoiceActivityDetector
from spokestack.wakeword.tflite import WakewordTrigger
from spokestack.asr.spokestack.speech_recognizer import SpeechRecognizer

mic = PyAudioInput()
vad = VoiceActivityDetector()
wake = WakewordTrigger("path_to_tflite_model")
asr = SpeechRecognizer("spokestack_id", "spokestack_secret")
timeout = ActivationTimeout()


pipeline = SpeechPipeline(mic, [vad, wake, asr, timeout])
pipeline.run()
```

Now that the pipeline is running, it becomes important to access the results from processes at certain events. For example, when speech is recognized there is a `recognize` event. These events allow code to be executed outside the pipeline in response. The process of registering a response is done with a pipeline callback, which we will cover in the next section.

#### Pipeline Callbacks

Pipeline callbacks allow additional code to be executed when a speech event is detected. For example, we can print when the pipeline is activated by registering a function with the `pipeline.event` decorator.

```python
@pipeline.event
def on_activate(context):
    print(context.is_active)
```

One of the most important use cases for a pipeline callback is accessing the ASR transcript for additional processing by the NLU. The transcript is accessed with the following:

```python
@pipeline.event
def on_recognize(context):
    print(context.transcript)
```

### Natural Language Understanding (NLU)

Natural Language Understanding turns an utterance into structured data a machine can act on. For our purposes, this is joint intent detection and slot filling. You can read more about the concepts [here](https://www.spokestack.io/docs/concepts/nlu). We like to think of intents as the action a user desires from an application, and slots as the optional arguments to fulfill the requested action. Our NLU model is initialized like this:

```python
from spokestack.nlu.tflite import TFLiteNLU

nlu = TFLiteNLU("path_to_tflite_model")
```

Now that the NLU is initialized we can go ahead and add that part to the callback.

```python
@pipeline.event
def on_recognize(context):
    results = nlu(context.transcript)
```

### Text To Speech (TTS)

Text To Speech, as the name implies, converts text into spoken audio. This the method for giving your application a voice. We provide one TTS voice for free when you sign up for a Spokestack account, but you can contact us to train a truly custom voice. The TTS API keys are the same as `SpeechRecognizer`. The basic TTS initialization is the following:

```python
from spokestack.tts.manager import TextToSpeechManager
from spokestack.tts.clients.spokestack import TextToSpeechClient
from spokestack.io.pyaudio import PyAudioOutput

client = TextToSpeechClient("spokestack_id", "spokestack_secret")
output = PyAudioOutput()
manager = TextToSpeechManager(client, output)
manager.synthesize("welcome to spokestack")
```

To demonstrate a simple TTS callback let's set up something that reads back what the ASR recognized:

```python
@pipeline.event
def on_recognize(context):
    manager.synthesize(context.transcript)
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

Copyright 2021 Spokestack, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License [here](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
