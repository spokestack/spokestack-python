# spokestack-python

## Status

[![CircleCI](https://circleci.com/gh/pylon/streamp3.svg?style=shield)](https://circleci.com/gh/spokestack/spokestack-python)
[![PyPI version](https://badge.fury.io/py/spokestack.svg)](https://badge.fury.io/py/spokestack)
[![Coverage Status](https://coveralls.io/repos/github/spokestack/spokestack-python/badge.svg?branch=master)](https://coveralls.io/github/spokestack/spokestack-python?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/spokestack-python/badge/?version=latest)](https://spokestack-python.readthedocs.io/en/latest/?badge=latest)

## Get Started

Add voice capability to your projects with one line.

    pip install spokestack

## Development

### Setup

We use `pyenv` for virtual environments. Below you will find the step-by-step commands to install a virtual environment.

    pyenv install 3.7.6
    pyenv virtualenv 3.7.6 spokestack
    pip install -r requirements.txt

### TFLite Interpreter

In addition to the Python dependencies, you will need to install the TFLite Interpreter. You can install it for your platform by following the instructions at [TFLite Interpreter](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter).
**Note:** this is not the full [Tensorflow](https://www.tensorflow.org/) package.

## Documentation

### Build the docs

From the root project directory:

    cd docs
    sphinx-apidoc -f -o docs/source ../spokestack
    make clean && make html

## Deployment

This project is distributed using [PyPI](https://pypi.org/). The following is the command to build for installation.

    python setup.py sdist bdist_wheel

[Twine](https://twine.readthedocs.io/en/latest/) is used to upload the wheel and source distribution.

    twine upload dist/*

## License

Copyright 2020 Spokestack, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
