# How to contribute to Spokestack Python

Would you like to contribute to Spokestack Python? We would love to have your contributions!

## Pull Requests

We gladly welcome pull requests.

Before making any changes, we recommend opening an issue (if it doesn’t
already exist) and discussing your proposed changes. This will let us give
you advice on the proposed changes. If the changes are minor, then feel free
to make them without discussion.

## Development Process

Contributing is as simple as making a fork of the repo and adding commits. We distribute the repo with a `.pre-commit-config.yml` file that contains the necessary pre-commit hooks for formatting and tests. This file configures the [pre-commit](pre-commit.com) framework which is installed with the other Python dependencies in `requirements.txt`. Success on these pre-commit checks will ensure that your branch will build with CircleCI and meets our formatting requirements.

### Format

We use [flake8](https://flake8.pycqa.org/en/latest/) and [black](https://github.com/psf/black) for formatting. Your code will be automatically formatted to our specification by the pre-commit hook. This ensures deterministic uniformity across every addition.

### Tests

For testing we use `pytest`.

### Test Coverage

The following command will show test coverage via pytest:

    pytest --cov=spokestack
