# Spokestack Docs

## Build the docs

From the root project directory:

    cd docs
    sphinx-apidoc -f -o docs/source ../spokestack
    make clean && make html
