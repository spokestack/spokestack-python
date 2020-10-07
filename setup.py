import setuptools  # type: ignore


with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spokestack",
    version="0.0.5",
    author="Spokestack",
    author_email="support@spokestack.io",
    description="Spokestack Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spokestack/spokestack-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
