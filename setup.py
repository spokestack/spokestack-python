import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spokestack-will-rice",
    version="0.0.1",
    author="Spokestack",
    author_email="will@spokestack.io",
    description="Spokestack Library for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/spokestack/spokestack-python",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Apache 2.0 Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
