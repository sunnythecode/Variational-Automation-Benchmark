from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "./README.md"), encoding="utf-8") as f:
    lines = f.readlines()
lines = [x for x in lines if ".png" not in x]
long_description = "".join(lines)

setup(
    name="libero",
    packages=find_packages(where="libero"),
    package_dir={"": "libero"},
    install_requires=[],
    eager_resources=["*"],
    include_package_data=True,
    python_requires=">=3.8",
    description="Variational Automation Benchmark (VAB): unified-config robotic manipulation benchmark",
    author="Eric Chen",
    version="0.2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
