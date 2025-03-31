from setuptools import setup, find_packages

setup(
    name="lwmecps-gym-core",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "kubernetes>=28.1.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
    ],
    python_requires=">=3.8",
    author="Ivan Filianin",
    author_email="adeptvin1@gmail.com",
    description="Core package for LWMECPS Gym environment",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adeptvin1/lwmecps-gym",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 