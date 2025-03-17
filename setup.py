from setuptools import setup, find_packages

setup(
    name="lwmecps_gym",
    version="0.0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "prometheus-client",
    ],
    python_requires=">=3.10",
) 