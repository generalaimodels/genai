# Terminal Service Setup

from setuptools import setup, find_packages

setup(
    name="terminal-service",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "redis>=5.0.0",
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
    ],
    python_requires=">=3.11",
)
