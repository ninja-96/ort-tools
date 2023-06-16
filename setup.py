"""
Setup script for ORT tools.
"""

from setuptools import find_packages, setup


setup(
    name='ort-tools',
    version=1.0,
    description='Tools for onnx models',
    author='Oleg Kachalov',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'onnx>=1.10',
        'onnxruntime>=1.10'
    ]
)
