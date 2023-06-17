"""
Setup script for ORT tools.
"""

from setuptools import find_packages, setup
import ort_tools


setup(
    name='ort-tools',
    version=ort_tools.__version__,
    description='Tools for ONNX models',
    author='Oleg Kachalov',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'onnx>=1.10',
        'onnxruntime>=1.10'
    ]
)
