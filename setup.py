"""
Setup script for image classification package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="image-classification-transfer-learning",
    version="1.0.0",
    author="Tushar Gupta",
    author_email="tg304429@.com",
    description="High-performance image classification using transfer learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tusharg007/image-classification-transfer-learning",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "train-classifier=scripts.train:main",
            "evaluate-classifier=scripts.evaluate:main",
            "predict-classifier=scripts.predict:main",
        ],
    },
)
