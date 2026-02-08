from setuptools import setup, find_packages

setup(
    name="constrai",
    version="0.2.0",
    author="Ambar",
    description="Formal safety framework for AI agents with provable guarantees",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ambar-13/ConstrAI",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[],  # Zero dependencies for core
    extras_require={
        "anthropic": ["anthropic>=0.20.0"],
        "openai": ["openai>=1.0.0"],
        "dev": ["pytest>=7.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
