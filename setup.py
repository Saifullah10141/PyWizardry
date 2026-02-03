from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PyWizardry",
    version="1.0.2",
    description="A magical collection of 150+ Python utilities for modern development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saif",
    author_email="saifullahanwar00040@gmail.com",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[],
    extras_require={
        "full": [
            "pillow>=10.0.0",
            "psutil>=5.9.0",
            "aiohttp>=3.9.0",
        ],
        "extras": [
            "pycryptodome>=3.19.0",
            "pydantic>=2.5.0",
        ]
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://pywizardry.vercel.app",
    project_urls={
        "Documentation": "https://pywizardry.vercel.app/docs",
        "Source Code": "https://github.com/Saifullah10141/pywizardry",
    },
    keywords=[
        "utilities",
        "tools",
        "development",
        "productivity",
        "python",
        "library",
        "magic",
        "wizardry",
    ],
)