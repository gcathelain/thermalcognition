import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="thermalcognition",  # Replace with your own username
    version="0.0",
    author="Guillaume Cathelain",
    author_email="guillaume.cathelain@gmail.com",
    description="Process thermal videos for emotion recognition.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gcathelain/thermalcognition",
    packages=setuptools.find_packages(where='src'),
    package_dir={'': 'src'},
    # entry_points=dict(console_scripts=["process=fealing.process:main"]), #no possible argument...
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)
