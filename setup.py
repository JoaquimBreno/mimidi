from setuptools import setup, find_packages

setup(
    name='midi-seq2seq',
    version='0.1.0',
    author='Your Name',
    author_email='your_email@example.com',
    description='A simplified Seq2Seq model for MIDI token embeddings.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.7.0',
        'torchvision',
        'pytorch-lightning',
        'transformers',
        'numpy',
        'pandas',
        'mido',
        'pretty_midi',
        'scipy',
        'pyyaml',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)