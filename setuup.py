from setuptools import setup, find_packages

setup(
    name='chunkanon',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tqdm'
    ],
    entry_points={
        'console_scripts': [
            'chunkanon=cli:main'
        ]
    },
    author='CDPG',
    description='Chunking-based K-Anonymization using Optimal Lattice Approach',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
