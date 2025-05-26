from setuptools import setup, find_packages

setup(
    name='chunkanon',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'tqdm',
        'pydantic'
    ],
    entry_points={
        'console_scripts': [
            'chunkanon=chunkanon.cli:main'
        ]
    },
    author='Kailash R',
    description='Chunking-based K-Anonymization using Optimal Lattice Approach',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown'
)
