from setuptools import setup, find_packages

setup(
    name='TransitionDipoleAnalyser',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'colorlog',
    ],
    entry_points={
        'console_scripts': [
            # Define any console scripts here
        ],
    },
)