from setuptools import setup, find_packages

setup(
    name='TransitionDipoleExplorer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'logging',
    ],
    entry_points={
        'console_scripts': [
            # Define any console scripts here
        ],
    },
)
