from setuptools import setup, find_packages

setup(
    name='DownstreamTasks',
    version='0.1.0',
    packages=find_packages(),
    description='Downstream Tasks with the DUE model from COATI on different datasets and the Linear Probing part of the CLAMP model for compound activity prediction. Comparison of four models: COATI grande closed, COATI autoregressive only, COATI2, and CLAMP.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy',
        'rdkit-pypi; sys_platform != "win32"',  # RDKit is typically installed via Conda, pip version is limited
        'boto3',
        'pytorch-ignite',
        'gpytorch',
        'git+https://github.com/y0ast/DUE.git',
        'git+https://github.com/ml-jku/clamp'
    ],
    python_requires='>=3.6',
)
