from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['tensorflow>=2.4.1',
                     'matplotlib>=3.5.1',
                     'tqdm>=4.64.0',
                     'tensorboard>=2.9.0',
                     'torch-scatter>=2.0.9',
                     'torch-sparse>=0.6.13',
                     'torch-cluster>=1.6.0',
                     'torch-spline-conv>=1.2.1',
                     'torch-geometric>=2.0.4']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    package_data={'': ['dataset/processed/*.pt']},
    include_package_data=True,
    description='IAS Brenda Gutierrez 2022'
)

# data_files=[('dataset', ['processed/*.pt'])],

