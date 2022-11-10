from setuptools import setup, find_packages
import sys, os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'collaborative_gym'))

directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'collaborative_gym', 'envs', 'assets')

setup(
    name='collaborative-gym',
    version='1.0',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['gym==0.21.0', 'pybullet @ git+https://github.com/gabriansa/bullet3.git#egg=pybullet', 'numpy==1.23.0', 'keras==2.10.0', 'tensorflow==2.10.0', 'h5py==3.7.0', 'ray[rllib]==1.13.0', 'numpngw==0.1.0', 'pyquaternion==0.9.9', 'tensorflow-probability==0.17.0'] + ['screeninfo==0.6.1' if sys.version_info >= (3, 6) else 'screeninfo==0.2'],  
    author='Gabriele Ansaldo',
    author_email='ansaldo.g@northeastern.edu',
)